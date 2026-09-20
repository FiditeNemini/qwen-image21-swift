// MLXEngine package over the QwenImage21 core — `textToImage` + `imageEdit` on ONE model
// (Qwen-Image-2.1 is a unified generator/editor; the surfaces dispatch inside `run(_:)`).
//
// ⚠️ LICENCE: Qwen RESEARCH License — research / evaluation only, no commercial use. Declared
// package-locally as `LicenseRef-Qwen-Research` and deliberately NOT allowlisted, so the default
// `.permissiveOnly` policy refuses (`.blocking`) or flags (`.advisory`) it. This package is a
// research tier, never a shipping default (AB-D-0085).
//
// Core facts (PORTING-SPEC.md): 7B single-stream block-causal DiT with a prefix KV cache, 64-ch
// 16x RGBA VAE, Qwen3-VL-8B conditioner (byte-identical to Qwen/Qwen3-VL-8B-Instruct, loaded from
// that snapshot), FlowMatchEuler 256/8192 · 0.5/0.9 · terminal 0.02, no guidance by default,
// 40 steps, outputs RGBA PNG (native transparency).

import Foundation
import MLX
import MLXToolKit
import QwenImage21

extension SPDXLicense {
    /// Qwen RESEARCH LICENSE AGREEMENT (release 2026-09-20): §1(i) non-commercial = research or
    /// evaluation only; §2(b) commercial use requires a separate licence from Tongyi. Non-SPDX,
    /// `LicenseRef-` convention. Intentionally absent from `permissiveAllowlist`.
    public static let qwenResearch: SPDXLicense = "LicenseRef-Qwen-Research"
}

/// Init-time configuration (C9): the two snapshot roots and generation defaults.
public struct QwenImage21Configuration: PackageConfiguration, ModelStorable, QuantConfigured, WeightSourcing {
    /// Qwen/Qwen-Image-2.1 root (`transformer/`, `vae/`, `processor/`, `scheduler/`). Empty = store.
    public var snapshotPath: String
    /// Qwen/Qwen3-VL-8B-Instruct root (weights + tokenizer). Empty = store.
    public var textEncoderPath: String
    public var defaultSteps: Int
    /// 1.0 — Qwen-Image-2.1 is meant to be sampled without guidance; > 1 with a negative prompt
    /// runs plain true CFG (two DiT forwards per step).
    public var defaultTrueCFGScale: Float
    /// `output_resolution`: side length whose square is the output area (T2I default size, and
    /// the area condition images are resized to). The model card recommends 2048 for T2I.
    public var defaultOutputResolution: Int
    /// Keep the ~17 GB Qwen3-VL encoder resident between requests (big-RAM tiers).
    public var keepEncoderResident: Bool
    /// Prefix KV cache across denoise steps (the reference default).
    public var useKVCache: Bool
    public var modelsRootDirectory: URL?

    /// bf16 DiT + fp32 VAE — the only tier for now.
    public var quant: Quant { .bf16 }

    public static let repo = "Qwen/Qwen-Image-2.1"
    /// The 2.1 `text_encoder/` is byte-identical to this repo (750/750 tensors); we materialise
    /// the stock snapshot instead of a second copy. ⚠ Fleet durability policy: a shipped package
    /// must source from a namespace we control — this is a research tier and does not ship.
    public static let textEncoderRepo = "Qwen/Qwen3-VL-8B-Instruct"

    public init(
        snapshotPath: String = "",
        textEncoderPath: String = "",
        defaultSteps: Int = 40,
        defaultTrueCFGScale: Float = 1.0,
        defaultOutputResolution: Int = 1024,
        keepEncoderResident: Bool = false,
        useKVCache: Bool = true,
        modelsRootDirectory: URL? = nil
    ) {
        self.snapshotPath = snapshotPath
        self.textEncoderPath = textEncoderPath
        self.defaultSteps = defaultSteps
        self.defaultTrueCFGScale = defaultTrueCFGScale
        self.defaultOutputResolution = defaultOutputResolution
        self.keepEncoderResident = keepEncoderResident
        self.useKVCache = useKVCache
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// Fresh-machine sources (MAT), split by role. The DiT/VAE/processor/scheduler come from the
    /// 2.1 repo; the conditioner from the stock Qwen3-VL-8B-Instruct repo.
    public var weightSources: [WeightSource] {
        [
            WeightSource(role: "transformer", repo: Self.repo, revision: "main", matching: ["transformer/*"]),
            WeightSource(role: "vae", repo: Self.repo, revision: "main", matching: ["vae/*"]),
            WeightSource(role: "pipeline-config", repo: Self.repo, revision: "main",
                         matching: ["model_index.json", "scheduler/*", "processor/*", "LICENSE"]),
            WeightSource(role: "text-encoder", repo: Self.textEncoderRepo, revision: "main",
                         matching: ["*.safetensors", "*.json", "merges.txt"]),
        ]
    }

    static func hasTransformer(_ path: String) -> Bool {
        !path.isEmpty && FileManager.default.fileExists(atPath: URL(fileURLWithPath: path).appendingPathComponent("transformer").path)
    }

    static func hasTextEncoder(_ path: String) -> Bool {
        !path.isEmpty && FileManager.default.fileExists(atPath: URL(fileURLWithPath: path).appendingPathComponent("config.json").path)
    }

    /// Explicit paths satisfy their roles; everything else resolves through the store layout.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        var missing = defaultMissingWeightSources(storeRoot: storeRoot)
        if Self.hasTransformer(snapshotPath) { missing.removeAll { $0.repo == Self.repo } }
        if Self.hasTextEncoder(textEncoderPath) { missing.removeAll { $0.repo == Self.textEncoderRepo } }
        return missing
    }

    static func resolve(explicit: String, repo: String, storeRoot: URL?, marker: String) -> URL? {
        if !explicit.isEmpty { return URL(fileURLWithPath: explicit) }
        let store = ModelStore(root: storeRoot)
        let fm = FileManager.default
        if let flat = store.directory(for: repo), fm.fileExists(atPath: flat.appendingPathComponent(marker).path) {
            return flat
        }
        if let snap = store.snapshotDirectory(for: repo, revision: "main"),
           fm.fileExists(atPath: snap.appendingPathComponent(marker).path) {
            return snap
        }
        return store.directory(for: repo)
    }

    /// Store-resolved 2.1 snapshot root (explicit path wins, then flat store, then hub snapshot).
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        Self.resolve(explicit: snapshotPath, repo: Self.repo, storeRoot: storeRoot, marker: "transformer")
    }

    /// Store-resolved Qwen3-VL-8B-Instruct root.
    public func resolvedTextEncoderDirectory(storeRoot: URL?) -> URL? {
        Self.resolve(explicit: textEncoderPath, repo: Self.textEncoderRepo, storeRoot: storeRoot, marker: "config.json")
    }

    private enum CodingKeys: String, CodingKey {
        case snapshotPath, textEncoderPath, defaultSteps, defaultTrueCFGScale, defaultOutputResolution,
             keepEncoderResident, useKVCache
    }
}

public enum QwenImage21PackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case imageDecode
    case pngEncode

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "Qwen-Image-2.1 snapshot not readable at \(p)."
        case .imageDecode: return "Could not decode an input image."
        case .pngEncode: return "PNG encoding failed."
        }
    }
}

@InferenceActor
public final class QwenImage21Package: ModelPackage {
    public typealias Configuration = QwenImage21Configuration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: Qwen RESEARCH License (non-commercial) — package-local LicenseRef, NOT allowlisted,
            // so `.permissiveOnly` refuses or flags this package by construction (AB-D-0085).
            // C8: port code MIT.
            license: LicenseDeclaration(weightLicense: .qwenResearch, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "Qwen/Qwen-Image-2.1", revision: "main", tier: 3),
            requirements: RequirementsManifest(
                // ESTIMATE, not yet a measured split (AB-T-0154 acceptance #3 owes the QI21_MEMBENCH
                // numbers): resident = DiT bf16 14.23 GB + VAE fp32 1.35 GB = 15.6 GB; the Qwen3-VL
                // encoder (~17.5 GB bf16) is a per-request TRANSIENT evicted before the denoise peak,
                // so it lands in the activation term. First bf16 1024²/40-step render: MLX peak
                // 31.2 GB → activation ≈ 15.6 GB → 19 GB at +20%. Smoke MLX-peak, not in-app phys.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 15_600_000_000, peakActivationBytes: 19_000_000_000)
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                // Conservative until measured on a 36–48 GB tier; the estimate says it should fit .pro.
                chipFloor: .max
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "qwen-image-2.1",
                    summary: "Qwen-Image-2.1 text-to-image (7B single-stream block-causal DiT + Qwen3-VL-8B "
                        + "conditioning, 64-ch RGBA VAE): 40 FlowMatch Euler steps, no guidance by default, "
                        + "native transparency (RGBA PNG output; prompt 'This is an RGBA image with "
                        + "transparency. … The image has alpha channel and the background is transparent.'), "
                        + "sizes divisible by 32, 2048² native. RESEARCH LICENCE — non-commercial only.",
                    modes: []
                ),
                IEditContract.descriptor(
                    name: "qwen-image-2.1",
                    summary: "Qwen-Image-2.1 instruction editing with up to 10 reference images "
                        + "(<image1>…<imageN> in the prompt), identity-preserving edits, transparent-layer "
                        + "editing and subject extraction; output follows the LAST image's aspect at "
                        + "output_resolution² (default 1024²). RESEARCH LICENCE — non-commercial only.",
                    modes: []
                ),
            ]
        )
    }

    private let configuration: Configuration
    private var generator: QwenImage21Generator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }
        // Materialisation is engine-executed (contract 1.24) before load(); this is the offline
        // backstop so absent weights fail legibly.
        guard let snapshot = configuration.resolvedSnapshotDirectory(storeRoot: configuration.modelsRootDirectory),
              FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("transformer").path)
        else {
            throw QwenImage21PackageError.unreadableSnapshot(
                configuration.snapshotPath.isEmpty ? Configuration.repo : configuration.snapshotPath)
        }
        guard let textEncoder = configuration.resolvedTextEncoderDirectory(storeRoot: configuration.modelsRootDirectory),
              FileManager.default.fileExists(atPath: textEncoder.appendingPathComponent("config.json").path)
        else {
            throw QwenImage21PackageError.unreadableSnapshot(
                configuration.textEncoderPath.isEmpty ? Configuration.textEncoderRepo : configuration.textEncoderPath)
        }
        // DiT (bf16) + VAE (fp32) stay resident; the encoder loads per request and is evicted
        // before the denoise peak unless `keepEncoderResident`.
        let transformer = try QwenImage21Weights.loadTransformer(
            directory: snapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
        let vae = try QwenImage21Weights.loadVAE(directory: snapshot.appendingPathComponent("vae"), dtype: .float32)
        let generator = QwenImage21Generator(
            encoderProvider: { try await QwenImage21PromptEncoder.load(qwenDir: textEncoder, dtype: .bfloat16) },
            transformer: transformer, vae: vae, keepEncoderResident: configuration.keepEncoderResident)
        generator.warmup()
        self.generator = generator
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint is the FIRST act of run(), before notLoaded validation. Mid-run
        // cadence lives in the core (post-encode seam, per-denoise-step checkpoint, pre-decode
        // seam), rethrowing CancellationError unchanged.
        try Task.checkCancellation()
        guard let generator else { throw PackageError.notLoaded }

        let prompt: String
        let negative: String?
        let images: [QwenImage21RGBAImage]
        let width: Int?, height: Int?, steps: Int, seed: UInt64
        let cfg: Float
        switch request.capability {
        case .textToImage:
            guard let t2i = request as? T2IRequest else { throw PackageError.unsupportedCapability(request.capability) }
            prompt = t2i.prompt; negative = t2i.negativePrompt; images = []
            width = t2i.width; height = t2i.height
            steps = t2i.steps ?? configuration.defaultSteps
            seed = t2i.seed ?? 0
            cfg = t2i.guidanceScale.map(Float.init) ?? configuration.defaultTrueCFGScale
        case .imageEdit:
            guard let edit = request as? IEditRequest else { throw PackageError.unsupportedCapability(request.capability) }
            guard !edit.images.isEmpty else { throw QwenImage21PackageError.imageDecode }
            prompt = edit.prompt; negative = edit.negativePrompt
            images = try edit.images.map { img in
                do { return try QwenImage21PNG.read(data: img.data) } catch { throw QwenImage21PackageError.imageDecode }
            }
            width = edit.width; height = edit.height
            steps = edit.steps ?? configuration.defaultSteps
            seed = edit.seed ?? 0
            cfg = edit.guidanceScale.map(Float.init) ?? configuration.defaultTrueCFGScale
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        let result = try await generator.generate(
            prompt: prompt, images: images, negativePrompt: negative, trueCFGScale: cfg,
            width: width, height: height, outputResolution: configuration.defaultOutputResolution,
            steps: steps, seed: seed, useKVCache: configuration.useKVCache,
            progress: { step, total in RunProgress.report(.denoise, step: step, totalSteps: total) })

        try Task.checkCancellation()
        let png: Data
        do { png = try QwenImage21PNG.pngData(result.image) } catch { throw QwenImage21PackageError.pngEncode }
        let artifact = Image(format: .png, data: png, width: result.image.width, height: result.image.height)
        return request.capability == .textToImage ? T2IResponse(image: artifact) : IEditResponse(image: artifact)
    }
}

extension QwenImage21Package {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration { .of(QwenImage21Package.self) }
}
