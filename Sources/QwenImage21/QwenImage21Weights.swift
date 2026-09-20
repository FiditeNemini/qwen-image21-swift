// Weight loading for Qwen-Image-2.1: the diffusers `transformer/` and `vae/` safetensors load
// with their own key names (two-way strict), plus the small renames noted inline.

import Foundation
import MLX
import MLXNN

public enum QwenImage21Weights {

    static func loadAllArrays(directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw QwenImage21Error.loading("no .safetensors under \(directory.path)") }
        var merged: [String: MLXArray] = [:]
        for f in files { merged.merge(try MLX.loadArrays(url: f)) { a, _ in a } }
        return merged
    }

    /// `modulation.1.weight` (Sequential(SiLU, Linear)) -> `modulation.weight`.
    static func sanitizeDiTKey(_ k: String) -> String {
        k.replacingOccurrences(of: "modulation.1.", with: "modulation.")
    }

    /// Load the DiT from the diffusers `transformer/` snapshot folder (bf16 as stored).
    public static func loadTransformer(directory: URL, dtype: DType = .bfloat16) throws -> QwenImage21Transformer2DModel {
        let model = QwenImage21Transformer2DModel()
        var weights: [String: MLXArray] = [:]
        for (k, v) in try loadAllArrays(directory: directory) { weights[sanitizeDiTKey(k)] = v.asType(dtype) }
        try verifyAndLoad(model: model, weights: weights, label: "DiT-2.1")
        return model
    }

    /// upstream `resample = Sequential(Upsample|ZeroPad2d, Conv2d)` -> index 1; ours `[Conv2d]` index 0.
    static func sanitizeVAEKey(_ k: String) -> String {
        k.replacingOccurrences(of: ".resample.1.", with: ".resample.0.")
    }

    /// Load the VAE from the diffusers `vae/` snapshot folder (fp32 as stored; bf16 optional).
    public static func loadVAE(directory: URL, dtype: DType = .float32) throws -> AutoencoderKLQwenImage21 {
        let vae = AutoencoderKLQwenImage21()
        var state: [String: MLXArray] = [:]
        for (rawKey, raw) in try loadAllArrays(directory: directory) {
            let k = sanitizeVAEKey(rawKey)
            var v = raw
            if k.hasSuffix("gamma") {  // (C,1,1,1) / (C,1,1) -> (C)
                v = v.reshaped([v.dim(0)])
            } else if v.ndim == 4 {  // Conv2d PT (O,I,kH,kW) -> MLX (O,kH,kW,I)
                v = v.transposed(0, 2, 3, 1)
            } else if v.ndim == 5 {
                throw QwenImage21Error.loading("unexpected 5-D conv weight \(rawKey) — 2.1's VAE is 2-D")
            }
            state[k] = v.asType(dtype)
        }
        try verifyAndLoad(model: vae, weights: state, label: "VAE-2.1")
        vae.weightDtype = dtype
        return vae
    }

    /// Two-way strict load: every module key filled AND every checkpoint key consumed.
    public static func verifyAndLoad(model: Module, weights: [String: MLXArray], label: String) throws {
        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys)
        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw QwenImage21Error.loading(
                "\(label): checkpoint missing \(missing.count) module keys, e.g. " + missing.prefix(4).joined(separator: ", "))
        }
        let unused = fileKeys.subtracting(moduleKeys).sorted()
        guard unused.isEmpty else {
            throw QwenImage21Error.loading(
                "\(label): \(unused.count) unconsumed checkpoint keys, e.g. " + unused.prefix(4).joined(separator: ", "))
        }
        // shape check before update (a mismatched shape would otherwise surface as a cryptic broadcast error)
        let shapes = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        for (k, v) in weights where shapes[k] != v.shape {
            throw QwenImage21Error.loading("\(label): shape mismatch at \(k): checkpoint \(v.shape) vs module \(shapes[k] ?? [])")
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
    }
}
