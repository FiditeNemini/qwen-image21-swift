// QwenImage21Gate — parity gates against the oracle goldens + a generate CLI.
//
//   QwenImage21Gate --sched <goldensDir>
//   QwenImage21Gate --attn-probe
//   QwenImage21Gate --vae <weightsRoot> <goldensDir>
//   QwenImage21Gate --encoder <qwenDir> <goldensDir> [--tokenizer <dir>]
//   QwenImage21Gate --dit <weightsRoot> <goldensDir> [--case name]
//   QwenImage21Gate --generate <weightsRoot> <qwenDir> --prompt "..." [--image p.png]... [--size N]
//                   [--out-res N] [--steps N] [--seed N] [--cfg S] [--neg "..."] [--out out.png]
//                   [--no-cache] [--fp32-vae] [--keep-encoder]
// Parity gates run fp32 on the CPU stream (the fleet's regime); generate runs bf16 on the GPU.

import Foundation
import MLX
import QwenImage21

setbuf(stdout, nil)  // line-by-line logs when redirected to a file

// MARK: - helpers

struct Cmp { let cos: Float; let maxAbs: Float; let relMax: Float; let shape: [Int] }

func compare(_ a: MLXArray, _ b: MLXArray) -> Cmp {
    let af = a.asType(.float32).flattened()
    let bf = b.asType(.float32).flattened()
    precondition(af.size == bf.size, "shape mismatch \(a.shape) vs \(b.shape)")
    let dot = sum(af * bf).item(Float.self)
    let na = sqrt(sum(af * af)).item(Float.self)
    let nb = sqrt(sum(bf * bf)).item(Float.self)
    let diff = abs(af - bf)
    let maxAbs = diff.max().item(Float.self)
    let refMax = abs(bf).max().item(Float.self)
    return Cmp(cos: dot / max(na * nb, 1e-30), maxAbs: maxAbs, relMax: maxAbs / max(refMax, 1e-30), shape: a.shape)
}

nonisolated(unsafe) var failures = 0
func report(_ name: String, _ c: Cmp, cosGate: Float = 0.9999, relGate: Float = 2e-2) {
    let ok = c.cos >= cosGate && c.relMax <= relGate
    if !ok { failures += 1 }
    print(String(format: "  %@ %-34@ cos %.8f  maxAbs %.3e  relMax %.3e  %@",
                 ok ? "PASS" : "FAIL", name, c.cos, c.maxAbs, c.relMax, "\(c.shape)"))
}

func loadJSON(_ url: URL) throws -> [String: Any] {
    try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
}

func arg(_ flag: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: flag), i + 1 < a.count else { return nil }
    return a[i + 1]
}
func args(_ flag: String) -> [String] {
    let a = CommandLine.arguments
    return a.indices.filter { a[$0] == flag && $0 + 1 < a.count }.map { a[$0 + 1] }
}
func has(_ flag: String) -> Bool { CommandLine.arguments.contains(flag) }

func bools(_ x: MLXArray) -> [Bool] { x.asType(.bool).asArray(Bool.self) }
func shapes(_ any: Any?) -> [(Int, Int, Int)] {
    (any as? [[Int]] ?? []).map { ($0[0], $0[1], $0[2]) }
}

// MARK: - gates

func gateSched(_ dir: URL) throws {
    let j = try loadJSON(dir.appendingPathComponent("scheduler.json"))
    for s in j["schedules"] as! [[String: Any]] {
        let steps = s["steps"] as! Int, tokens = s["tokens"] as! Int
        let ref = (s["sigmas"] as! [Double]).map { Float($0) }
        let mu = QwenImage21Scheduler.calculateShift(imageSeqLen: tokens)
        let ours = QwenImage21Scheduler.sigmas(steps: steps, mu: mu)
        let maxAbs = zip(ours, ref).map { abs($0 - $1) }.max() ?? 0
        let ok = ours.count == ref.count && maxAbs < 2e-6 && abs(mu - Float(s["mu"] as! Double)) < 1e-6
        if !ok { failures += 1 }
        print(String(format: "  %@ sched steps=%d tokens=%d mu=%.5f maxAbs %.2e  first %@ last %@",
                     ok ? "PASS" : "FAIL", steps, tokens, mu, maxAbs, "\(ours.prefix(3))", "\(ours.suffix(3))"))
    }
}

func attnProbe() {
    // MLXFast SDPA `.causal` with Lq < Lk must align the query block to the END of the keys
    // (q_i sees keys ≤ i + (Lk - Lq)) — the assumption behind the segment prefill.
    let (lq, lk, h, d) = (5, 9, 2, 16)
    let q = MLXRandom.normal([1, h, lq, d], key: MLXRandom.key(1))
    let k = MLXRandom.normal([1, h, lk, d], key: MLXRandom.key(2))
    let v = MLXRandom.normal([1, h, lk, d], key: MLXRandom.key(3))
    var m = [Float](repeating: -Float.infinity, count: lq * lk)
    for i in 0..<lq { for j in 0..<lk where j <= i + (lk - lq) { m[i * lk + j] = 0 } }
    let mask = MLXArray(m, [1, 1, lq, lk])
    let a = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.25, mask: .causal)
    let b = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.25, mask: .array(mask))
    report("sdpa causal Lq<Lk alignment", compare(a, b), cosGate: 0.999999, relGate: 1e-5)
    // and the `.causal` square case equals the explicit tril
    let q2 = MLXRandom.normal([1, h, lk, d], key: MLXRandom.key(4))
    var m2 = [Float](repeating: -Float.infinity, count: lk * lk)
    for i in 0..<lk { for j in 0...i { m2[i * lk + j] = 0 } }
    let a2 = MLXFast.scaledDotProductAttention(queries: q2, keys: k, values: v, scale: 0.25, mask: .causal)
    let b2 = MLXFast.scaledDotProductAttention(queries: q2, keys: k, values: v, scale: 0.25, mask: .array(MLXArray(m2, [1, 1, lk, lk])))
    report("sdpa causal square", compare(a2, b2), cosGate: 0.999999, relGate: 1e-5)
}

func gateVAE(root: URL, goldens: URL) throws {
    let vae = try QwenImage21Weights.loadVAE(directory: root.appendingPathComponent("vae"), dtype: .float32)
    print("VAE loaded (fp32)")
    for name in ["vae_img_a", "vae_img_b"] {
        let g = try MLX.loadArrays(url: goldens.appendingPathComponent("\(name).safetensors"))
        let pixels = g["pixels_rgba"]![.newAxis]  // (1,4,1,H,W)
        let raw = vae.encodeRaw(pixels)
        eval(raw)
        report("\(name) encode raw", compare(raw[0], g["latents_raw"]!))
        report("\(name) encode normalized", compare(AutoencoderKLQwenImage21.normalize(raw)[0], g["latents_normalized"]!))
        let dec = vae.decode(g["latents_raw"]![.newAxis])
        eval(dec)
        report("\(name) decode", compare(dec[0], g["decoded_rgba"]!))
        // VAE-input construction from the golden PNG (exact bytes expected)
        let caseName = name == "vae_img_a" ? "edit_1img_img_a" : "edit_2img_img_b"
        let png = try QwenImage21PNG.read(url: goldens.appendingPathComponent("\(caseName)_resized_rgba.png"))
        let ours = MLXArray(png.vaePixelsCHW(), [4, 1, png.height, png.width])
        report("\(name) png->vae pixels", compare(ours, g["pixels_rgba"]!), cosGate: 0.999999, relGate: 1e-6)
    }
    let g = try MLX.loadArrays(url: goldens.appendingPathComponent("vae_random_decode.safetensors"))
    let dec = vae.decode(AutoencoderKLQwenImage21.deNormalize(g["latents_normalized"]![.newAxis]))
    eval(dec)
    report("vae_random_decode", compare(dec[0], g["decoded_rgba"]!))
}

func gateResize(goldens: URL) throws {
    for (orig, resizedName, w, h) in [("img_a", "edit_1img_img_a", 320, 320), ("img_b", "edit_2img_img_b", 288, 384)] {
        let src = try QwenImage21PNG.read(url: goldens.appendingPathComponent("\(orig).png"))
        let ref = try QwenImage21PNG.read(url: goldens.appendingPathComponent("\(resizedName)_resized_rgba.png"))
        let ours = QwenImage21PILResize.resizeRGBA(src, outWidth: w, outHeight: h)
        var maxDiff = 0, nDiff = 0
        for i in 0..<ours.rgba.count {
            let d = abs(Int(ours.rgba[i]) - Int(ref.rgba[i]))
            if d > 0 { nDiff += 1 }
            maxDiff = max(maxDiff, d)
        }
        let ok = maxDiff <= 1 && nDiff < ours.rgba.count / 100
        if !ok { failures += 1 }
        print("  \(ok ? "PASS" : "FAIL") lanczos rgba \(orig) -> \(w)x\(h): maxDiff \(maxDiff) differing bytes \(nDiff)/\(ours.rgba.count)")
    }
}

func gateEncoder(qwenDir: URL, goldens: URL, tokenizerDir: URL?) async throws {
    let enc = try await QwenImage21PromptEncoder.load(qwenDir: qwenDir, tokenizerDir: tokenizerDir, dtype: .float32)
    let meta = try loadJSON(goldens.appendingPathComponent("encoder_meta.json"))
    let refDrop = meta["drop_idx"] as! Int
    print("encoder loaded (fp32); drop_idx ours \(enc.dropIdx) ref \(refDrop)")
    if enc.dropIdx != refDrop { failures += 1 }
    for name in ["encoder_t2i_short", "encoder_t2i_card", "encoder_edit_1img", "encoder_edit_2img"] {
        let g = try MLX.loadArrays(url: goldens.appendingPathComponent("\(name).safetensors"))
        let j = try loadJSON(goldens.appendingPathComponent("\(name).json"))
        let prompt = j["prompt"] as! String
        let sizes = j["image_sizes_wh"] as! [[Int]]
        var images: [QwenImage21RGBAImage] = []
        let imgNames = name == "encoder_edit_1img" ? ["img_a"] : name == "encoder_edit_2img" ? ["img_a", "img_b"] : []
        for (i, n) in imgNames.enumerated() {
            let png = try QwenImage21PNG.read(url: goldens.appendingPathComponent("\(name.replacingOccurrences(of: "encoder_", with: ""))_\(n)_resized_rgba.png"))
            precondition(png.width == sizes[i][0] && png.height == sizes[i][1])
            images.append(png)
        }
        let out = try enc.encode(prompt: prompt, images: images)
        let refIds = g["input_ids"]!.asArray(Int32.self).map { Int($0) }
        let idsOK = out.inputIds == refIds
        if !idsOK { failures += 1 }
        print("  \(idsOK ? "PASS" : "FAIL") \(name) token ids: ours \(out.inputIds.count) ref \(refIds.count)\(idsOK ? "" : " first diff at \(zip(out.inputIds, refIds).enumerated().first { $0.1.0 != $0.1.1 }.map { $0.0 } ?? -1)")")
        let refMask = bools(g["image_pad_mask"]!)
        let maskOK = out.imagePadMask == refMask
        if !maskOK { failures += 1 }
        print("  \(maskOK ? "PASS" : "FAIL") \(name) image_pad_mask (\(out.imagePadMask.filter { $0 }.count) pads)")
        if let pv = g["pixel_values"] {
            // our preprocessing of the golden resized RGBA vs the processor's pixel_values
            var parts: [MLXArray] = []
            for img in images { parts.append(enc.processor.preprocess(rgb: img.compositedOverWhiteRGB(), width: img.width, height: img.height).0) }
            report("\(name) pixel_values", compare(parts.count == 1 ? parts[0] : concatenated(parts, axis: 0), pv), cosGate: 0.99999, relGate: 5e-3)
        }
        eval(out.embeds)
        report("\(name) prompt_embeds (pre-norm)", compare(out.embeds[0], g["prompt_embeds_prenorm"]!), cosGate: 0.999, relGate: 5e-2)
    }
}

func gateDiT(root: URL, goldens: URL, only: String?) throws {
    let tr = try QwenImage21Weights.loadTransformer(directory: root.appendingPathComponent("transformer"), dtype: .float32)
    print("transformer loaded (fp32)")
    for name in ["dit_t2i_short", "dit_edit_1img", "dit_edit_2img"] where only == nil || name == only {
        let g = try MLX.loadArrays(url: goldens.appendingPathComponent("\(name).safetensors"))
        let j = try loadJSON(goldens.appendingPathComponent("\(name).json"))
        let imgShapes = shapes(j["img_shapes"])
        let sigmas = (j["sigmas"] as! [Double]).map { Float($0) }
        let nTarget = j["n_target"] as! Int
        print("== \(name): shapes \(imgShapes) sigmas \(sigmas.prefix(2))")
        let layout = try tr.buildLayout(imgMask: bools(g["img_mask"]!), imgShapes: imgShapes)
        // token metadata
        let refPad = bools(g["image_pad_mask_joint"]!)
        let refTarget = bools(g["target_token_mask"]!)
        let prefixOK = layout.imagePadMask == refPad && layout.prefixLen == (j["prefix_len"] as! Int) && refTarget.filter { $0 }.count == nTarget
        if !prefixOK { failures += 1 }
        print("  \(prefixOK ? "PASS" : "FAIL") layout: joint \(layout.jointLen) prefix \(layout.prefixLen) target \(layout.targetLen) segments \(layout.segments.map { "\($0.isText ? "T" : "I")\($0.start)-\($0.end)" })")
        report("rotary cos", compare(layout.cos, g["rotary_emb.real"]!), cosGate: 0.999999, relGate: 1e-4)
        report("rotary sin", compare(layout.sin, g["rotary_emb.imag"]!), cosGate: 0.999999, relGate: 1e-4)

        let pe = g["prompt_embeds"]![.newAxis]
        let hidden0 = g["hidden_in_step0"]![.newAxis]
        var taps: [Int: MLXArray] = [:]
        tr.blockTap = { i, x in taps[i] = x }
        let cache = QwenImage21KVCache(numLayers: tr.numLayers)
        let out0 = tr(hiddenStates: hidden0, encoderHiddenStates: pe, timestep: MLXArray([sigmas[0]]), layout: layout,
                      kvCache: cache, mode: .extract)
        eval(out0)
        report("temb", compare(taps[-2]!, g["temb"]!), cosGate: 0.999999, relGate: 1e-3)
        report("modulation", compare(taps[-3]!, g["modulation"]!), cosGate: 0.999999, relGate: 1e-3)
        for i in [0, 1, 7, 15, 31] {
            report(String(format: "block_%02d", i), compare(taps[i]![0], g[String(format: "block_%02d", i)]!), cosGate: 0.9999, relGate: 2e-2)
        }
        report("out_step0_joint", compare(out0[0], g["out_step0_joint"]!), cosGate: 0.9999, relGate: 2e-2)
        report("out_step0 target rows", compare(out0[0, (out0.dim(1) - nTarget)...], g["out_step0_joint"]![(out0.dim(1) - nTarget)...]), cosGate: 0.9999, relGate: 2e-2)
        report("kv cache L0 k", compare(cache.layers[0].k![0], g["kv_cache_layer0_k"]!), cosGate: 0.99999, relGate: 5e-3)
        report("kv cache L0 v", compare(cache.layers[0].v![0], g["kv_cache_layer0_v"]!), cosGate: 0.99999, relGate: 5e-3)
        report("kv cache L31 k", compare(cache.layers[31].k![0], g["kv_cache_layer31_k"]!), cosGate: 0.9999, relGate: 2e-2)
        tr.blockTap = nil
        // step 1: cached vs uncached with the golden step-1 latents
        let lat1 = g["latents_step1"]![.newAxis]
        let condLen = hidden0.dim(1) - nTarget
        let hidden1 = condLen > 0 ? concatenated([hidden0[0..., ..<condLen], lat1], axis: 1) : lat1
        let out1c = tr(hiddenStates: hidden1, encoderHiddenStates: pe, timestep: MLXArray([sigmas[1]]), layout: layout,
                       kvCache: cache, mode: .cached)
        eval(out1c)
        report("out_step1_cached", compare(out1c[0], g["out_step1_cached"]!), cosGate: 0.9999, relGate: 2e-2)
        let out1u = tr(hiddenStates: hidden1, encoderHiddenStates: pe, timestep: MLXArray([sigmas[1]]), layout: layout, mode: .none)
        eval(out1u)
        report("out_step1_uncached_joint", compare(out1u[0], g["out_step1_uncached_joint"]!), cosGate: 0.9999, relGate: 2e-2)
        report("cached vs uncached (ours)", compare(out1c[0], out1u[0, (out1u.dim(1) - nTarget)...]), cosGate: 0.9999, relGate: 2e-2)
    }
}

func generate(root: URL, qwenDir: URL) async throws {
    let prompt = arg("--prompt") ?? "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
    let steps = Int(arg("--steps") ?? "40")!
    let seed = UInt64(arg("--seed") ?? "42")!
    let size = arg("--size").flatMap(Int.init)
    let outRes = Int(arg("--out-res") ?? "1024")!
    let cfg = Float(arg("--cfg") ?? "1")!
    let outPath = arg("--out") ?? "qwen-image-2.1.png"
    let images = try args("--image").map { try QwenImage21PNG.read(url: URL(fileURLWithPath: $0)) }
    // `--latents file.safetensors` injects the reference's initial noise (key `latents_packed`, [1, hw, 64]
    // or [hw, 64]) so a render can be compared numerically with a torch run despite the RNG mismatch.
    var injected: MLXArray? = nil
    if let lp = arg("--latents") {
        let d = try MLX.loadArrays(url: URL(fileURLWithPath: lp))
        var l = d["latents_packed"] ?? d.values.first!
        if l.ndim == 2 { l = l[.newAxis] }
        injected = l
        print("injected noise \(l.shape)")
    }
    let t0 = Date()
    let tr = try QwenImage21Weights.loadTransformer(directory: root.appendingPathComponent("transformer"), dtype: .bfloat16)
    let vae = try QwenImage21Weights.loadVAE(directory: root.appendingPathComponent("vae"), dtype: has("--fp32-vae") ? .float32 : .bfloat16)
    print(String(format: "loaded DiT bf16 + VAE in %.1fs", Date().timeIntervalSince(t0)))
    let gen = QwenImage21Generator(
        encoderProvider: { try await QwenImage21PromptEncoder.load(qwenDir: qwenDir, dtype: .bfloat16) },
        transformer: tr, vae: vae, keepEncoderResident: has("--keep-encoder"))
    let t1 = Date()
    var last = Date()
    let r = try await gen.generate(
        prompt: prompt, images: images, negativePrompt: arg("--neg"), trueCFGScale: cfg,
        width: size, height: size, outputResolution: outRes, steps: steps, seed: seed, useKVCache: !has("--no-cache"),
        latents: injected,
        progress: { i, n in
            let now = Date()
            print(String(format: "  step %d/%d  %.2fs  peak %.1f GB", i, n, now.timeIntervalSince(last), Double(Memory.peakMemory) / 1e9))
            last = now
        })
    print(String(format: "generated %dx%d in %.1fs (steps %d, cache %@); peak %.1f GB", r.image.width, r.image.height,
                 Date().timeIntervalSince(t1), steps, has("--no-cache") ? "off" : "on", Double(Memory.peakMemory) / 1e9))
    try QwenImage21PNG.write(r.image, to: URL(fileURLWithPath: outPath))
    print("wrote \(outPath)")
    if let sl = arg("--save-latents") {
        try MLX.save(arrays: ["latents_packed": r.latentsPacked.asType(.float32)], url: URL(fileURLWithPath: sl))
        print("saved final latents to \(sl)")
    }
}

// MARK: - main

let cli = CommandLine.arguments
do {
    if has("--sched") {
        try gateSched(URL(fileURLWithPath: arg("--sched")!))
    } else if has("--attn-probe") {
        attnProbe()
    } else if has("--resize") {
        try gateResize(goldens: URL(fileURLWithPath: arg("--resize")!))
    } else if has("--vae") {
        let root = URL(fileURLWithPath: arg("--vae")!)
        let goldens = URL(fileURLWithPath: cli[cli.firstIndex(of: "--vae")! + 2])
        try Device.withDefaultDevice(.cpu) { try gateVAE(root: root, goldens: goldens) }
    } else if has("--encoder") {
        let q = URL(fileURLWithPath: arg("--encoder")!)
        let goldens = URL(fileURLWithPath: cli[cli.firstIndex(of: "--encoder")! + 2])
        let tok = arg("--tokenizer").map { URL(fileURLWithPath: $0) }
        Device.setDefault(device: Device.cpu)
        try await gateEncoder(qwenDir: q, goldens: goldens, tokenizerDir: tok)
    } else if has("--dit") {
        let root = URL(fileURLWithPath: arg("--dit")!)
        let goldens = URL(fileURLWithPath: cli[cli.firstIndex(of: "--dit")! + 2])
        if has("--gpu") {
            print("DiT gate on the GPU stream (fp32; expect ~1e-3 GPU accumulation noise vs the CPU goldens)")
            try gateDiT(root: root, goldens: goldens, only: arg("--case"))
        } else {
            try Device.withDefaultDevice(.cpu) { try gateDiT(root: root, goldens: goldens, only: arg("--case")) }
        }
    } else if has("--generate") {
        let root = URL(fileURLWithPath: arg("--generate")!)
        let q = URL(fileURLWithPath: cli[cli.firstIndex(of: "--generate")! + 2])
        try await generate(root: root, qwenDir: q)
    } else {
        print("usage: see header"); exit(2)
    }
} catch {
    print("ERROR: \(error)")
    exit(1)
}
print(failures == 0 ? "ALL GATES PASS" : "\(failures) GATE(S) FAILED")
exit(failures == 0 ? 0 : 1)
