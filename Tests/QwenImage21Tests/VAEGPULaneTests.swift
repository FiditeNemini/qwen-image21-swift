// Qwen-Image-2.1 RGBA VAE on the GPU lane (conv route .conv3d / raw .winograd) vs the CPU lane
// and the oracle goldens. `--vae` runs on the CPU lane and `--vae-tile` compares GPU with GPU, so
// neither saw the Winograd window.
//
// Goldens (qwen-image21-oracle/goldens, torch fp32 CPU): vae_img_a / vae_img_b — real-image
// encode (pixels_rgba → latents_raw) and decode (latents_raw → decoded_rgba), 320² and 288×384.
// Production size: a DIV2K photo (alpha = 1), center-cropped to 1024², encoded on the CPU lane;
// every decode of that latent is compared with the CPU-lane decode, and the GPU halo-tiled decode
// (2×2, halo 12) with the GPU untiled one under each route.
//
// Run: QWEN21_PARITY=1 QWEN21_ROOT=<.../weights/Qwen-Image-2.1> \
//      swift test -c release -Xswiftc -enable-testing --filter VAEGPULaneTests
// Overrides: QWEN21_GOLDENS, QWEN21_REAL_IMAGE.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import XCTest

@testable import QwenImage21

final class VAEGPULaneTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static let goldens = URL(
        fileURLWithPath: env["QWEN21_GOLDENS"]
            ?? "/Volumes/Satechi/Development/mlxengine-image/WIP/qwen-image21-oracle/goldens")
    static let realImage = URL(
        fileURLWithPath: env["QWEN21_REAL_IMAGE"]
            ?? "/Volumes/Satechi/Development/mlxengine-image/corpus/sr-bench/DIV2K_valid_HR/0801.png")

    static func stats(_ a: MLXArray, _ ref: MLXArray) -> String {
        let d = a.asType(.float32) - ref.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref.asType(.float32))))
        let mx = abs(d).max()
        let mse = mean(d * d)
        eval(rel, mx, mse)
        let psnr = 10 * log10(4 / max(mse.item(Float.self), 1e-30))
        return String(format: "relL2 %.2e  maxAbs %.2e  PSNR %6.2f dB", rel.item(Float.self), mx.item(Float.self), psnr)
    }

    static func relL2(_ a: MLXArray, _ ref: MLXArray) -> Float {
        let d = a.asType(.float32) - ref.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref.asType(.float32))))
        eval(rel)
        return rel.item(Float.self)
    }

    static func onCPU(_ f: () -> MLXArray) -> MLXArray {
        Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let r = f()
            eval(r)
            return r
        }
    }

    /// GPU run with encoder AND decoder on `route`; warm-up, then the mean of `reps` timed runs.
    static func gpuRun(
        _ vae: AutoencoderKLQwenImage21, route: QwenImage21VAEConvRoute, reps: Int = 3,
        _ f: () -> MLXArray
    ) -> (MLXArray, Double) {
        let saved = (vae.encoderConvRoute, vae.decoderConvRoute)
        vae.encoderConvRoute = route
        vae.decoderConvRoute = route
        defer { (vae.encoderConvRoute, vae.decoderConvRoute) = saved }
        var out = f()
        eval(out)
        let t0 = Date()
        for _ in 0..<reps {
            out = f()
            eval(out)
        }
        return (out, Date().timeIntervalSince(t0) / Double(reps) * 1000)
    }

    /// Center crop to side×side → (1, 4, 1, side, side) RGBA in [-1, 1], alpha opaque.
    static func loadCropRGBA(_ url: URL, side: Int) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw NSError(domain: "Q21VAE", code: 1, userInfo: [NSLocalizedDescriptionKey: "unreadable \(url.path)"]) }
        let (w, h) = (cg.width, cg.height)
        precondition(w >= side && h >= side, "image smaller than crop")
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        let (x0, y0) = ((w - side) / 2, (h - side) / 2)
        let plane = side * side
        var chw = [Float](repeating: 1, count: 4 * plane)  // alpha plane stays 1 (opaque)
        for y in 0..<side {
            for x in 0..<side {
                let p = ((y0 + y) * w + (x0 + x)) * 4
                let i = y * side + x
                for c in 0..<3 { chw[c * plane + i] = Float(rgba[p + c]) / 127.5 - 1 }
            }
        }
        return MLXArray(chw, [1, 4, 1, side, side])
    }

    func root() throws -> URL {
        try XCTSkipUnless(Self.env["QWEN21_PARITY"] == "1", "set QWEN21_PARITY=1 to run")
        return URL(fileURLWithPath: try XCTUnwrap(Self.env["QWEN21_ROOT"], "set QWEN21_ROOT"))
    }

    func testGoldens() throws {
        let vae = try QwenImage21Weights.loadVAE(directory: try root().appendingPathComponent("vae"), dtype: .float32)
        for name in ["vae_img_a", "vae_img_b"] {
            let g = try MLX.loadArrays(url: Self.goldens.appendingPathComponent("\(name).safetensors"))
            let pixels = g["pixels_rgba"]![.newAxis]
            let lat = g["latents_raw"]![.newAxis]
            let encCPU = Self.onCPU { vae.encodeRaw(pixels) }
            let encR = Self.gpuRun(vae, route: .conv3d) { vae.encodeRaw(pixels) }.0
            let encW = Self.gpuRun(vae, route: .winograd) { vae.encodeRaw(pixels) }.0
            let decCPU = Self.onCPU { vae.decode(lat) }
            let decR = Self.gpuRun(vae, route: .conv3d) { vae.decode(lat) }.0
            let decW = Self.gpuRun(vae, route: .winograd) { vae.decode(lat) }.0
            print("[\(name) \(pixels.dim(3))×\(pixels.dim(4)) vs torch fp32 CPU]")
            print("  encode CPU lane   \(Self.stats(encCPU[0], g["latents_raw"]!))")
            print("  encode GPU conv3d \(Self.stats(encR[0], g["latents_raw"]!))")
            print("  encode GPU raw    \(Self.stats(encW[0], g["latents_raw"]!))")
            print("  decode CPU lane   \(Self.stats(decCPU[0], g["decoded_rgba"]!))")
            print("  decode GPU conv3d \(Self.stats(decR[0], g["decoded_rgba"]!))")
            print("  decode GPU raw    \(Self.stats(decW[0], g["decoded_rgba"]!))")
        }
    }

    func testRealPhoto1024() throws {
        let vae = try QwenImage21Weights.loadVAE(directory: try root().appendingPathComponent("vae"), dtype: .float32)
        let pixels = try Self.loadCropRGBA(Self.realImage, side: 1024)
        let lat = Self.onCPU { vae.encodeRaw(pixels) }
        let t0 = Date()
        let ref = Self.onCPU { vae.decode(lat) }
        print(String(format: "[real 1024² photo: %@ — CPU-lane fp32 decode %.1f s]",
                     Self.realImage.lastPathComponent, Date().timeIntervalSince(t0)))
        Memory.clearCache()
        let encR = Self.gpuRun(vae, route: .conv3d) { vae.encodeRaw(pixels) }.0
        let encW = Self.gpuRun(vae, route: .winograd) { vae.encodeRaw(pixels) }.0
        print("  encode GPU conv3d vs CPU lane \(Self.stats(encR, lat))")
        print("  encode GPU raw    vs CPU lane \(Self.stats(encW, lat))")
        let decR = Self.gpuRun(vae, route: .conv3d) { vae.decode(lat) }.0
        let decW = Self.gpuRun(vae, route: .winograd) { vae.decode(lat) }.0
        print("  decode GPU conv3d vs CPU lane \(Self.stats(decR, ref))")
        print("  decode GPU raw    vs CPU lane \(Self.stats(decW, ref))")
        for route in [QwenImage21VAEConvRoute.winograd, .conv3d] {
            let untiled = Self.gpuRun(vae, route: route, reps: 1) { vae.decode(lat) }.0
            let tiled = Self.gpuRun(vae, route: route, reps: 1) {
                vae.decodeTiled(lat, tilesH: 2, tilesW: 2, halo: 12)
            }.0
            let mx = abs(tiled - untiled).max().item(Float.self)
            print(String(format: "  GPU %@: tiled 2×2 halo 12 vs untiled  relL2 %.2e  max|Δ| %.2e",
                         route.rawValue, Self.relL2(tiled, untiled), mx))
            if route == .conv3d { XCTAssertEqual(mx, 0, "conv3d route: halo-tiled decode must be exact") }
        }
        XCTAssertLessThan(Self.relL2(encR, lat), Self.relL2(encW, lat), "encode route vs raw")
    }

    func testTiming1024() throws {
        let vae = try QwenImage21Weights.loadVAE(directory: try root().appendingPathComponent("vae"), dtype: .float32)
        let pixels = try Self.loadCropRGBA(Self.realImage, side: 1024)
        let lat = vae.encodeRaw(pixels)
        eval(lat)
        var t: [String: [Double]] = [:]
        for _ in 0..<3 {
            for route in [QwenImage21VAEConvRoute.conv3d, .winograd] {
                t["decode \(route.rawValue)", default: []].append(Self.gpuRun(vae, route: route) { vae.decode(lat) }.1)
                t["encode \(route.rawValue)", default: []].append(Self.gpuRun(vae, route: route) { vae.encodeRaw(pixels) }.1)
            }
        }
        for k in t.keys.sorted() {
            let v = t[k]!
            print(String(format: "[timing 1024² fp32] %@: %@ ms (median %.0f)", k,
                         v.map { String(format: "%.0f", $0) }.joined(separator: "/"), v.sorted()[1]))
        }
    }

    /// Repro for the context dependence of the raw-Winograd decode (2026-09-24, mlx-swift 0.31.6):
    /// vae_img_b's raw decode is max|Δ| 0.319 vs the golden in a fresh process, but 1.996 (alpha
    /// +1.000 where the golden is −0.996, top edge) once any CPU-lane forward (cpuEnc / cpuDec) has
    /// run in the process — an unrelated CPU op or CPU-materialized weights do not trigger it, and
    /// isolated Winograd probes are deterministic and stale-buffer-proof. The conv3d route is
    /// ≤2.8e-3 in every context. Steps from QWEN21_STEPS (comma list): cpuEnc, encR, encW, cpuDec,
    /// decR, decW, cpuNoise, cpuWeights, gpuWeights; then the raw decode is compared.
    func testRawDecodeBisect() throws {
        let vae = try QwenImage21Weights.loadVAE(directory: try root().appendingPathComponent("vae"), dtype: .float32)
        let g = try MLX.loadArrays(url: Self.goldens.appendingPathComponent("vae_img_b.safetensors"))
        let pixels = g["pixels_rgba"]![.newAxis]
        let lat = g["latents_raw"]![.newAxis]
        let ref = g["decoded_rgba"]![.newAxis]
        let steps = (Self.env["QWEN21_STEPS"] ?? "").split(separator: ",").map(String.init)
        for st in steps {
            switch st {
            case "cpuEnc": _ = Self.onCPU { vae.encodeRaw(pixels) }
            case "encR": _ = Self.gpuRun(vae, route: .conv3d) { vae.encodeRaw(pixels) }
            case "encW": _ = Self.gpuRun(vae, route: .winograd) { vae.encodeRaw(pixels) }
            case "cpuDec": _ = Self.onCPU { vae.decode(lat) }
            case "decR": _ = Self.gpuRun(vae, route: .conv3d) { vae.decode(lat) }
            case "decW": _ = Self.gpuRun(vae, route: .winograd) { vae.decode(lat) }
            case "cpuNoise":  // an unrelated CPU-stream op
                _ = Self.onCPU { matmul(MLXArray.ones([64, 64]), MLXArray.ones([64, 64])) }
            case "cpuWeights":  // materialize the VAE's (lazy) weights on the CPU stream only
                Device.withDefaultDevice(.cpu) { eval(vae.parameters()) }
            case "gpuWeights":
                eval(vae.parameters())
            default: XCTFail("unknown step \(st)")
            }
        }
        let b = Self.gpuRun(vae, route: .winograd) { vae.decode(lat) }.0
        print(String(format: "  steps [%@] then raw decode: vs ref max %.3e", steps.joined(separator: ","),
                     abs(b - ref).max().item(Float.self)))
    }
}
