// RGBA image helpers for the Qwen-Image-2.1 pipeline: PIL-exact LANCZOS resize with Pillow's
// RGBA premultiply/unpremultiply round trip, white compositing for the VL branch, and PNG
// read/write (CoreGraphics) for the gate/CLI.
//
// Pillow `Image.resize` on an RGBA image converts to premultiplied "RGBa" (MULDIV255), resamples
// all four 8-bit bands with the same fixed-point kernel, then converts back
// (`(255 * c) / a`, alpha 0 or 255 passes through). The diffusers `VaeImageProcessor.resize`
// (LANCZOS default) feeds both the VL encoder (after compositing over white) and the VAE.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct QwenImage21RGBAImage: Sendable {
    public var rgba: [UInt8]
    public var width: Int
    public var height: Int

    public init(rgba: [UInt8], width: Int, height: Int) {
        precondition(rgba.count == width * height * 4)
        self.rgba = rgba
        self.width = width
        self.height = height
    }

    /// Interleaved RGB8 -> opaque RGBA.
    public init(rgb: [UInt8], width: Int, height: Int) {
        precondition(rgb.count == width * height * 3)
        var out = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            out[i * 4] = rgb[i * 3]; out[i * 4 + 1] = rgb[i * 3 + 1]; out[i * 4 + 2] = rgb[i * 3 + 2]
        }
        self.rgba = out
        self.width = width
        self.height = height
    }

    /// Composite over white (the VL-encoder copy): straight alpha, `white.paste(img, mask=A)`.
    /// PIL paste with a mask blends `out = (img * a + white * (255 - a)) / 255` per channel with
    /// its BLEND rounding: `((a*m + b*(255-m)) + 127) / 255` via MULDIV255-style arithmetic.
    public func compositedOverWhiteRGB() -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            let a = Int(rgba[i * 4 + 3])
            for c in 0..<3 {
                let v = Int(rgba[i * 4 + c])
                // Pillow ImagingPaste mask blend (Paste.c `BLEND` for 8-bit): 
                // tmp = a*mask + b*(255-mask) + 128; (tmp + (tmp >> 8)) >> 8  with a=in, b=under
                let tmp = v * a + 255 * (255 - a) + 128
                out[i * 3 + c] = UInt8(truncatingIfNeeded: (tmp + (tmp >> 8)) >> 8)
            }
        }
        return out
    }

    /// (4, 1, H, W) float array values in [-1, 1] — the VAE input (`VaeImageProcessor.preprocess`).
    public func vaePixelsCHW() -> [Float] {
        var out = [Float](repeating: 0, count: 4 * width * height)
        let plane = width * height
        for i in 0..<plane {
            for c in 0..<4 { out[c * plane + i] = Float(rgba[i * 4 + c]) / 255 * 2 - 1 }
        }
        return out
    }
}

// MARK: - PIL-exact LANCZOS (4 bands, premultiplied round trip)

public enum QwenImage21PILResize {
    static let precisionBits = 32 - 8 - 2

    static func sinc(_ x: Double) -> Double {
        if x == 0 { return 1 }
        let p = Double.pi * x
        return sin(p) / p
    }

    static func lanczos(_ xIn: Double) -> Double {
        let x = abs(xIn)
        return x < 3 ? sinc(x) * sinc(x / 3) : 0
    }

    static func coefficients(inSize: Int, outSize: Int) -> (bounds: [(min: Int, count: Int)], coeffs: [[Int32]]) {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)
        let support = 3.0 * filterscale
        let one = Double(1 << precisionBits)
        var bounds: [(Int, Int)] = []
        var coeffs: [[Int32]] = []
        for xx in 0..<outSize {
            let center = (Double(xx) + 0.5) * scale
            var xmin = Int(center - support + 0.5)
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            let count = xmax - xmin
            var w = [Double](repeating: 0, count: count)
            var total = 0.0
            for x in 0..<count {
                let v = lanczos((Double(x + xmin) - center + 0.5) / filterscale)
                w[x] = v
                total += v
            }
            var k = [Int32](repeating: 0, count: count)
            for x in 0..<count {
                let normalized = total != 0 ? w[x] / total : w[x]
                let scaled = normalized * one
                k[x] = Int32(scaled < 0 ? scaled - 0.5 : scaled + 0.5)
            }
            bounds.append((xmin, count))
            coeffs.append(k)
        }
        return (bounds, coeffs)
    }

    @inline(__always) static func clip8(_ v: Int32) -> UInt8 {
        UInt8(min(max(v >> Int32(precisionBits), 0), 255))
    }

    @inline(__always) static func mulDiv255(_ a: Int, _ b: Int) -> UInt8 {
        let tmp = a * b + 128
        return UInt8(truncatingIfNeeded: ((tmp >> 8) + tmp) >> 8)
    }

    /// Resample `bands` interleaved 8-bit channels (no alpha handling).
    static func resampleBands(_ src: [UInt8], bands: Int, width: Int, height: Int, outWidth: Int, outHeight: Int) -> [UInt8] {
        let half = Int32(1 << (precisionBits - 1))
        let (hB, hC) = coefficients(inSize: width, outSize: outWidth)
        var temp = [UInt8](repeating: 0, count: height * outWidth * bands)
        src.withUnsafeBufferPointer { s in
            temp.withUnsafeMutableBufferPointer { d in
                for y in 0..<height {
                    let rowIn = y * width * bands
                    let rowOut = y * outWidth * bands
                    for xx in 0..<outWidth {
                        let (xmin, count) = hB[xx]
                        let k = hC[xx]
                        for c in 0..<bands {
                            var acc = half
                            for x in 0..<count { acc += Int32(s[rowIn + (xmin + x) * bands + c]) * k[x] }
                            d[rowOut + xx * bands + c] = clip8(acc)
                        }
                    }
                }
            }
        }
        let (vB, vC) = coefficients(inSize: height, outSize: outHeight)
        var out = [UInt8](repeating: 0, count: outHeight * outWidth * bands)
        temp.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { d in
                for yy in 0..<outHeight {
                    let (ymin, count) = vB[yy]
                    let k = vC[yy]
                    let rowOut = yy * outWidth * bands
                    for xx in 0..<outWidth {
                        for c in 0..<bands {
                            var acc = half
                            for y in 0..<count { acc += Int32(s[(ymin + y) * outWidth * bands + xx * bands + c]) * k[y] }
                            d[rowOut + xx * bands + c] = clip8(acc)
                        }
                    }
                }
            }
        }
        return out
    }

    /// PIL `Image.resize((w, h), LANCZOS)` on an RGBA image (premultiplied round trip).
    public static func resizeRGBA(_ image: QwenImage21RGBAImage, outWidth: Int, outHeight: Int) -> QwenImage21RGBAImage {
        if image.width == outWidth && image.height == outHeight { return image }
        let n = image.width * image.height
        var pre = [UInt8](repeating: 0, count: n * 4)
        for i in 0..<n {
            let a = Int(image.rgba[i * 4 + 3])
            pre[i * 4] = mulDiv255(Int(image.rgba[i * 4]), a)
            pre[i * 4 + 1] = mulDiv255(Int(image.rgba[i * 4 + 1]), a)
            pre[i * 4 + 2] = mulDiv255(Int(image.rgba[i * 4 + 2]), a)
            pre[i * 4 + 3] = UInt8(a)
        }
        let r = resampleBands(pre, bands: 4, width: image.width, height: image.height, outWidth: outWidth, outHeight: outHeight)
        var out = [UInt8](repeating: 0, count: outWidth * outHeight * 4)
        for i in 0..<(outWidth * outHeight) {
            let a = Int(r[i * 4 + 3])
            if a == 255 || a == 0 {
                out[i * 4] = r[i * 4]; out[i * 4 + 1] = r[i * 4 + 1]; out[i * 4 + 2] = r[i * 4 + 2]
            } else {
                out[i * 4] = UInt8(min(255, (255 * Int(r[i * 4])) / a))
                out[i * 4 + 1] = UInt8(min(255, (255 * Int(r[i * 4 + 1])) / a))
                out[i * 4 + 2] = UInt8(min(255, (255 * Int(r[i * 4 + 2])) / a))
            }
            out[i * 4 + 3] = UInt8(a)
        }
        return QwenImage21RGBAImage(rgba: out, width: outWidth, height: outHeight)
    }
}

// MARK: - PNG I/O

public enum QwenImage21PNG {
    /// Decode a PNG/JPEG file to straight (non-premultiplied) RGBA8.
    public static func read(url: URL) throws -> QwenImage21RGBAImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
        else { throw QwenImage21Error.invalidInput("cannot decode image at \(url.path)") }
        let w = cg.width, h = cg.height
        // Draw into a premultiplied context is lossy for translucent pixels, so read the decoded
        // bytes directly when they are already 8-bit RGBA/RGB.
        if cg.bitsPerComponent == 8, let data = cg.dataProvider?.data as Data?,
           cg.bitsPerPixel == 32 || cg.bitsPerPixel == 24
        {
            let info = cg.alphaInfo
            let bpp = cg.bitsPerPixel / 8
            let bpr = cg.bytesPerRow
            let straight = info == .last || info == .noneSkipLast || info == .none
            let premul = info == .premultipliedLast
            let first = info == .first || info == .noneSkipFirst || info == .premultipliedFirst
            if (straight || premul || first), cg.byteOrderInfo == .orderDefault || cg.byteOrderInfo == .order32Big {
                var out = [UInt8](repeating: 255, count: w * h * 4)
                data.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
                    for y in 0..<h {
                        for x in 0..<w {
                            let o = y * bpr + x * bpp
                            let d = (y * w + x) * 4
                            if bpp == 3 {
                                out[d] = p[o]; out[d + 1] = p[o + 1]; out[d + 2] = p[o + 2]
                            } else if first {
                                let a = info == .noneSkipFirst ? 255 : p[o]
                                var r = Int(p[o + 1]), g = Int(p[o + 2]), b = Int(p[o + 3])
                                if info == .premultipliedFirst, a != 0, a != 255 { r = r * 255 / Int(a); g = g * 255 / Int(a); b = b * 255 / Int(a) }
                                out[d] = UInt8(min(r, 255)); out[d + 1] = UInt8(min(g, 255)); out[d + 2] = UInt8(min(b, 255)); out[d + 3] = a
                            } else {
                                let a = info == .noneSkipLast || info == .none ? 255 : p[o + 3]
                                var r = Int(p[o]), g = Int(p[o + 1]), b = Int(p[o + 2])
                                if premul, a != 0, a != 255 { r = r * 255 / Int(a); g = g * 255 / Int(a); b = b * 255 / Int(a) }
                                out[d] = UInt8(min(r, 255)); out[d + 1] = UInt8(min(g, 255)); out[d + 2] = UInt8(min(b, 255)); out[d + 3] = a
                            }
                        }
                    }
                }
                return QwenImage21RGBAImage(rgba: out, width: w, height: h)
            }
        }
        // Fallback: render through a premultiplied RGBA context (exact for opaque images).
        var out = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &out, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw QwenImage21Error.invalidInput("cannot create bitmap context") }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        for i in 0..<(w * h) {
            let a = Int(out[i * 4 + 3])
            if a != 0, a != 255 {
                for c in 0..<3 { out[i * 4 + c] = UInt8(min(255, Int(out[i * 4 + c]) * 255 / a)) }
            }
        }
        return QwenImage21RGBAImage(rgba: out, width: w, height: h)
    }

    /// Write straight RGBA8 as a PNG (alpha preserved).
    public static func write(_ image: QwenImage21RGBAImage, to url: URL) throws {
        let cs = CGColorSpaceCreateDeviceRGB()
        let data = Data(image.rgba)
        guard let provider = CGDataProvider(data: data as CFData),
              let cg = CGImage(width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 32,
                               bytesPerRow: image.width * 4, space: cs,
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw QwenImage21Error.invalidInput("cannot encode PNG") }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw QwenImage21Error.invalidInput("PNG write failed: \(url.path)") }
    }
}
