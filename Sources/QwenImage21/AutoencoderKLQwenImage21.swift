// AutoencoderKLQwenImage21 — Swift/MLX port of diffusers main `autoencoder_kl_qwenimage21.py`.
//
// A Wan2.2-style RESIDUAL VAE (param-free AvgDown3D / DupUp3D shortcuts around every stage)
// specialised to images: `QwenImage21CausalConv3d` folds the single frame away and is a plain
// symmetric-padded Conv2d, so every conv here is 2-D and the checkpoint's weights are 4-D.
// 64-channel latents, 16x spatial (5 stages, dim_mult [1,2,4,8,8]), RGBA in/out (4 channels),
// encoder base 96, decoder base 144. Norms are the Wan "RMS" norm: L2-normalise over channels
// (eps 1e-12) · sqrt(C) · gamma — identical to qwen-image-edit-swift's `WanRMSNorm`.
//
// Layout: channels-last [B, H, W, C] (MLX Conv2d native) with one transpose at the encode /
// decode boundary to the pipeline's PT convention (B, C, T=1, H, W). The temporal shortcuts
// keep an explicit T axis internally so the AvgDown3D front zero-pad and the DupUp3D
// first-chunk frame drop are reproduced exactly (they change the numbers for T = 1: half of a
// temporal AvgDown3D's output channels are the zero frame's mean).
//
// Reuse: block structure mirrors wan-core-mlx-swift `WanVAE22.swift` (parity-locked 48-ch
// vae22 port) and qwen-image-edit-swift `QwenVAE.swift`.

import Foundation
import MLX
import MLXNN

// MARK: - Norm / conv primitives

/// `QwenImage21RMS_norm`: F.normalize(x, dim=C) · sqrt(C) · gamma (+ bias 0). eps 1e-12.
public final class QwenImage21RMSNorm: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    let scale: Float
    let eps: Float

    public init(channels: Int, eps: Float = 1e-12) {
        self._gamma.wrappedValue = MLXArray.ones([channels])
        self.scale = Float(channels).squareRoot()
        self.eps = eps
        super.init()
    }

    /// x: (..., C)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let l2 = sqrt(sum(xf * xf, axis: -1, keepDims: true))
        let denom = maximum(l2, MLXArray(eps))
        return ((xf / denom).asType(x.dtype)) * scale * gamma.asType(x.dtype)
    }
}

// `QwenImage21CausalConv3d` (image specialisation) == a symmetric-padded MLXNN Conv2d: its
// parameter keys are `weight`/`bias`, matching the checkpoint's `<name>.weight`/`<name>.bias`
// after the PT (O,I,kH,kW) -> MLX (O,kH,kW,I) transpose at load.

public final class QwenImage21ResidualBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: QwenImage21RMSNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: QwenImage21RMSNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    public init(inDim: Int, outDim: Int) {
        self._norm1.wrappedValue = QwenImage21RMSNorm(channels: inDim)
        self._conv1.wrappedValue = Conv2d(inputChannels: inDim, outputChannels: outDim, kernelSize: 3, padding: 1)
        self._norm2.wrappedValue = QwenImage21RMSNorm(channels: outDim)
        self._conv2.wrappedValue = Conv2d(inputChannels: outDim, outputChannels: outDim, kernelSize: 3, padding: 1)
        self._convShortcut.wrappedValue = inDim != outDim
            ? Conv2d(inputChannels: inDim, outputChannels: outDim, kernelSize: 1) : nil
        super.init()
    }

    /// x: [N, H, W, C]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h0 = convShortcut.map { $0(x) } ?? x
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        return h + h0
    }
}

/// Single-head self-attention over H·W (mid blocks).
public final class QwenImage21AttentionBlock: Module {
    @ModuleInfo(key: "norm") var norm: QwenImage21RMSNorm
    @ModuleInfo(key: "to_qkv") var toQKV: Conv2d
    @ModuleInfo(key: "proj") var proj: Conv2d
    let dim: Int

    public init(dim: Int) {
        self.dim = dim
        self._norm.wrappedValue = QwenImage21RMSNorm(channels: dim)
        self._toQKV.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1)
        self._proj.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
        super.init()
    }

    /// x: [N, H, W, C]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (n, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let identity = x
        let qkv = toQKV(norm(x)).reshaped(n, h * w, 3, c)
        let q = qkv[0..., 0..., 0]
        let k = qkv[0..., 0..., 1]
        let v = qkv[0..., 0..., 2]
        let scale = 1.0 / sqrt(Float(c))
        let scores = softmax((matmul(q, k.transposed(0, 2, 1)) * scale).asType(.float32), axis: -1).asType(x.dtype)
        var out = matmul(scores, v).reshaped(n, h, w, c)
        out = proj(out)
        return out + identity
    }
}

public final class QwenImage21MidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [QwenImage21ResidualBlock]
    @ModuleInfo(key: "attentions") var attentions: [QwenImage21AttentionBlock]

    public init(dim: Int) {
        self._resnets.wrappedValue = [
            QwenImage21ResidualBlock(inDim: dim, outDim: dim), QwenImage21ResidualBlock(inDim: dim, outDim: dim),
        ]
        self._attentions.wrappedValue = [QwenImage21AttentionBlock(dim: dim)]
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = resnets[0](x)
        x = attentions[0](x)
        return resnets[1](x)
    }
}

// MARK: - Param-free residual shortcuts (T axis explicit, T = 1 in practice)

/// `AvgDown3D`: group (ft, fs, fs) neighbourhoods into channels and average `groupSize` of
/// them per output channel. Front zero-pads T to a multiple of `factorT` — for a single frame
/// with factorT 2 that pad is HALF of every group's temporal slots.
public final class QwenImage21AvgDown3D {
    public let inChannels: Int, outChannels: Int, factorT: Int, factorS: Int, groupSize: Int

    public init(_ inChannels: Int, _ outChannels: Int, factorT: Int, factorS: Int = 1) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.factorT = factorT
        self.factorS = factorS
        let factor = factorT * factorS * factorS
        precondition(inChannels * factor % outChannels == 0)
        self.groupSize = inChannels * factor / outChannels
    }

    /// x: [B, T, H, W, C] -> [B, T/ft, H/fs, W/fs, outChannels]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        let b = x.dim(0)
        var t = x.dim(1)
        let (h, w, c) = (x.dim(2), x.dim(3), x.dim(4))
        let padT = (factorT - t % factorT) % factorT
        if padT > 0 {
            x = padded(x, widths: [IntOrPair([0, 0]), IntOrPair([padT, 0]), IntOrPair([0, 0]),
                                   IntOrPair([0, 0]), IntOrPair([0, 0])])
            t += padT
        }
        let (ft, fs) = (factorT, factorS)
        x = x.reshaped(b, t / ft, ft, h / fs, fs, w / fs, fs, c)
        x = x.transposed(0, 1, 3, 5, 7, 2, 4, 6)  // [B, T', H', W', C, ft, fs, fs]
        x = x.reshaped(b, t / ft, h / fs, w / fs, outChannels, groupSize)
        return mean(x, axis: -1)
    }
}

/// `DupUp3D`: repeat channels then scatter them over (ft, fs, fs); `firstChunk` drops the
/// leading `factorT - 1` frames.
public final class QwenImage21DupUp3D {
    public let inChannels: Int, outChannels: Int, factorT: Int, factorS: Int, repeats: Int

    public init(_ inChannels: Int, _ outChannels: Int, factorT: Int, factorS: Int = 1) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.factorT = factorT
        self.factorS = factorS
        let factor = factorT * factorS * factorS
        precondition(outChannels * factor % inChannels == 0)
        self.repeats = outChannels * factor / inChannels
    }

    /// x: [B, T, H, W, C] -> [B, T·ft (- (ft-1) on the first chunk), H·fs, W·fs, outChannels]
    public func callAsFunction(_ x: MLXArray, firstChunk: Bool = false) -> MLXArray {
        let (b, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        var y = repeated(x, count: repeats, axis: -1)
        y = y.reshaped(b, t, h, w, outChannels, factorT, factorS, factorS)
        y = y.transposed(0, 1, 5, 2, 6, 3, 7, 4)  // [B, T, ft, H, fs, W, fs, outC]
        y = y.reshaped(b, t * factorT, h * factorS, w * factorS, outChannels)
        if firstChunk, factorT > 1 { y = y[0..., (factorT - 1)...] }
        return y
    }
}

// MARK: - Resample

/// `QwenImage21Resample`: upsample = nearest-2x + Conv2d(dim -> upsample_out_dim);
/// downsample = ZeroPad2d((0,1,0,1)) + Conv2d(dim -> dim, stride 2). The 3d modes also
/// checkpoint a `time_conv` (1x1) that the reference never applies to a first/only frame —
/// kept as a parameter holder so the strict loader consumes its keys.
public final class QwenImage21Resample: Module {
    public let mode: String
    /// upstream `Sequential(Upsample/ZeroPad2d, Conv2d)` -> key `resample.1`; ours is `[Conv2d]`
    /// at index 0 and the loader renames `.resample.1.` -> `.resample.0.`.
    @ModuleInfo(key: "resample") var resample: [Conv2d]
    @ModuleInfo(key: "time_conv") var timeConv: Conv2d?

    public init(dim: Int, mode: String, upsampleOutDim: Int? = nil) {
        self.mode = mode
        switch mode {
        case "upsample2d", "upsample3d":
            self._resample.wrappedValue = [
                Conv2d(inputChannels: dim, outputChannels: upsampleOutDim ?? dim / 2, kernelSize: 3, padding: 1)
            ]
            self._timeConv.wrappedValue = mode == "upsample3d"
                ? Conv2d(inputChannels: dim, outputChannels: dim * 2, kernelSize: 1) : nil
        case "downsample2d", "downsample3d":
            self._resample.wrappedValue = [
                Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2, padding: 0)
            ]
            self._timeConv.wrappedValue = mode == "downsample3d"
                ? Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1) : nil
        default:
            fatalError("unsupported resample mode \(mode)")
        }
        super.init()
    }

    /// x: [N, H, W, C]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch mode {
        case "upsample2d", "upsample3d":
            let up = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
            return resample[0](up)
        default:
            let p = padded(x, widths: [IntOrPair([0, 0]), IntOrPair([0, 1]), IntOrPair([0, 1]), IntOrPair([0, 0])])
            return resample[0](p)
        }
    }
}

// MARK: - Residual stages

public final class QwenImage21ResidualDownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [QwenImage21ResidualBlock]
    @ModuleInfo(key: "downsampler") var downsampler: QwenImage21Resample?
    public let avgShortcut: QwenImage21AvgDown3D

    public init(inDim: Int, outDim: Int, numResBlocks: Int, temporalDownsample: Bool, downFlag: Bool) {
        var blocks: [QwenImage21ResidualBlock] = []
        var d = inDim
        for _ in 0..<numResBlocks {
            blocks.append(QwenImage21ResidualBlock(inDim: d, outDim: outDim))
            d = outDim
        }
        self._resnets.wrappedValue = blocks
        self._downsampler.wrappedValue = downFlag
            ? QwenImage21Resample(dim: outDim, mode: temporalDownsample ? "downsample3d" : "downsample2d") : nil
        self.avgShortcut = QwenImage21AvgDown3D(
            inDim, outDim, factorT: temporalDownsample ? 2 : 1, factorS: downFlag ? 2 : 1)
        super.init()
    }

    /// x: [N, H, W, C] (single frame)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let downsampler { h = downsampler(h) }
        let sc = avgShortcut(x.expandedDimensions(axis: 1)).squeezed(axis: 1)
        return h + sc
    }
}

public final class QwenImage21ResidualUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [QwenImage21ResidualBlock]
    @ModuleInfo(key: "upsampler") var upsampler: QwenImage21Resample?
    public let avgShortcut: QwenImage21DupUp3D?

    public init(inDim: Int, outDim: Int, numResBlocks: Int, temporalUpsample: Bool, upFlag: Bool) {
        var blocks: [QwenImage21ResidualBlock] = []
        var d = inDim
        for _ in 0..<(numResBlocks + 1) {
            blocks.append(QwenImage21ResidualBlock(inDim: d, outDim: outDim))
            d = outDim
        }
        self._resnets.wrappedValue = blocks
        self._upsampler.wrappedValue = upFlag
            ? QwenImage21Resample(dim: outDim, mode: temporalUpsample ? "upsample3d" : "upsample2d", upsampleOutDim: outDim)
            : nil
        self.avgShortcut = upFlag ? QwenImage21DupUp3D(inDim, outDim, factorT: temporalUpsample ? 2 : 1, factorS: 2) : nil
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let upsampler { h = upsampler(h) }
        if let avgShortcut {
            h = h + avgShortcut(x.expandedDimensions(axis: 1), firstChunk: true).squeezed(axis: 1)
        }
        return h
    }
}

// MARK: - Encoder / decoder

public final class QwenImage21Encoder3d: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [QwenImage21ResidualDownBlock]
    @ModuleInfo(key: "mid_block") var midBlock: QwenImage21MidBlock
    @ModuleInfo(key: "norm_out") var normOut: QwenImage21RMSNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    public init(inChannels: Int = 4, dim: Int = 96, zDim: Int = 128, dimMult: [Int] = [1, 2, 4, 8, 8],
                numResBlocks: Int = 2, temporalDownsample: [Bool] = [false, true, true, true]) {
        let dims = ([1] + dimMult).map { dim * $0 }
        self._convIn.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: dims[0], kernelSize: 3, padding: 1)
        var blocks: [QwenImage21ResidualDownBlock] = []
        for i in 0..<dimMult.count {
            let last = i == dimMult.count - 1
            blocks.append(QwenImage21ResidualDownBlock(
                inDim: dims[i], outDim: dims[i + 1], numResBlocks: numResBlocks,
                temporalDownsample: last ? false : temporalDownsample[i], downFlag: !last))
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = QwenImage21MidBlock(dim: dims.last!)
        self._normOut.wrappedValue = QwenImage21RMSNorm(channels: dims.last!)
        self._convOut.wrappedValue = Conv2d(inputChannels: dims.last!, outputChannels: zDim, kernelSize: 3, padding: 1)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = convIn(x)
        for b in downBlocks { x = b(x) }
        x = midBlock(x)
        return convOut(silu(normOut(x)))
    }
}

public final class QwenImage21Decoder3d: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: QwenImage21MidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [QwenImage21ResidualUpBlock]
    @ModuleInfo(key: "norm_out") var normOut: QwenImage21RMSNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    public init(dim: Int = 144, zDim: Int = 64, dimMult: [Int] = [1, 2, 4, 8, 8], numResBlocks: Int = 2,
                temporalUpsample: [Bool] = [true, true, true, false], outChannels: Int = 4) {
        let dims = ([dimMult.last!] + dimMult.reversed()).map { dim * $0 }
        self._convIn.wrappedValue = Conv2d(inputChannels: zDim, outputChannels: dims[0], kernelSize: 3, padding: 1)
        self._midBlock.wrappedValue = QwenImage21MidBlock(dim: dims[0])
        var blocks: [QwenImage21ResidualUpBlock] = []
        for i in 0..<dimMult.count {
            let upFlag = i != dimMult.count - 1
            blocks.append(QwenImage21ResidualUpBlock(
                inDim: dims[i], outDim: dims[i + 1], numResBlocks: numResBlocks,
                temporalUpsample: upFlag ? temporalUpsample[i] : false, upFlag: upFlag))
        }
        self._upBlocks.wrappedValue = blocks
        self._normOut.wrappedValue = QwenImage21RMSNorm(channels: dims.last!)
        self._convOut.wrappedValue = Conv2d(inputChannels: dims.last!, outputChannels: outChannels, kernelSize: 3, padding: 1)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = convIn(x)
        x = midBlock(x)
        for b in upBlocks { x = b(x) }
        return convOut(silu(normOut(x)))
    }
}

// MARK: - Top level

public final class AutoencoderKLQwenImage21: Module {
    @ModuleInfo(key: "encoder") var encoder: QwenImage21Encoder3d
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    @ModuleInfo(key: "decoder") var decoder: QwenImage21Decoder3d

    public let zDim: Int
    public static let spatialCompression = 16
    /// Compute dtype of the loaded weights (set by the loader); inputs are cast to it.
    public var weightDtype: DType = .float32

    /// vae/config.json latents_mean / latents_std (64 each).
    public static let latentsMean: [Float] = [
        0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732,
        0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
        -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555,
        0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
        -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632,
        0.5547, -1.7121, -0.1562, 0.0304,
    ]
    public static let latentsStd: [Float] = [
        3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904,
        2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
        3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075,
        3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
        3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149,
        3.7662, 3.6746, 3.4563, 3.8161,
    ]

    public init(baseDim: Int = 96, decoderBaseDim: Int = 144, zDim: Int = 64, dimMult: [Int] = [1, 2, 4, 8, 8],
                numResBlocks: Int = 2, temporalDownsample: [Bool] = [false, true, true, true],
                inChannels: Int = 4, outChannels: Int = 4) {
        self.zDim = zDim
        self._encoder.wrappedValue = QwenImage21Encoder3d(
            inChannels: inChannels, dim: baseDim, zDim: zDim * 2, dimMult: dimMult, numResBlocks: numResBlocks,
            temporalDownsample: temporalDownsample)
        self._quantConv.wrappedValue = Conv2d(inputChannels: zDim * 2, outputChannels: zDim * 2, kernelSize: 1)
        self._postQuantConv.wrappedValue = Conv2d(inputChannels: zDim, outputChannels: zDim, kernelSize: 1)
        self._decoder.wrappedValue = QwenImage21Decoder3d(
            dim: decoderBaseDim, zDim: zDim, dimMult: dimMult, numResBlocks: numResBlocks,
            temporalUpsample: Array(temporalDownsample.reversed()), outChannels: outChannels)
        super.init()
    }

    /// image (B, 4, 1, H, W) in [-1, 1] -> RAW latent mode (B, 64, 1, H/16, W/16)
    /// (`encode(x).latent_dist.mode()` = first 64 of quant_conv's 128 channels).
    public func encodeRaw(_ image: MLXArray) -> MLXArray {
        precondition(image.ndim == 5 && image.dim(2) == 1, "single frame (B, C, 1, H, W) expected")
        var x = image.asType(weightDtype).squeezed(axis: 2).transposed(0, 2, 3, 1)  // (B, H, W, C)
        x = encoder(x)
        x = quantConv(x)
        let mean = x[.ellipsis, ..<zDim]
        return mean.transposed(0, 3, 1, 2).expandedDimensions(axis: 2)  // (B, 64, 1, h, w)
    }

    /// Pipeline `_encode_vae_image`: raw mode -> (z - mean) / std.
    public func encode(_ image: MLXArray) -> MLXArray { Self.normalize(encodeRaw(image)) }

    /// De-normalised latents (B, 64, 1, h, w) -> RGBA (B, 4, 1, 16h, 16w) clamped to [-1, 1].
    public func decode(_ latents: MLXArray) -> MLXArray {
        precondition(latents.ndim == 5 && latents.dim(2) == 1)
        var x = latents.asType(weightDtype).squeezed(axis: 2).transposed(0, 2, 3, 1)
        x = postQuantConv(x)
        x = decoder(x)
        x = clip(x, min: -1, max: 1)
        return x.transposed(0, 3, 1, 2).expandedDimensions(axis: 2)
    }

    public static func normalize(_ raw: MLXArray) -> MLXArray {
        let m = MLXArray(latentsMean).reshaped(1, 64, 1, 1, 1).asType(raw.dtype)
        let s = MLXArray(latentsStd).reshaped(1, 64, 1, 1, 1).asType(raw.dtype)
        return (raw - m) / s
    }

    public static func deNormalize(_ normalized: MLXArray) -> MLXArray {
        let m = MLXArray(latentsMean).reshaped(1, 64, 1, 1, 1).asType(normalized.dtype)
        let s = MLXArray(latentsStd).reshaped(1, 64, 1, 1, 1).asType(normalized.dtype)
        return normalized * s + m
    }
}
