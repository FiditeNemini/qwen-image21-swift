// Qwen-Image-2.1 denoising transformer — Swift/MLX port.
//
// Isomorphic to diffusers main `transformer_qwenimage21.py` (QwenImage21Transformer2DModel,
// huggingface/diffusers#14804): 32 SINGLE-stream blocks over one interleaved text/image
// sequence, block-causal attention ((q >= k) or same-image-block), one shared modulation
// projection for every block (`causal_condition`: text + condition-image tokens modulate from
// t = 0, target-image tokens from the sampled t), and a prefix KV cache so denoise steps
// after the first only recompute the target image's tokens.
//
// Deltas vs the 2511 port (qwen-image-edit-swift `Transformer.swift`): no text stream / no
// joint dual-stream attention, SwiGLU FFN (2511: GELU-approx), tanh-squashed gates, scale-only
// final AdaLN, zero-centered text RMSNorm, no biases anywhere, RoPE positions from the
// reference's cursor walk (text advances all three axes; image blocks freeze the frame axis
// and centre h/w on zero), and patch_size 1 (latents are consumed unpatched).

import Foundation
import MLX
import MLXFast
import MLXNN

public enum QwenImage21Error: Error, CustomStringConvertible {
    case loading(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .loading(let m): return "QwenImage21 loading error: \(m)"
        case .invalidInput(let m): return "QwenImage21 input error: \(m)"
        }
    }
}

// MARK: - Embeddings

/// `QwenImage21TemporalTimesteps`: sinusoid with cos in the FIRST half, sin in the second,
/// freqs = exp(-ln(maxPeriod) · i / half), timestep scaled by `timeFactor` (1000).
func qwenImage21TimestepProj(_ timestep: MLXArray, dim: Int = 256, maxPeriod: Float = 10000,
                             timeFactor: Float = 1000) -> MLXArray {
    precondition(timestep.ndim == 1)
    let half = dim / 2
    let freqs = exp(-log(maxPeriod) * MLXArray(0..<half).asType(.float32) / Float(half))
    let args = (timestep.asType(.float32) * timeFactor)[0..., .newAxis] * freqs[.newAxis, 0...]
    return concatenated([cos(args), sin(args)], axis: -1)
}

/// diffusers `TimestepEmbedding(in=256, dim, sample_proj_bias=False)`: linear_1 -> SiLU -> linear_2.
public final class QwenImage21TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(inChannels: Int, timeEmbedDim: Int) {
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim, bias: false)
        self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

public final class QwenImage21TimestepProjEmbeddings: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: QwenImage21TimestepEmbedding

    public init(embeddingDim: Int) {
        self._timestepEmbedder.wrappedValue = QwenImage21TimestepEmbedding(
            inChannels: 256, timeEmbedDim: embeddingDim)
        super.init()
    }

    /// The reference casts `timestep` to the activation dtype BEFORE the sinusoid
    /// (`timestep.to(hidden_states.dtype)` in the model, `.float()` inside), so a bf16 run
    /// embeds the bf16-rounded sigma. Mirrored here via `dtype`.
    public func callAsFunction(_ timestep: MLXArray, dtype: DType) -> MLXArray {
        let proj = qwenImage21TimestepProj(timestep.asType(dtype)).asType(dtype)
        return timestepEmbedder(proj)
    }
}

/// RMSNorm whose checkpointed weight is `scale - 1` (effective scale = weight + 1), fp32 math.
public final class QwenImage21ZeroCenterRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    public init(dim: Int, eps: Float = 1e-5) {
        self._weight.wrappedValue = MLXArray.zeros([dim])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let rrms = rsqrt(mean(xf * xf, axis: -1, keepDims: true) + eps)
        return (xf * rrms * (weight.asType(.float32) + 1)).asType(x.dtype)
    }
}

/// `txt_in`: zero-centred RMSNorm -> Linear -> GELU(tanh) -> Linear (no biases).
public final class QwenImage21TextProjection: Module {
    @ModuleInfo(key: "text_norm") var textNorm: QwenImage21ZeroCenterRMSNorm
    @ModuleInfo(key: "in_layer") var inLayer: Linear
    @ModuleInfo(key: "out_layer") var outLayer: Linear

    public init(contextInDim: Int, hiddenSize: Int, eps: Float = 1e-6) {
        self._textNorm.wrappedValue = QwenImage21ZeroCenterRMSNorm(dim: contextInDim, eps: eps)
        self._inLayer.wrappedValue = Linear(contextInDim, hiddenSize, bias: false)
        self._outLayer.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        outLayer(geluApproximate(inLayer(textNorm(x))))
    }
}

// MARK: - Feed-forward (SwiGLU)

/// `out(silu(gate_layer(x)) * proj(x))`, mlp_hidden = 3 · dim = 12288.
public final class QwenImage21SwiGLUFeedForward: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "out") var out: Linear
    @ModuleInfo(key: "gate_layer") var gateLayer: Linear

    public init(hiddenSize: Int, mlpHiddenSize: Int) {
        self._proj.wrappedValue = Linear(hiddenSize, mlpHiddenSize, bias: false)
        self._out.wrappedValue = Linear(mlpHiddenSize, hiddenSize, bias: false)
        self._gateLayer.wrappedValue = Linear(hiddenSize, mlpHiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        Self.downProjected(out, silu(gateLayer(x)) * proj(x))
    }

    /// Row-chunked down-projection — the mlx#3797 workaround (fixed upstream by mlx#3810 on
    /// 2026-07-07, NOT yet in any mlx-swift release; latest tag is 0.31.6 from 2026-07-02).
    ///
    /// On mlx-swift ≤ 0.31.6 a half-precision matmul in the window
    /// `M·N ≥ 2048² ∧ K ≥ 10240 ∧ K ≥ 3·max(M,N)` is mis-instantiated on M5-class GPUs.
    /// Here K = 12288, N = 4096, so the window is **1024 ≤ M ≤ 4096 rows** — the cached-decode
    /// pass of every 512²…1024² render (M = target tokens exactly). Row-chunking is exact.
    /// Removal path: when the mlx-swift pin vendors mlx ≥ a8c3e9c, run the fleet NAX probe
    /// (`BooguGate --nax-probe` / `NAXProbeTests`); on PASS call `out` directly.
    /// `QI21_NO_CHUNK=1` disables it for validation.
    static let chunkRows = 896

    static func inNAXWindow(rows: Int, k: Int, n: Int) -> Bool {
        rows * n >= 2048 * 2048 && k >= 10240 && k >= 3 * max(rows, n)
    }

    static func downProjected(_ out: Linear, _ h: MLXArray) -> MLXArray {
        let rows = h.ndim >= 2 ? h.dim(-2) : 0
        let k = h.dim(-1)
        let n = out.weight.dim(0)
        guard rows > chunkRows, h.dtype != .float32,
            Self.inNAXWindow(rows: rows, k: k, n: n),
            ProcessInfo.processInfo.environment["QI21_NO_CHUNK"] != "1"
        else { return out(h) }
        var parts: [MLXArray] = []
        var start = 0
        while start < rows {
            let end = min(start + chunkRows, rows)
            parts.append(out(h[.ellipsis, start..<end, 0...]))
            start = end
        }
        return concatenated(parts, axis: -2)
    }
}

// MARK: - RoPE

/// Complex RoPE (`apply_rotary_emb_qwen(use_real=False)`) as a real interleaved-pair rotation
/// in fp32. x: [B, S, H, D]; cos/sin: [S, D/2].
func qwenImage21ApplyRotary(_ x: MLXArray, cos cosT: MLXArray, sin sinT: MLXArray) -> MLXArray {
    let shape = x.shape
    let pairs = x.asType(.float32).reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
    let xR = pairs[.ellipsis, 0]
    let xI = pairs[.ellipsis, 1]
    let c = cosT[.newAxis, 0..., .newAxis, 0...]
    let s = sinT[.newAxis, 0..., .newAxis, 0...]
    let outR = xR * c - xI * s
    let outI = xR * s + xI * c
    return stacked([outR, outI], axis: -1).reshaped(shape).asType(x.dtype)
}

/// `QwenImage21Rope`: 3-axis (frame, height, width) rotary tables over the joint sequence.
/// Text tokens advance one shared position on all three axes; each image block freezes the
/// frame axis at the position reached by the preceding text, lays its tokens on an h/w grid
/// centred on zero, then advances the position by max(h, w).
public final class QwenImage21Rope {
    public let theta: Int
    public let axesDim: [Int]

    public init(theta: Int = 10000, axesDim: [Int] = [16, 56, 56]) {
        self.theta = theta
        self.axesDim = axesDim
    }

    /// Integer positions per token: (frame, height, width), each of length S.
    public func positions(imgShapes: [(Int, Int, Int)], imagePadMask: [Bool])
        -> (frame: [Int], height: [Int], width: [Int])
    {
        var frame: [Int] = [], hIdx: [Int] = [], wIdx: [Int] = []
        frame.reserveCapacity(imagePadMask.count)
        hIdx.reserveCapacity(imagePadMask.count)
        wIdx.reserveCapacity(imagePadMask.count)
        var cursor = 0
        var position = 0
        let total = imagePadMask.count
        for (_, height, width) in imgShapes {
            var blockStart = cursor
            while blockStart < total, !imagePadMask[blockStart] { blockStart += 1 }
            precondition(blockStart < total, "image block not found in imagePadMask")
            let textLen = blockStart - cursor
            for p in position..<(position + textLen) {
                frame.append(p); hIdx.append(p); wIdx.append(p)
            }
            position += textLen
            cursor = blockStart + height * width
            for h in (-(height - height / 2))..<(height / 2) {
                for w in (-(width - width / 2))..<(width / 2) {
                    frame.append(position); hIdx.append(h); wIdx.append(w)
                }
            }
            position += max(height, width)
        }
        if cursor < total {
            for p in position..<(position + total - cursor) {
                frame.append(p); hIdx.append(p); wIdx.append(p)
            }
        }
        precondition(frame.count == total)
        return (frame, hIdx, wIdx)
    }

    /// (cos, sin) tables [S, sum(axesDim)/2] in fp32 — `polar(1, pos · theta^(-2j/d))`.
    public func callAsFunction(imgShapes: [(Int, Int, Int)], imagePadMask: [Bool])
        -> (cos: MLXArray, sin: MLXArray)
    {
        let (f, h, w) = positions(imgShapes: imgShapes, imagePadMask: imagePadMask)
        let axes = [f, h, w]
        var angles: [MLXArray] = []
        for (axis, dim) in axesDim.enumerated() {
            let invFreq = 1.0 / pow(
                MLXArray(Float(theta)),
                MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) }) / Float(dim))
            let pos = MLXArray(axes[axis].map { Int32($0) }).asType(.float32)
            angles.append(outer(pos, invFreq))
        }
        let all = concatenated(angles, axis: -1)
        return (cos(all), sin(all))
    }
}

// MARK: - Layout (token metadata computed once per request)

/// One run of the prefix with equal `image_ids`: `[start, end)`, text runs are causal.
public struct QwenImage21Segment: Sendable {
    public let start: Int
    public let end: Int
    public let isText: Bool
}

/// The joint text/image sequence layout for one request. Built once by
/// `QwenImage21Transformer2DModel.buildLayout` and reused across every denoise step.
public struct QwenImage21Layout {
    /// Per-image `(frame, h, w)` in latent tokens, condition images first, target LAST.
    public let imgShapes: [(Int, Int, Int)]
    /// Joint-sequence mask, `true` at latent-token positions (each VL image slot -> 2x2 tokens).
    public let imagePadMask: [Bool]
    /// Joint index -> source index into `[text ‖ image]` (the T encoder rows first, then packed latents).
    public let gatherIndex: [Int32]
    /// T — the encoder rows (after drop_idx), i.e. the VL slots before the appended target slots.
    public let encoderLen: Int
    /// Number of joint tokens before the target image block.
    public let prefixLen: Int
    /// Target-image token count (the trailing block).
    public let targetLen: Int
    /// Prefix runs for the exact multi-pass prefill.
    public let segments: [QwenImage21Segment]
    /// RoPE tables over the WHOLE joint sequence [S, 64].
    public let cos: MLXArray
    public let sin: MLXArray

    public var jointLen: Int { imagePadMask.count }
}

// MARK: - Prefix KV cache

public final class QwenImage21KVLayerCache {
    public var k: MLXArray?
    public var v: MLXArray?
    public init() {}
}

public final class QwenImage21KVCache {
    public let layers: [QwenImage21KVLayerCache]
    public init(numLayers: Int) { layers = (0..<numLayers).map { _ in QwenImage21KVLayerCache() } }
    /// All cached arrays (for a pipeline-side `eval` after the prefill step).
    public var arrays: [MLXArray] { layers.flatMap { [$0.k, $0.v].compactMap { $0 } } }
}

public enum QwenImage21KVCacheMode: Sendable {
    /// Full joint forward every step (reference `kv_cache=None`).
    case none
    /// Prefill: full joint forward, store the prefix K/V per layer.
    case extract
    /// Decode: only the target tokens run; keys/values = [cached prefix, target].
    case cached
}

// MARK: - Attention

/// `QwenImage21Attention` + `QwenImage21AttnProcessor` (the exact multi-pass prefill).
public final class QwenImage21Attention: Module {
    let heads: Int
    let dimHead: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    public init(dim: Int, heads: Int, dimHead: Int, eps: Float = 1e-6) {
        self.heads = heads
        self.dimHead = dimHead
        let inner = heads * dimHead
        self._toQ.wrappedValue = Linear(dim, inner, bias: false)
        self._toK.wrappedValue = Linear(dim, inner, bias: false)
        self._toV.wrappedValue = Linear(dim, inner, bias: false)
        self._toOut.wrappedValue = [Linear(inner, dim, bias: false)]
        self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
        super.init()
    }

    /// - x: [B, Sq, D] — the whole joint sequence (prefill / none) or the target only (cached).
    /// - cos/sin: RoPE tables for exactly the rows of `x`.
    public func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, layout: QwenImage21Layout,
        layerCache: QwenImage21KVLayerCache?, mode: QwenImage21KVCacheMode
    ) -> MLXArray {
        let b = x.dim(0)
        let sq = x.dim(1)
        var q = normQ(toQ(x).reshaped(b, sq, heads, dimHead))
        var k = normK(toK(x).reshaped(b, sq, heads, dimHead))
        var v = toV(x).reshaped(b, sq, heads, dimHead)
        q = qwenImage21ApplyRotary(q, cos: cos, sin: sin)
        k = qwenImage21ApplyRotary(k, cos: cos, sin: sin)

        switch mode {
        case .extract:
            layerCache!.k = k[0..., ..<layout.prefixLen]
            layerCache!.v = v[0..., ..<layout.prefixLen]
        case .cached:
            k = concatenated([layerCache!.k!, k], axis: 1)
            v = concatenated([layerCache!.v!, v], axis: 1)
        case .none:
            break
        }

        let qT = q.transposed(0, 2, 1, 3)  // [B, H, Sq, Dh]
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)
        let scale = 1.0 / sqrt(Float(dimHead))

        var out: MLXArray
        if mode == .cached {
            // decode: target rows see the entire prefix + their own block -> full attention.
            out = MLXFast.scaledDotProductAttention(
                queries: qT, keys: kT, values: vT, scale: scale, mask: .none)
        } else {
            // prefill: every prefix segment attends to keys [0, end) — text runs with a causal
            // triangle over their own keys (MLX's causal mode aligns a shorter query block to
            // the END of the key range, i.e. q_i sees keys ≤ start + i) — then the target block
            // attends to everything.
            var parts: [MLXArray] = []
            for seg in layout.segments {
                parts.append(
                    MLXFast.scaledDotProductAttention(
                        queries: qT[0..., 0..., seg.start..<seg.end, 0...],
                        keys: kT[0..., 0..., ..<seg.end, 0...],
                        values: vT[0..., 0..., ..<seg.end, 0...],
                        scale: scale, mask: seg.isText ? .causal : .none))
            }
            parts.append(
                MLXFast.scaledDotProductAttention(
                    queries: qT[0..., 0..., layout.prefixLen..., 0...],
                    keys: kT, values: vT, scale: scale, mask: .none))
            out = parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
        }
        out = out.transposed(0, 2, 1, 3).reshaped(b, sq, heads * dimHead).asType(x.dtype)
        return toOut[0](out)
    }
}

// MARK: - Block

/// Single-stream block. Modulation comes from the parent's shared projection: `mod` is
/// [rows, 4·dim] = [mod1.scale, mod1.gate, mod2.scale, mod2.gate]; row 0 = sampled t,
/// row 1 (when present) = t = 0 for the prefix tokens.
public final class QwenImage21TransformerBlock: Module {
    @ModuleInfo(key: "img_norm1") var imgNorm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: QwenImage21Attention
    @ModuleInfo(key: "img_norm2") var imgNorm2: LayerNorm
    @ModuleInfo(key: "img_mlp") var imgMLP: QwenImage21SwiGLUFeedForward

    public init(dim: Int, numAttentionHeads: Int, attentionHeadDim: Int, mlpRatio: Int = 3,
                eps: Float = 1e-6) {
        self._imgNorm1.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self._attn.wrappedValue = QwenImage21Attention(
            dim: dim, heads: numAttentionHeads, dimHead: attentionHeadDim, eps: eps)
        self._imgNorm2.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self._imgMLP.wrappedValue = QwenImage21SwiGLUFeedForward(
            hiddenSize: dim, mlpHiddenSize: dim * mlpRatio)
        super.init()
    }

    /// `x * (1 + scale)` and `tanh(gate)` per token: rows [0, prefixLen) take the t=0 row
    /// (index 1), the rest the sampled-t row (index 0). `prefixLen == 0` = every token real-t.
    static func modulate(_ x: MLXArray, scale: MLXArray, gate: MLXArray, prefixLen: Int)
        -> (MLXArray, MLXArray)
    {
        let dtype = x.dtype
        if prefixLen == 0 || scale.dim(0) == 1 {
            return (x * (1 + scale[0].asType(dtype)), tanh(gate[0].asType(dtype)))
        }
        let modulated = concatenated(
            [x[0..., ..<prefixLen] * (1 + scale[1].asType(dtype)),
             x[0..., prefixLen...] * (1 + scale[0].asType(dtype))], axis: 1)
        let targetLen = x.dim(1) - prefixLen
        let g = concatenated(
            [broadcast(tanh(gate[1].asType(dtype))[.newAxis, .newAxis, 0...], to: [1, prefixLen, x.dim(-1)]),
             broadcast(tanh(gate[0].asType(dtype))[.newAxis, .newAxis, 0...], to: [1, targetLen, x.dim(-1)])],
            axis: 1)
        return (modulated, g)
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray, modulation: MLXArray, cos: MLXArray, sin: MLXArray,
        layout: QwenImage21Layout, prefixLen: Int,
        layerCache: QwenImage21KVLayerCache?, mode: QwenImage21KVCacheMode
    ) -> MLXArray {
        let mods = split(modulation, parts: 4, axis: -1)  // scale1, gate1, scale2, gate2
        var h = hiddenStates
        let (m1, g1) = Self.modulate(imgNorm1(h), scale: mods[0], gate: mods[1], prefixLen: prefixLen)
        let a = attn(m1, cos: cos, sin: sin, layout: layout, layerCache: layerCache, mode: mode)
        h = h + g1 * a
        let (m2, g2) = Self.modulate(imgNorm2(h), scale: mods[2], gate: mods[3], prefixLen: prefixLen)
        h = h + g2 * imgMLP(m2)
        return h
    }
}

/// Final AdaLN — scale only: `LN(x) * (1 + linear(silu(temb)))`, rows selected like the blocks.
public final class QwenImage21AdaLayerNormContinuous: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    public init(embeddingDim: Int, conditioningEmbeddingDim: Int, eps: Float = 1e-6) {
        self._linear.wrappedValue = Linear(conditioningEmbeddingDim, embeddingDim, bias: false)
        self._norm.wrappedValue = LayerNorm(dimensions: embeddingDim, eps: eps, affine: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, conditioning: MLXArray, prefixLen: Int) -> MLXArray {
        let scale = linear(silu(conditioning).asType(x.dtype))  // [rows, dim]
        let n = norm(x)
        if prefixLen == 0 || scale.dim(0) == 1 { return n * (1 + scale[0]) }
        return concatenated(
            [n[0..., ..<prefixLen] * (1 + scale[1]), n[0..., prefixLen...] * (1 + scale[0])], axis: 1)
    }
}

// MARK: - Top-level model

public final class QwenImage21Transformer2DModel: Module {
    public let patchSize: Int
    public let inChannels: Int
    public let outChannels: Int
    public let innerDim: Int
    public let numLayers: Int
    public let causalCondition: Bool

    public let posEmbed: QwenImage21Rope  // plain class — tables are computed, not checkpointed

    @ModuleInfo(key: "time_text_embed") var timeTextEmbed: QwenImage21TimestepProjEmbeddings
    @ModuleInfo(key: "txt_in") var txtIn: QwenImage21TextProjection
    @ModuleInfo(key: "img_in") var imgIn: Linear
    /// upstream `Sequential(SiLU, Linear)` -> key `modulation.1.weight`, sanitized to `modulation.`
    @ModuleInfo(key: "modulation") var modulation: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [QwenImage21TransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: QwenImage21AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    /// Break the lazy graph after every block (the fleet's long-graph fused-dispatch lever;
    /// see qwen-image-edit-swift `chainBlockGraphs`). Off by default.
    public var chainBlockGraphs = false

    /// Parity-gate tap: called with (blockIndex, blockOutput) after every block, and with
    /// (-1, joint input) / (-2, temb) / (-3, modulation) before the loop. nil in production.
    public var blockTap: ((Int, MLXArray) -> Void)?

    public init(
        patchSize: Int = 1, inChannels: Int = 64, outChannels: Int? = 64, numLayers: Int = 32,
        attentionHeadDim: Int = 128, numAttentionHeads: Int = 32, contextInDim: Int = 4096,
        mlpRatio: Int = 3, axesDimsRope: [Int] = [16, 56, 56], eps: Float = 1e-6,
        causalCondition: Bool = true
    ) {
        let inner = numAttentionHeads * attentionHeadDim
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.outChannels = outChannels ?? inChannels
        self.innerDim = inner
        self.numLayers = numLayers
        self.causalCondition = causalCondition
        self.posEmbed = QwenImage21Rope(theta: 10000, axesDim: axesDimsRope)
        self._timeTextEmbed.wrappedValue = QwenImage21TimestepProjEmbeddings(embeddingDim: inner)
        self._txtIn.wrappedValue = QwenImage21TextProjection(
            contextInDim: contextInDim, hiddenSize: inner, eps: eps)
        self._imgIn.wrappedValue = Linear(inChannels * patchSize * patchSize, inner, bias: false)
        self._modulation.wrappedValue = Linear(inner, 4 * inner, bias: false)
        self._transformerBlocks.wrappedValue = (0..<numLayers).map { _ in
            QwenImage21TransformerBlock(
                dim: inner, numAttentionHeads: numAttentionHeads,
                attentionHeadDim: attentionHeadDim, mlpRatio: mlpRatio, eps: eps)
        }
        self._normOut.wrappedValue = QwenImage21AdaLayerNormContinuous(
            embeddingDim: inner, conditioningEmbeddingDim: inner, eps: eps)
        self._projOut.wrappedValue = Linear(
            inner, patchSize * patchSize * (outChannels ?? inChannels), bias: false)
        super.init()
    }

    /// Label the joint sequence (reference `forward` preamble + `build_token_metadata` +
    /// `_qwenimage21_prefix_segments` + `pos_embed`).
    ///
    /// - imgMask: the vision-language sequence's image-slot mask (`true` per `<|image_pad|>`
    ///   left after drop_idx) with ONE `true` appended per 2x2 group of target latents.
    /// - imgShapes: per-image `(frame, h, w)` latent grids, condition images first, target last.
    public func buildLayout(imgMask: [Bool], imgShapes: [(Int, Int, Int)]) throws -> QwenImage21Layout {
        var imagePadMask: [Bool] = []
        imagePadMask.reserveCapacity(imgMask.count * 4)
        for slot in imgMask {
            if slot { imagePadMask.append(contentsOf: [true, true, true, true]) } else { imagePadMask.append(false) }
        }
        let total = imagePadMask.count
        let blockLengths = imgShapes.map { $0.0 * $0.1 * $0.2 }
        let imageCount = imagePadMask.lazy.filter { $0 }.count
        guard blockLengths.reduce(0, +) == imageCount else {
            throw QwenImage21Error.invalidInput(
                "img_shapes account for \(blockLengths.reduce(0, +)) image tokens but the mask marks \(imageCount)")
        }
        // image ids + gather index. The reference builds the joint sequence as
        // `cat([encoder_hidden_states (T rows, INCLUDING the VL pad rows), zeros(target slots)])`
        // expanded 4x at image slots and then OVERWRITES every image position with the packed
        // latents — so a text position reads its own VL slot row (slot index < T), and the i-th
        // image position (sequence order) reads packed-latent row i, addressed at T + i in the
        // `[text ‖ image]` source. (Indexing images at `number-of-text-positions + i` was the
        // first port's bug: correct only when the prompt has no image pads, i.e. T2I.)
        let targetLen = blockLengths.last ?? 0
        let encoderLen = imgMask.count - targetLen / 4  // T: VL rows before the appended target slots
        var imageIds = [Int](repeating: -1, count: total)
        var gather = [Int32](repeating: 0, count: total)
        var imgIdx: Int32 = 0
        var block = 0
        var remainingInBlock = blockLengths.isEmpty ? 0 : blockLengths[0]
        var pos = 0
        for (slot, isImage) in imgMask.enumerated() {
            if isImage {
                for _ in 0..<4 {
                    while remainingInBlock == 0, block + 1 < blockLengths.count {
                        block += 1; remainingInBlock = blockLengths[block]
                    }
                    imageIds[pos] = block
                    remainingInBlock -= 1
                    gather[pos] = Int32(encoderLen) + imgIdx
                    imgIdx += 1
                    pos += 1
                }
            } else {
                precondition(slot < encoderLen, "text slot beyond the encoder rows")
                gather[pos] = Int32(slot)
                pos += 1
            }
        }
        precondition(pos == total)
        let prefixLen = total - targetLen
        // the target block must be the trailing run (the pipeline appends its slots last)
        for i in prefixLen..<total where imageIds[i] != blockLengths.count - 1 {
            throw QwenImage21Error.invalidInput("target image block is not the trailing run of the joint sequence")
        }
        var segments: [QwenImage21Segment] = []
        var start = 0
        for i in 1...max(prefixLen, 1) where prefixLen > 0 {
            if i == prefixLen || imageIds[i] != imageIds[start] {
                segments.append(QwenImage21Segment(start: start, end: i, isText: imageIds[start] < 0))
                start = i
            }
        }
        let (c, s) = posEmbed(imgShapes: imgShapes, imagePadMask: imagePadMask)
        return QwenImage21Layout(
            imgShapes: imgShapes, imagePadMask: imagePadMask, gatherIndex: gather, encoderLen: encoderLen,
            prefixLen: prefixLen, targetLen: targetLen, segments: segments, cos: c, sin: s)
    }

    /// - hiddenStates: [B, L_img, 64] packed latents — condition images first, target LAST.
    ///   In `.cached` mode only the trailing `layout.targetLen` rows are used.
    /// - encoderHiddenStates: [B, T, 4096] pre-norm Qwen3-VL features (after drop_idx).
    /// - timestep: [B] sigma in 0…1 (the reference's `timestep / 1000`).
    /// Returns the joint output [B, S, 64] (`.none` / `.extract`) or the target rows only
    /// [B, targetLen, 64] (`.cached`). Callers take the LAST `targetLen` rows either way.
    public func callAsFunction(
        hiddenStates: MLXArray, encoderHiddenStates: MLXArray, timestep: MLXArray,
        layout: QwenImage21Layout, kvCache: QwenImage21KVCache? = nil,
        mode: QwenImage21KVCacheMode = .none
    ) -> MLXArray {
        precondition(hiddenStates.dim(0) == 1, "batch size 1 only")
        precondition(mode == .none || kvCache != nil, "kv cache required for extract/cached")
        precondition(!(mode != .none) || causalCondition, "kv cache requires causal_condition")
        let dtype = hiddenStates.dtype
        let t = timestep.asType(dtype)

        // temb rows: [t] + [0] (causal_condition) -> shared modulation [rows, 4·dim]
        let tRows = causalCondition && mode != .cached ? concatenated([t, t * 0], axis: 0) : t
        let temb = timeTextEmbed(tRows, dtype: dtype)
        let mod = modulation(silu(temb))

        var joint: MLXArray
        var cos = layout.cos
        var sin = layout.sin
        let prefixLen: Int
        if mode == .cached {
            let target = hiddenStates[0..., (hiddenStates.dim(1) - layout.targetLen)...]
            joint = imgIn(target)
            cos = cos[layout.prefixLen...]
            sin = sin[layout.prefixLen...]
            prefixLen = 0
        } else {
            let img = imgIn(hiddenStates)                 // [1, L_img, D]
            let txt = txtIn(encoderHiddenStates)          // [1, T, D]
            precondition(
                txt.dim(1) + img.dim(1) == layout.encoderLen + layout.imgShapes.reduce(0) { $0 + $1.0 * $1.1 * $1.2 }
                    && txt.dim(1) == layout.encoderLen,
                "encoder rows \(txt.dim(1)) / image tokens \(img.dim(1)) do not match the layout (T = \(layout.encoderLen))")
            let source = concatenated([txt, img], axis: 1)
            joint = take(source, MLXArray(layout.gatherIndex), axis: 1)
            prefixLen = causalCondition ? layout.prefixLen : 0
        }

        blockTap?(-1, joint)
        blockTap?(-2, temb)
        blockTap?(-3, mod)
        for (i, block) in transformerBlocks.enumerated() {
            joint = block(
                joint, modulation: mod, cos: cos, sin: sin, layout: layout, prefixLen: prefixLen,
                layerCache: kvCache?.layers[i], mode: mode)
            if chainBlockGraphs { eval(joint) }
            blockTap?(i, joint)
        }
        joint = normOut(joint, conditioning: temb, prefixLen: prefixLen)
        return projOut(joint)
    }
}
