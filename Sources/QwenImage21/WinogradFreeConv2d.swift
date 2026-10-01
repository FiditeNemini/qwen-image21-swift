// Route for 3×3 convs inside mlx's Winograd conv2d window (mlx-swift ≤ 0.31.6 Metal numerics).
//
// mlx's Metal conv2d (mlx/backend/metal/conv.cpp `dispatch_conv_2D_gpu`) runs a Winograd
// F(6×6,3×3) kernel when ALL of these hold: kernel 3×3, stride 1, dilation 1, groups 1,
// C % 32 == 0, O % 32 == 0, C + O ≥ 256, N·H·W ≥ 4096. On M5 that path is lossy — relL2 per
// conv against an exact reference: fp32 6.4e-3 (its batched GEMM runs TF32, MLX_ENABLE_TF32
// defaults on, and the output transform amplifies that ~8×), bf16 5.8e-2, fp16 7.5e-3. Every
// other conv path is exact-class (fp32 ~1e-6; bf16 1.7e-3 = output rounding). conv3d with
// kT = 1 is the same conv on the implicit-GEMM path: exact, but 1.3–4× slower than Winograd at
// 256/512-channel shapes.
//
// The Qwen-Image-2.1 RGBA VAE hits the window in most 3×3 convs: 33 decoder convs at 1024²
// (conv_in 64→1152, the 1152/576/288-ch resnets and upsamplers) and 21 per edit-image encode.
// Default `.conv3d` for both: the encoder's raw loss is material for edit-image latents, and the
// decoder's raw path gives large, context-dependent alpha errors at transparent edges (up to a
// full-range flip on a real golden). `.conv3d` also makes the GPU halo-tiled decode equal the
// untiled one, as it already is on the CPU stream (AB-R-0310).
// Probe: `swift test --filter WinogradProbeTests` (weight-free). Removal: when the probe reports
// raw conv2d exact on a new mlx-swift pin, go back to plain Conv2d.
// `QWEN21_VAE_CONV_ROUTE=winograd|conv3d|fp32Winograd` overrides the defaults (validation).
//
// The routing lives in MLXExactConv (mlx-exact-conv-swift), shared across the fleet. Its exact path
// holds on mlx-swift 0.31.x and 0.32.x; the old kT = 1 conv3d form went back to Winograd on 0.32
// (mlx#3785); a real opt-out is requested in mlx#4595.

import Foundation
import MLX
import MLXExactConv
import MLXNN

/// How a 3×3 conv inside mlx's Winograd window runs. Shapes outside the window always take plain
/// conv2d, which is mlx's exact implicit-GEMM path.
public enum QwenImage21VAEConvRoute: String, Sendable {
    /// mlx's default Winograd kernel — fastest; on M5 ~6.4e-3 relL2 per conv in fp32, ~5.8e-2 in bf16.
    case winograd
    /// Exact implicit-GEMM path (MLXExactConv); the name is kept from the original kT = 1 conv3d route.
    case conv3d
    /// Half-precision input upcast to fp32 for the Winograd kernel and the result cast back:
    /// ~6.8e-3 per conv instead of bf16's ~5.8e-2, at fp32-Winograd speed. `.winograd` for fp32.
    case fp32Winograd

    /// `QWEN21_VAE_CONV_ROUTE` = winograd | conv3d | fp32Winograd, if set.
    static var environmentOverride: QwenImage21VAEConvRoute? {
        getenv("QWEN21_VAE_CONV_ROUTE").flatMap { QwenImage21VAEConvRoute(rawValue: String(cString: $0)) }
    }

    var exactConvRoute: ExactConvRoute {
        switch self {
        case .winograd: .winograd
        case .conv3d: .exact
        case .fp32Winograd: .fp32Winograd
        }
    }
}

final class WinogradFreeConv2d: Conv2d {
    var route: QwenImage21VAEConvRoute = .conv3d

    /// mlx's Winograd dispatch predicate, evaluated on the actual input (NHWC).
    static func takesWinograd(
        input x: MLXArray, weight: MLXArray, stride: (Int, Int), dilation: (Int, Int),
        groups: Int
    ) -> Bool {
        ExactConv.takesWinograd2D(
            input: x, weight: weight, stride: stride, dilation: dilation, groups: groups)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        ExactConv.conv2d(
            x, weight: weight, bias: bias, stride: stride, padding: padding, dilation: dilation,
            groups: groups, route: route.exactConvRoute)
    }
}
