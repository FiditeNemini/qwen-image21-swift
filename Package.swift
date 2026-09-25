// swift-tools-version: 6.2
// qwen-image21-swift — Swift/MLX port of Qwen/Qwen-Image-2.1 (Qwen RESEARCH licence — research /
// evaluation only, see PORTING-SPEC.md §1): a 7B single-stream block-causal DiT (32 layers, prefix
// KV cache), a 64-ch 16x RGBA VAE, and the Qwen3-VL-8B conditioner (byte-identical to
// Qwen/Qwen3-VL-8B-Instruct — served by the fleet's qwen3vl-mlx-swift backbone).
// Reference = diffusers main `QwenImage21Pipeline` (huggingface/diffusers#14804, 2026-09-20);
// goldens from mlxengine-image/WIP/qwen-image21-oracle (fp32 CPU torch).

import PackageDescription

let package = Package(
    name: "QwenImage21",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "QwenImage21", targets: ["QwenImage21"]),
        // MLXEngine wrapper: textToImage + imageEdit on one core (PackageID qwen-image-2.1).
        // C7 = LicenseRef-Qwen-Research (package-local, never allowlisted) — research tier.
        .library(name: "MLXQwenImage21", targets: ["MLXQwenImage21"]),
        .executable(name: "QwenImage21Gate", targets: ["QwenImage21Gate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.4"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        // Qwen3-VL conditioner backbone (≥ 0.3.0 for lastHiddenState(applyFinalNorm:), the
        // pre-final-norm feature Qwen-Image-2.1 conditions on). Tagged 2026-09-20.
        .package(url: "https://github.com/xocialize/qwen3vl-mlx-swift", from: "0.3.0"),
        // MLXEngine contract (MLXToolKit) + the executable MAT/CAN gates, for the wrapper only.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.56.0"),
    ],
    targets: [
        .target(
            name: "QwenImage21",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Qwen3VL", package: "qwen3vl-mlx-swift"),
            ],
            path: "Sources/QwenImage21"
        ),
        .target(
            name: "MLXQwenImage21",
            dependencies: [
                "QwenImage21",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/MLXQwenImage21"
        ),
        .executableTarget(
            name: "QwenImage21Gate",
            dependencies: ["QwenImage21", .product(name: "MLX", package: "mlx-swift"),
                           .product(name: "MLXRandom", package: "mlx-swift")],
            path: "Sources/QwenImage21Gate"
        ),
        .testTarget(
            name: "QwenImage21Tests",
            dependencies: [
                "QwenImage21",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Tests/QwenImage21Tests"
        ),
        .testTarget(
            name: "MLXQwenImage21Tests",
            dependencies: [
                "MLXQwenImage21",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXQwenImage21Tests"
        ),
    ]
)
