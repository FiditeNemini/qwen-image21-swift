// Qwen3-VL-8B prompt encoder for Qwen-Image-2.1 — the diffusers `QwenImage21Pipeline`
// `_get_qwen_prompt_embeds` path on the fleet's `qwen3vl-mlx-swift` backbone.
//
// Facts pinned by the oracle goldens (qwen-image21-oracle/goldens/encoder_*.json):
//   - raw template strings, NOT apply_chat_template; system prompt "Comprehend and analyze the
//     provided prompt."; drop_idx = 14 = the tokenized system block (derived at init here too);
//   - conditioning images enter as `<image{i}><|vision_start|><|image_pad|><|vision_end|>`
//     blocks (space-separated from the second on), each `<|image_pad|>` expanded to the merged
//     vision-token count; the VL copy is the LANCZOS-resized RGBA composited over WHITE;
//   - the feature is the last decoder layer's output BEFORE the final RMSNorm
//     (`applyFinalNorm: false`; |x| mean ≈ 9 vs ≈ 1.5 after the norm);
//   - position ids are real 3-D M-RoPE (get_rope_index): image tokens share the frame index and
//     span an h/w grid; text after an image resumes at max + 1. The backbone computes this.
// The text encoder weights are byte-identical to Qwen/Qwen3-VL-8B-Instruct (per-tensor check,
// oracle/text_encoder_identity.json), so the stock snapshot is loaded.

import Foundation
import MLX
import MLXLMCommon
import Qwen3VL
import Tokenizers

public final class QwenImage21PromptEncoder {
    public let model: Qwen3VL
    public let tokenizer: any Tokenizers.Tokenizer
    public let processor: Qwen3VLImageProcessor
    public let imagePadId: Int
    public let dropIdx: Int
    public let dtype: DType

    public static let sysPrompt = "Comprehend and analyze the provided prompt."

    public static func templateT2I(_ prompt: String) -> String {
        "<|im_start|>system\n\(sysPrompt)<|im_end|>\n<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
    }

    public static func templateTI2I(_ prompt: String, imageCount: Int) -> String {
        var block = "<image1><|vision_start|><|image_pad|><|vision_end|>"
        if imageCount > 1 {
            for i in 2...imageCount { block += " <image\(i)><|vision_start|><|image_pad|><|vision_end|>" }
        }
        return "<|im_start|>system\n\(sysPrompt)<|im_end|>\n<|im_start|>user\n\(block)\(prompt)<|im_end|>\n<|im_start|>assistant\n"
    }

    public init(model: Qwen3VL, tokenizer: any Tokenizers.Tokenizer, dtype: DType) throws {
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = dtype
        // Qwen3-VL processor defaults == the 2.1 snapshot's preprocessor_config.json
        // (patch 16, merge 2, min 256², max 4096², mean/std 0.5).
        self.processor = Qwen3VLImageProcessor()
        guard let pad = tokenizer.convertTokenToId("<|image_pad|>") else {
            throw QwenImage21Error.loading("tokenizer lacks <|image_pad|>")
        }
        self.imagePadId = pad
        // Reference: `len(processor.apply_chat_template(system_message, tokenize=True))`.
        self.dropIdx = tokenizer.encode(
            text: "<|im_start|>system\n\(Self.sysPrompt)<|im_end|>\n", addSpecialTokens: false).count
    }

    /// Load the stock Qwen3-VL-8B snapshot (weights + tokenizer). `tokenizerDir` may point at the
    /// 2.1 snapshot's `processor/` folder; the vocab/merges are identical either way.
    public static func load(qwenDir: URL, tokenizerDir: URL? = nil, dtype: DType = .bfloat16,
                            loadOnCPU: Bool = true) async throws -> QwenImage21PromptEncoder {
        var model: Qwen3VL!
        if loadOnCPU {
            try Device.withDefaultDevice(.cpu) { model = try Qwen3VLLoader.load(directory: qwenDir, dtype: dtype) }
        } else {
            model = try Qwen3VLLoader.load(directory: qwenDir, dtype: dtype)
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir ?? qwenDir)
        return try QwenImage21PromptEncoder(model: model, tokenizer: tokenizer, dtype: dtype)
    }

    public struct Encoded {
        /// [1, T, 4096] pre-norm features after drop_idx.
        public let embeds: MLXArray
        /// `true` where the (dropped) token is an `<|image_pad|>` — one per merged 2x2 patch.
        public let imagePadMask: [Bool]
        /// Merged vision grids (t, h/merge, w/merge) per image, in order.
        public let mergedGrids: [(Int, Int, Int)]
        public let inputIds: [Int]
    }

    /// Encode a prompt with zero or more ALREADY-RESIZED condition images (the pipeline's
    /// `input_images`: LANCZOS to the /32 size derived from `output_resolution`).
    public func encode(prompt: String, images: [QwenImage21RGBAImage] = []) throws -> Encoded {
        let text = prompt.isEmpty ? " " : prompt  // Qwen has no BOS; an empty prompt leaves nothing to read
        var pixelParts: [MLXArray] = []
        var grids: [THW] = []
        var merged: [(Int, Int, Int)] = []
        for img in images {
            let (pv, thw) = processor.preprocess(rgb: img.compositedOverWhiteRGB(), width: img.width, height: img.height)
            pixelParts.append(pv)
            grids.append(thw)
            merged.append((thw.t, thw.h / processor.mergeSize, thw.w / processor.mergeSize))
        }
        let template = images.isEmpty ? Self.templateT2I(text) : Self.templateTI2I(text, imageCount: images.count)
        var ids = tokenizer.encode(text: template, addSpecialTokens: false)
        // Expand each `<|image_pad|>` occurrence (in order) to that image's merged token count.
        var cursor = 0
        for (t, h, w) in merged {
            guard let idx = ids[cursor...].firstIndex(of: imagePadId) else {
                throw QwenImage21Error.invalidInput("template lacks an <|image_pad|> for image \(merged.count)")
            }
            let count = t * h * w
            ids.replaceSubrange(idx...idx, with: Array(repeating: imagePadId, count: count))
            cursor = idx + count
        }
        let inputIds = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let hidden: MLXArray
        if pixelParts.isEmpty {
            hidden = try model.lastHiddenState(inputIds: inputIds, applyFinalNorm: false)
        } else {
            let pixels = (pixelParts.count == 1 ? pixelParts[0] : concatenated(pixelParts, axis: 0)).asType(dtype)
            hidden = try model.lastHiddenState(
                inputIds: inputIds, pixelValues: pixels, imageGridTHW: grids, applyFinalNorm: false)
        }
        guard ids.count > dropIdx else { throw QwenImage21Error.invalidInput("prompt tokenized shorter than drop_idx") }
        let embeds = hidden[0..., dropIdx..., 0...]
        let mask = ids[dropIdx...].map { $0 == imagePadId }
        return Encoded(embeds: embeds, imagePadMask: mask, mergedGrids: merged, inputIds: ids)
    }
}
