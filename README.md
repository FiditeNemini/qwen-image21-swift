# qwen-image21-swift

Swift/MLX port of [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) — a unified
text-to-image + multi-reference editing model: 7B single-stream block-causal DiT with a prefix KV
cache, 64-ch 16× **RGBA** VAE (native transparency), Qwen3-VL-8B conditioner.

> **Licence: Qwen RESEARCH License — research / evaluation only, no commercial use.** This port
> is a research tier; see `PORTING-SPEC.md` §1 for the fleet posture (never allowlisted; blocked
> under `.permissiveOnly` + `.blocking`).

Reference: diffusers main `QwenImage21Pipeline` (PR #14804). Spec, architecture delta vs the
2511 port, reuse map and parity plan: `PORTING-SPEC.md`. Goldens + oracle:
`../qwen-image21-oracle`.

> Known reference-side issue: the reference pipeline's image editing degrades at its default
> `output_resolution=1024` (512/768 are correct) — the port reproduces it faithfully, so the engine
> wrapper defaults edits to 768². Tracked upstream: https://github.com/huggingface/diffusers/issues/14824.

```
swift build
.build/debug/QwenImage21Gate --sched  ../qwen-image21-oracle/goldens
.build/debug/QwenImage21Gate --vae    ../../weights/Qwen-Image-2.1 ../qwen-image21-oracle/goldens
.build/debug/QwenImage21Gate --encoder ../../weights/Qwen3-VL-8B-Instruct ../qwen-image21-oracle/goldens
.build/debug/QwenImage21Gate --dit    ../../weights/Qwen-Image-2.1 ../qwen-image21-oracle/goldens
.build/debug/QwenImage21Gate --generate ../../weights/Qwen-Image-2.1 ../../weights/Qwen3-VL-8B-Instruct \
    --prompt "a red fox in fresh snow" --size 1024 --steps 40 --out fox.png
```

Weights: `transformer/` + `vae/` + `processor/` + `scheduler/` from the 2.1 repo
(`weights/Qwen-Image-2.1`); the text encoder is byte-identical to `Qwen/Qwen3-VL-8B-Instruct`
and is loaded from that snapshot.

## GPU numerics: the VAE's 3×3 convs (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses about
6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32.

This VAE hits it in 33 decoder convs at 1024² (conv_in 64→1152, the 1152/576/288-channel resnets
and upsamplers) and 21 convs per edit-image encode. Coverage from the existing gates was thin:
`--vae` runs on the CPU lane, and `--vae-tile` only compares GPU against GPU.

Every stride-1 3×3 conv is now a `WinogradFreeConv2d`. **Default `.conv3d` for both encoder and
decoder** (`encoderConvRoute` / `decoderConvRoute`, type `QwenImage21VAEConvRoute`).

Measurements, against the torch fp32 golden (320² / 288×384 real images) and the CPU lane (1024²
DIV2K photo, alpha 1):

| | Raw conv2d (Winograd) | conv3d route |
|---|---|---|
| Golden `vae_img_b` decode | 1.2e-2 · **max 2.0** (47 dB) | 4.5e-5 · max 2.8e-3 (96 dB) |
| Golden `vae_img_a` decode | 2.7e-4 · 79.9 dB | 8.7e-6 · 109.7 dB |
| Golden encode (a / b) | 7.1e-4 / 1.7e-3 | 3.0e-4 / 3.0e-4 |
| 1024² encode vs CPU lane | 5.6e-3 · max 1.13 | 4.9e-4 |
| 1024² decode vs CPU lane | 1.1e-3 · 69 dB | 2.1e-5 · 103 dB |
| GPU halo-tiled (2×2, halo 12) vs untiled, 1024² | 5.0e-4 · max 3.5e-2 | **exactly 0** |
| 1024² time, decode / encode | 1586 ms / 413 ms | +1253 ms / +142 ms |

**Why the decoder routes here, when the FLUX-class RGB decoders don't:**

- The raw path's worst errors land in the alpha channel at transparent image edges, and they depend
  on context. On `vae_img_b` the top-edge alpha is off by 0.32 in a fresh process, but by 1.996 (a
  full −1 → +1 flip) once any CPU-lane forward has run in the same process.
- Neither isolated conv probes nor a stale-buffer poison test reproduce this; `testRawDecodeBisect`
  is the repro.
- The route is stable in every context.
- It also makes the GPU halo-tiled decode bit-identical to the untiled one. On the CPU stream this
  was already exact.

This corrects the attribution in AB-R-0310. The 64–67 dB GPU tiled-vs-untiled gap was this
Winograd window: its 6×6 tiles realign at tile edges and its GEMM runs TF32. It was not generic
"GPU accumulation order".

Remaining encode residual: the ~3e-4 left on the encoder is TF32 in the mid-block attention, the
same effect measured in the other fleet VAEs.

Controls and tests:

- Environment override: `QWEN21_VAE_CONV_ROUTE=winograd|conv3d|fp32Winograd`.
- `swift test --filter WinogradProbeTests` is weight-free.
- `QWEN21_PARITY=1 QWEN21_ROOT=<weights/Qwen-Image-2.1> swift test -c release -Xswiftc
  -enable-testing --filter VAEGPULaneTests` covers the goldens, the 1024² photo, tiling and timing.
