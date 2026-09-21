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
