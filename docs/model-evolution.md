# How image generation models evolved, and what we implement from it (Oct 2026)

Companion to docs/modern-image-models.md (which covers the current frontier).
This one traces the arc and lists exactly which ideas this repo adopts.

## The arc

1. **DDPM (Ho et al., 2020).** Denoising diffusion: learn to reverse a slow
   noising process, pixel space, U-Net. Beautiful samples, slow and expensive.
2. **Latent diffusion / Stable Diffusion (Rombach et al., 2022).** Move the
   diffusion into a VAE's latent space (~8x smaller), condition on CLIP text
   embeddings. This is what made open image models practical - and what SD
   1.5/SDXL fine-tunes and LoRAs build on.
3. **DALL-E 3 (OpenAI, 2023).** The lesson was data, not architecture:
   recaptioning training images with long, accurate synthetic captions fixed
   prompt-following more than any model change.
4. **DiT (Peebles & Xie, 2023).** Replace the U-Net with a plain transformer
   operating on latent patches. Scales like an LLM.
5. **SD3 / rectified flow (Esser et al., 2024).** Flow-matching objective
   (straight-line interpolation between noise and data) + multimodal DiT that
   lets text and image tokens attend to each other. Faster sampling, better
   prompt binding.
6. **Qwen-Image (Aug 2025).** MMDiT open model whose text-rendering lead came
   from data engineering: a progressive curriculum from no-text to
   paragraph-level rendered text, plus a heavy captioning pipeline.
   (https://arxiv.org/abs/2508.02324)
7. **GPT-4o image generation (Mar 2025) and Gemini 2.0 -> 2.5 Flash Image
   (2025).** Image generation moved inside the language model: one model
   reads and writes text and images natively. Wins: world knowledge in
   prompts, conversational multi-turn editing, subject consistency,
   multi-image fusion. Capability comes from joint image-text training at
   LLM scale plus post-training - not a trick.

## What this repo implements

| Idea | Source | Where |
|---|---|---|
| Two-phase curriculum (clean -> busy scenes) | Qwen-Image TR | runner notebook phases + synth_posters stages |
| Fully synthetic, structured captions (style/subject/composition/palette/zone) | DALL-E 3 recaptioning lesson | synth_posters.py captions |
| Flow-matching loss on the base's own sigmas | SD3 / Qwen-Image | train_lora.py |
| Min-SNR-style loss weighting across noise levels | Min-SNR (Hang et al., 2023) | train_lora.py --min-snr-gamma |
| EMA of trained weights for eval/save | standard diffusion practice | train_lora.py --ema-decay |
| Mid-sigma sampling bias (avoid near-pure-noise steps) | common flow-matching practice | train_lora.py --sigma-bias |
| Warmup + cosine LR decay | standard transformer practice | train_lora.py --warmup |

## Deliberately not copied (scale reasons)

- Native autoregressive/unified generation (GPT-4o/Gemini): needs LLM-scale
  joint training; a LoRA cannot inject it.
- Text-rendering training: Qwen needed a multi-stage rendered-text curriculum
  at full-model scale; at LoRA scale the base forgets nothing and learns
  little. Lettering stays out of training data instead.
- RLHF-style post-training: no reward model or budget at this scale.
