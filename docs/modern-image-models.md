# How modern image models work, and what a small fine-tune can copy (Oct 2026)

## The frontier models

**GPT-4o image generation (OpenAI, Mar 2025).** Image generation is native to
the language model, not a bolted-on diffusion model. OpenAI: trained on the
joint distribution of online images and text, plus aggressive post-training.
Observed generation (blurry full image first, then sharpening top-to-bottom in
raster order) is consistent with autoregressive token prediction with KV
caching, possibly MAR-style continuous tokens with a lightweight diffusion
head. Practical wins: accurate text rendering, binding 10-20 objects to their
attributes, multi-turn conversational editing, in-context learning from
uploaded images.
Sources: https://openai.com/index/introducing-4o-image-generation/ ,
https://www.philipnielsen.info/blog/blog_posts/4o-image.md

**Gemini 2.5 Flash Image "nano banana" (Google, Aug 2025).** Same direction:
native multimodal generation inside Gemini. Strengths: subject consistency
across prompts, natural-language local edits, multi-image fusion, world
knowledge. Google says it is still working to improve long-form text
rendering.
Source: https://developers.googleblog.com/en/introducing-gemini-2-5-flash-image/

**The common thread.** The capability jump came from (1) generating images
inside the language model so prompts are read with full world knowledge,
(2) massive paired image-text training data, (3) heavy post-training. There
is no small trick here a LoRA can copy.

**Qwen-Image technical report (Aug 2025) - the open model closest to this.**
MMDiT architecture; its text-rendering lead came mostly from data
engineering: a multi-stage curriculum with progressively harder rendered-text
images (synthetic text on clean, then textured, then photographic
backgrounds) and a serious captioning pipeline. THIS part is copyable at
small scale.
Source: https://arxiv.org/abs/2508.02324

## What transfers to a free-T4 LoRA

1. Curriculum over volume: stage training data from clean flat art to busier
   scenes (Qwen-style progression), not one mixed bucket.
2. Synthetic data: render training images programmatically
   (src/synth_posters.py) and caption them with a fixed pattern (style,
   subject, composition, palette, empty-zone note) so attributes stay
   controllable by name.
3. Keep training images free of lettering so the LoRA never learns fake text.
4. A LoRA targets style consistency only; exact text accuracy is not a
   realistic training goal at this scale (even Google calls long-form text
   unsolved).
5. Image conditioning (IP-Adapter / img2img) is the open-model analog of
   Gemini's multi-image fusion, and the way to preserve a real input subject.

Base model: Qwen-Image-2.1-Turbo (7B). Research license - fine for in-house
training experiments; any commercial shipping would need the license
question resolved separately.
