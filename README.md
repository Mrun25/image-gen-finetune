# image-gen-finetune

Fine-tuning an open image model on free Kaggle GPUs, with the training-data
curriculum treated as the main lever.

## What's here

- `docs/modern-image-models.md` - study notes on how current image models
  (GPT-4o image generation, Gemini 2.5 Flash Image, Qwen-Image) work and what
  a small fine-tune can and cannot copy from them.
- `src/synth_posters.py` - synthetic training-data generator: renders
  flat-illustration poster art with PIL (named palettes, 5 layouts, simple
  focal subjects, clear empty zones), paired with structured captions.
- `src/train_lora.py` - Qwen-Image-2.1-Turbo LoRA trainer (4bit nf4 + PEFT,
  flow-matching loss, two-phase curriculum over stage-1/stage-2 data).
- `kaggle/` - thin runner notebook that clones this repo into a Kaggle
  session and runs the pipeline, plus Kaggle-side requirements.

## Workflow

All code lives here. Kaggle notebooks are thin runners: they `git clone` this
repo at session start and run entry points, so training always uses the
latest pushed code. Outputs (LoRA weights, sample sheets) are saved as
Kaggle outputs, never committed back.

## License

MIT.
