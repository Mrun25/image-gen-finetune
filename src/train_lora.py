"""Qwen-Image-2.1-Turbo LoRA trainer (free-T4 recipe).

Recipe proven in a private Kaggle smoke test (peak 4.8GB on a T4): 4bit nf4
transformer+text_encoder, PEFT LoRA r16 on attention projections,
flow-matching loss on the Turbo checkpoint's own saved sigmas, gradient
checkpointing, batch 1.

On top of the smoke recipe:
- data from synth_posters.py (per-image captions via metadata.jsonl)
- two-phase curriculum: phase 1 trains stage-1 (clean) data, phase 2 trains
  stage-1+stage-2 mixed (see docs/modern-image-models.md)
- base-vs-LoRA comparison sheet on held-out prompts

Run ON Kaggle (GPU T4 x2, internet on) via kaggle/runner.ipynb, which installs
deps first. Not meant for local CPU.
"""
import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"


def load_dataset(data_dirs):
    """Return [(image_path, caption)] from metadata.jsonl in each dir."""
    items = []
    for d in data_dirs:
        d = Path(d)
        for line in (d / "metadata.jsonl").read_text().splitlines():
            r = json.loads(line)
            items.append((d / r["file_name"], r["text"]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True, help="dir with stage-1 metadata.jsonl")
    ap.add_argument("--stage2", required=True, help="dir with stage-2 metadata.jsonl")
    ap.add_argument("--steps1", type=int, default=250)
    ap.add_argument("--steps2", type=int, default=250)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import bitsandbytes as bnb
    from diffusers import QwenImage21Pipeline, PipelineQuantizationConfig
    from peft import LoraConfig, get_peft_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H, W = args.height, args.width
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    qc = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                      "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=["transformer", "text_encoder"])
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, quantization_config=qc,
                                               torch_dtype=torch.bfloat16)
    F = pipe.vae_scale_factor
    LH, LW = H // F, W // F
    L = LH * LW
    sigmas = pipe.config.sample_sigmas
    if sigmas is None:
        sigmas = list(np.linspace(1.0, 1 / 8, 8))
    SIGMAS = [float(s) for s in sigmas]
    print("vae_scale", F, "sigmas", SIGMAS, flush=True)

    # ---- precompute caption embeddings ----
    s1 = load_dataset([args.stage1])
    s2 = load_dataset([args.stage2])
    print(f"data: stage1={len(s1)} stage2={len(s2)}", flush=True)
    pipe.text_encoder.to("cuda")
    EMB = []
    for i, (_, cap) in enumerate(s1 + s2):
        pe, pm, _ = pipe.encode_prompt(cap, device="cuda")
        EMB.append((pe.detach().cpu(), pm.detach().cpu() if pm is not None else None))
        if (i + 1) % 100 == 0:
            print(f"encoded {i+1}/{len(s1)+len(s2)}", flush=True)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # ---- precompute VAE latents (Qwen-Image-2.1 VAE reads 4 channels) ----
    pipe.vae.to("cuda")
    LATS = []
    with torch.no_grad():
        for p, _ in s1 + s2:
            im = Image.open(p).convert("RGB").resize((W, H)).convert("RGBA")
            a = np.asarray(im).astype(np.float32) / 127.5 - 1.0
            t = torch.from_numpy(a).permute(2, 0, 1)[None, :, None].to("cuda", torch.bfloat16)
            z = pipe._encode_vae_image(t, None)
            LATS.append(z.view(1, z.shape[1], LH * LW).transpose(1, 2).float().cpu())
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    print("precompute done:", len(LATS), "latents", flush=True)

    # ---- LoRA ----
    tr = pipe.transformer
    tr.enable_gradient_checkpointing()
    targets = sorted({n for n, m in tr.named_modules()
                      if isinstance(m, (torch.nn.Linear, bnb.nn.Linear4bit))
                      and n.split(".")[-1] in ("to_q", "to_k", "to_v", "to_out")})
    cfg = LoraConfig(r=16, lora_alpha=16, target_modules=targets, lora_dropout=0.0)
    tr = get_peft_model(tr, cfg)
    tr.print_trainable_parameters()
    tr = tr.to("cuda")
    opt = torch.optim.AdamW([p for p in tr.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)

    IMGS = [[(1, LH, LW)]]
    hist = []

    def train_phase(idxs, steps, tag):
        for step in range(1, steps + 1):
            i = random.choice(idxs)
            lat = LATS[i].to("cuda", torch.bfloat16)
            pe, pm = EMB[i]
            pe = pe.to("cuda", torch.bfloat16)
            pm = pm.to("cuda") if pm is not None else None
            IM = torch.cat([torch.zeros(1, pe.shape[1], dtype=torch.bool),
                            torch.ones(1, L // 4, dtype=torch.bool)], dim=1).to("cuda")
            noise = torch.randn_like(lat)
            s = random.choice(SIGMAS)
            xt = (1.0 - s) * lat + s * noise
            tgt = noise - lat
            ts = torch.tensor([s], device="cuda", dtype=torch.bfloat16)
            pred = tr(hidden_states=xt, timestep=ts, encoder_hidden_states=pe,
                      encoder_hidden_states_mask=pm, img_shapes=IMGS, img_mask=IM,
                      return_dict=False)[0]
            pred = pred[:, -lat.shape[1]:]
            loss = Fn.mse_loss(pred.float(), tgt.float())
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step == 1 or step % 25 == 0:
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f"{tag} step {step}/{steps} loss {loss.item():.4f} peak {peak:.1f}GB", flush=True)
                hist.append((tag, step, round(loss.item(), 4)))

    torch.cuda.reset_peak_memory_stats()
    train_phase(list(range(len(s1))), args.steps1, "p1")
    train_phase(list(range(len(s1) + len(s2))), args.steps2, "p2")

    (out / "lora").mkdir(exist_ok=True)
    tr.save_pretrained(str(out / "lora"))
    json.dump({"hist": hist, "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)},
              open(out / "train_log.json", "w"))
    print("TRAIN OK peak GB", round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), flush=True)

    # ---- eval: base vs LoRA on held-out prompts ----
    from diffusers import FlowMatchEulerDiscreteScheduler
    del opt
    gc.collect()
    torch.cuda.empty_cache()

    ROWS = [
        ("in_chai", "vintage advertising poster illustration, flat color, a chai cup as the focal subject, centered composition, one focal point, warm orange and teal palette, large clear sky area at the top"),
        ("in_mithai", "retro Indian bazaar poster illustration, flat color, a mithai stack as the focal subject, diagonal split composition, subject in the lower half, marigold and plum palette, large clear sky area at the top"),
        ("in_bottle", "minimalist modern poster, flat geometric shapes, a bottle as the focal subject, centered composition inside a thin frame border, deep green and cream palette, large clear sky area at the top"),
        ("hold_jar", "vintage advertising poster illustration, flat color, a jar as the focal subject, centered composition, one focal point, indigo and coral palette, large clear sky area at the top"),
        ("hold_leaf", "clean editorial poster illustration, flat shapes, a leaf as the focal subject, centered composition with corner accents at the bottom, rose and forest palette, large clear sky area at the top"),
        ("hold_star", "minimalist modern poster, flat geometric shapes, a star as the focal subject, diagonal split composition, subject in the lower half, mustard and navy palette, large clear sky area at the top"),
    ]
    pipe.text_encoder.to("cuda")
    EMBEV = {}
    for tag, p in ROWS:
        pe, pm, _ = pipe.encode_prompt(p, device="cuda")
        EMBEV[tag] = (pe.detach().to("cuda", torch.bfloat16),
                      pm.detach().to("cuda") if pm is not None else None)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    sch = pipe.scheduler
    m_shift = (1.15 - 0.5) / (4096 - 256)

    def mu_for(seq):
        return seq * m_shift + (0.5 - m_shift * 256)

    def sgen(tag, seed, EH=640, EW=512):
        eh, ew = EH // F, EW // F
        Le = eh * ew
        pe, pm = EMBEV[tag]
        im_mask = torch.cat([torch.zeros(1, pe.shape[1], dtype=torch.bool),
                             torch.ones(1, Le // 4, dtype=torch.bool)], dim=1).to("cuda")
        ish = [[(1, eh, ew)]]
        g = torch.Generator("cpu").manual_seed(seed)
        C = tr.config.in_channels
        lat = torch.randn((1, C, 1, eh, ew), generator=g).to("cuda", torch.bfloat16).view(1, C, Le).transpose(1, 2)
        sch.set_timesteps(sigmas=SIGMAS, device="cuda", mu=mu_for(Le))
        for t in sch.timesteps:
            ts = t.reshape(1).to(torch.bfloat16) / 1000.0
            with torch.no_grad():
                pred = tr(hidden_states=lat, timestep=ts, encoder_hidden_states=pe,
                          encoder_hidden_states_mask=pm, img_shapes=ish, img_mask=im_mask,
                          return_dict=False)[0]
            lat = sch.step(pred[:, -Le:], t, lat, return_dict=False)[0]
        return lat.transpose(1, 2).reshape(1, C, 1, eh, ew)

    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    def decode(z):
        tr.to("cpu")
        torch.cuda.empty_cache()
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        pipe.vae.to("cuda")
        with torch.no_grad():
            img = pipe.vae.decode((z * std + mean).to(pipe.vae.dtype)).sample
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        tr.to("cuda")
        a = (img[0, :, 0].float().clamp(-1, 1).cpu().numpy().transpose(1, 2, 0) + 1) / 2
        return Image.fromarray((a * 255).astype("uint8"))

    EH, EW = 640, 512
    cols = []
    with tr.disable_adapter():
        for tag, _ in ROWS:
            cols.append(decode(sgen(tag, 77, EH, EW)))
            print(" base done", tag, flush=True)
    for tag, _ in ROWS:
        cols.append(decode(sgen(tag, 77, EH, EW)))
        print(" lora done", tag, flush=True)

    n = len(ROWS)
    sheet = Image.new("RGB", (EW * n, EH * 2), "white")
    for i, im in enumerate(cols[:n]):
        sheet.paste(im, (i * EW, 0))
    for i, im in enumerate(cols[n:]):
        sheet.paste(im, (i * EW, EH))
    sheet.save(out / "compare_base_vs_lora.png")
    print("EVAL SHEET saved (top=base, bottom=lora)", flush=True)


if __name__ == "__main__":
    main()
