"""Synthetic poster-art training data generator (curriculum stage data).

Renders flat-illustration poster art with PIL: named palettes, 5 layouts,
simple focal subjects, and deliberately CLEAR empty zones. Every image is
lettering-free so a LoRA never learns fake text.

Captions follow a fixed pattern so attributes stay controllable by name:
  "<style> poster illustration, flat color, <focal subject and scene>,
   <composition note>, <palette>, <empty-zone note>"

Curriculum:
  stage1 = clean: flat background, few shapes, one focal subject, big clear zones
  stage2 = busier: banded/gradient backgrounds, texture dots, more shapes,
           still a clear top zone

Outputs: <out>/images/*.png and <out>/metadata.jsonl ({"file_name", "text"}).
"""
import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

# Named palettes: (name, [bg, primary, secondary, accent], harmony note)
PALETTES = [
    ("warm orange and teal", [(244, 162, 97), (42, 111, 110), (233, 196, 106), (231, 111, 81)], "complementary"),
    ("deep green and cream", [(34, 74, 63), (243, 238, 220), (116, 141, 108), (224, 122, 95)], "complementary"),
    ("terracotta and sand", [(196, 108, 75), (238, 214, 183), (154, 84, 60), (246, 235, 214)], "analogous"),
    ("indigo and coral", [(57, 72, 124), (244, 132, 107), (240, 200, 141), (252, 250, 240)], "complementary"),
    ("marigold and plum", [(240, 166, 66), (94, 54, 92), (247, 220, 160), (192, 108, 132)], "split-complementary"),
    ("mint and charcoal", [(172, 216, 190), (45, 52, 54), (240, 244, 235), (255, 159, 91)], "monochromatic-accent"),
    ("rose and forest", [(219, 127, 142), (35, 61, 47), (244, 230, 218), (146, 180, 167)], "complementary"),
    ("mustard and navy", [(228, 176, 62), (31, 46, 84), (242, 239, 227), (210, 105, 73)], "complementary"),
    ("sage and blush", [(165, 190, 162), (228, 182, 178), (250, 246, 238), (96, 116, 98)], "analogous"),
    ("cocoa and mint", [(104, 72, 56), (168, 218, 200), (238, 226, 205), (214, 148, 90)], "complementary"),
]

LAYOUTS = ["centered hero", "diagonal split", "top banner", "framed", "corner accent"]

STYLE_WORDS = [
    "vintage advertising poster illustration, flat color",
    "minimalist modern poster, flat geometric shapes",
    "retro Indian bazaar poster illustration, flat color",
    "clean editorial poster illustration, flat shapes",
]

# Focal subjects drawn from primitives; (name, draw fn key)
SUBJECTS = ["chai cup", "sun", "leaf", "jar", "bottle", "mithai stack", "lotus", "star"]


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_subject(draw, kind, cx, cy, r, col, col2):
    if kind == "chai cup":
        draw.rounded_rectangle([cx - r, cy - r // 2, cx + r, cy + r], radius=r // 5, fill=col)
        draw.arc([cx + r - r // 3, cy - r // 6, cx + r + r // 2, cy + r // 2], -90, 90, fill=col, width=max(4, r // 8))
        for i in (-1, 0, 1):  # steam
            draw.arc([cx + i * r // 2 - r // 6, cy - r, cx + i * r // 2 + r // 6, cy - r // 3], 200, 340, fill=col2, width=max(3, r // 12))
    elif kind == "sun":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        for k in range(12):
            a = k * math.pi / 6
            x1, y1 = cx + int(1.25 * r * math.cos(a)), cy + int(1.25 * r * math.sin(a))
            x2, y2 = cx + int(1.6 * r * math.cos(a)), cy + int(1.6 * r * math.sin(a))
            draw.line([x1, y1, x2, y2], fill=col, width=max(3, r // 10))
    elif kind == "leaf":
        draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=col)
        draw.line([cx - r, cy, cx + r, cy], fill=col2, width=max(3, r // 12))
    elif kind == "jar":
        draw.rounded_rectangle([cx - 3 * r // 4, cy - r // 3, cx + 3 * r // 4, cy + r], radius=r // 6, fill=col)
        draw.rectangle([cx - r // 2, cy - 2 * r // 3, cx + r // 2, cy - r // 3], fill=col2)
    elif kind == "bottle":
        draw.rounded_rectangle([cx - r // 2, cy - r // 4, cx + r // 2, cy + r], radius=r // 6, fill=col)
        draw.rectangle([cx - r // 6, cy - 3 * r // 4, cx + r // 6, cy - r // 4], fill=col)
        draw.rectangle([cx - r // 4, cy - r, cx + r // 4, cy - 3 * r // 4], fill=col2)
    elif kind == "mithai stack":
        for i in range(3):
            yy = cy + r // 2 - i * r // 2
            draw.ellipse([cx - r + i * r // 8, yy - r // 3, cx + r - i * r // 8, yy + r // 3], fill=col if i % 2 == 0 else col2)
    elif kind == "lotus":
        for k in range(6):
            a = k * math.pi / 3
            px, py = cx + int(0.7 * r * math.cos(a)), cy + int(0.7 * r * math.sin(a))
            draw.ellipse([px - r // 2, py - r // 2, px + r // 2, py + r // 2], fill=col)
        draw.ellipse([cx - r // 3, cy - r // 3, cx + r // 3, cy + r // 3], fill=col2)
    else:  # star
        pts = []
        for k in range(10):
            rr = r if k % 2 == 0 else r // 2
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((cx + int(rr * math.cos(a)), cy + int(rr * math.sin(a))))
        draw.polygon(pts, fill=col)


def scatter_shapes(draw, W, H, rng, cols, n, avoid_top_frac=0.28):
    for _ in range(n):
        c = cols[rng.randrange(len(cols))]
        x = rng.randrange(0, W)
        y = rng.randrange(int(H * avoid_top_frac), H)
        r = rng.randrange(W // 60, W // 14)
        t = rng.randrange(4)
        if t == 0:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
        elif t == 1:
            draw.rectangle([x - r, y - r // 2, x + r, y + r // 2], fill=c)
        elif t == 2:
            draw.polygon([(x, y - r), (x + r, y + r), (x - r, y + r)], fill=c)
        else:
            draw.arc([x - r, y - r, x + r, y + r], 0, 360, fill=c, width=max(3, r // 5))


def make_poster(rng, W, H, stage):
    pname, cols, _harmony = PALETTES[rng.randrange(len(PALETTES))]
    bg, c1, c2, c3 = cols
    layout = LAYOUTS[rng.randrange(len(LAYOUTS))]
    style = STYLE_WORDS[rng.randrange(len(STYLE_WORDS))]
    subject = SUBJECTS[rng.randrange(len(SUBJECTS))]
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    text_zone = "large clear sky area at the top"
    if stage == 2 and rng.random() < 0.6:  # busier backgrounds
        for i in range(H):
            d.line([(0, i), (W, i)], fill=_lerp(bg, c2, i / H * 0.35))

    if layout == "diagonal split":
        d.polygon([(0, H), (W, int(H * 0.45)), (W, H)], fill=c2)
        comp = "diagonal split composition, subject in the lower half"
    elif layout == "top banner":
        d.rectangle([0, 0, W, int(H * 0.22)], fill=_lerp(bg, c1, 0.25))
        comp = "centered composition with a plain top band"
    elif layout == "framed":
        m = int(W * 0.06)
        d.rectangle([m, m, W - m, H - m], outline=c3, width=max(4, W // 90))
        comp = "centered composition inside a thin frame border"
    elif layout == "corner accent":
        d.polygon([(0, H), (int(W * 0.45), H), (0, int(H * 0.55))], fill=c2)
        d.polygon([(W, H), (int(W * 0.55), H), (W, int(H * 0.6))], fill=c3)
        comp = "centered composition with corner accents at the bottom"
    else:
        comp = "centered composition, one focal point"

    n_shapes = rng.randrange(1, 4) if stage == 1 else rng.randrange(4, 9)
    scatter_shapes(d, W, H, rng, [c1, c2, c3], n_shapes)

    r = int(W * 0.16)
    cy = int(H * 0.52) if layout != "diagonal split" else int(H * 0.62)
    draw_subject(d, subject, W // 2, cy, r, c1, c3)

    mdot = int(H * 0.86)  # small detail-zone dots instead of text
    for i in range(3):
        x = W // 2 + (i - 1) * int(W * 0.05)
        d.ellipse([x - 6, mdot - 6, x + 6, mdot + 6], fill=c3)

    caption = (f"{style}, a {subject} as the focal subject, {comp}, "
               f"{pname} palette, {text_zone}")
    return img, caption


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--size", type=int, nargs=2, default=[1024, 1280])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    meta_path = out / "metadata.jsonl"
    with meta_path.open("a") as f:
        for i in range(args.n):
            img, caption = make_poster(rng, args.size[0], args.size[1], args.stage)
            name = f"{args.prefix}stage{args.stage}_{args.seed:03d}_{i:04d}.png"
            img.save(out / "images" / name)
            f.write(json.dumps({"file_name": f"images/{name}", "text": caption}) + "\n")
    print(f"wrote {args.n} stage-{args.stage} posters -> {out}")


if __name__ == "__main__":
    main()
