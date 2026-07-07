"""Desenha as deteccoes (componentes + fluxos) sobre a imagem original, para
verificacao visual do que os modelos treinados realmente enxergam -- util
tanto para debug quanto para o video de apresentacao do hackathon.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from ..schema import Component, Flow

BOUNDARY_COLOR = "#1f77b4"
COMPONENT_COLOR = "#d62728"
TAIL_COLOR = "#2ca02c"
TIP_COLOR = "#ff7f0e"
LINE_COLOR = "#9467bd"


def draw_annotations(
    image_path: str | Path, components: list[Component], flows: list[Flow], output_path: str | Path
) -> None:
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size

    for f in flows:
        if f.tail is None or f.tip is None:
            continue
        tx, ty = f.tail[0] * w, f.tail[1] * h
        px, py = f.tip[0] * w, f.tip[1] * h
        draw.line([tx, ty, px, py], fill=LINE_COLOR, width=2)
        draw.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=TAIL_COLOR)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=TIP_COLOR)

    for c in components:
        b = c.bbox
        x0, y0, x1, y1 = b.x0 * w, b.y0 * h, b.x1 * w, b.y1 * h
        color = BOUNDARY_COLOR if c.is_boundary else COMPONENT_COLOR
        width = 2 if c.is_boundary else 3
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        label = f"{c.cls_name} {c.confidence:.2f}"
        text_y = y0 - 12 if y0 - 12 > 0 else y1 + 2
        draw.rectangle([x0, text_y, x0 + 7 * len(label), text_y + 11], fill=color)
        draw.text((x0 + 1, text_y), label, fill="white")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(output_path)
