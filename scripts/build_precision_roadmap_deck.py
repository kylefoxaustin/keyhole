#!/usr/bin/env python3
"""
build_precision_roadmap_deck.py — append the two precision charts to a repo copy
of the roadmap deck.

Reads the ORIGINAL ~/Downloads/precision-roadmap-combined.pptx (never mutates it),
appends the full-bleed chart slides, and writes the 7-slide result to
data/output/precision-roadmap-combined.pptx.

  slide 4: migration timeline (the thesis)
  slide 5: today-anchor composition (the evidence)
  slide 6: decode bandwidth-bound ladder + MoE footprint decoupling (llama.cpp Tier-2)
  slide 7: INT4-vs-FP4 — a 4-bit memory format is not a compute format (vLLM breadth)

Run:  python3 scripts/build_precision_roadmap_deck.py
"""
import os
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.expanduser("~/Downloads/precision-roadmap-combined.pptx")
OUT = os.path.join(REPO, "data", "output", "precision-roadmap-combined.pptx")

SLIDES = [
    ("data/output/precision_migration.png",
     "Edge AI Use Cases on NPU Silicon", "4 · Precision migration timeline · Jun 2026"),
    ("data/output/precision_composition.png",
     "Edge AI Use Cases on NPU Silicon", "5 · Today's measured INT/FP composition · Jun 2026"),
    ("data/output/precision_5090_ladder.png",
     "Edge AI Use Cases on NPU Silicon", "6 · Decode is bandwidth-bound — measured ladder + MoE decoupling · Jun 2026"),
    ("data/output/precision_5090_breadth.png",
     "Edge AI Use Cases on NPU Silicon", "7 · INT4 vs FP4 — a 4-bit memory format is not a compute format · Jun 2026"),
]


def strip_placeholders(slide):
    """Remove empty layout placeholders so they don't render in the show."""
    for ph in list(slide.placeholders):
        ph._element.getparent().remove(ph._element)


def add_image_slide(prs, img_path, footer_left, footer_right):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    strip_placeholders(slide)

    sw, sh = prs.slide_width, prs.slide_height
    from PIL import Image
    iw, ih = Image.open(os.path.join(REPO, img_path)).size
    ar = iw / ih
    margin = Emu(int(0.18 * 914400))            # 0.18" side margin
    avail_w = sw - 2 * margin
    avail_h = sh - Emu(int(0.55 * 914400))       # leave a footer strip
    # Fit within avail box, preserving aspect ratio.
    if avail_w / avail_h > ar:                   # height-bound
        h = avail_h
        w = Emu(int(int(h) * ar))
    else:                                        # width-bound
        w = avail_w
        h = Emu(int(int(w) / ar))
    left = Emu(int((sw - int(w)) / 2))
    top = Emu(int((sh - Emu(int(0.45 * 914400)) - int(h)) / 2))
    slide.shapes.add_picture(os.path.join(REPO, img_path), left, top, width=w, height=h)

    # Footer band (match the existing deck's footer convention). Left + right footers each
    # own HALF the width so they can't collide in the centre: left is left-aligned from the
    # left margin, right is right-aligned to the right margin, leaving a gap between them.
    from pptx.enum.text import PP_ALIGN
    half = sw // 2 - margin
    tb = slide.shapes.add_textbox(margin, sh - Emu(int(0.40 * 914400)),
                                  half, Emu(int(0.32 * 914400)))
    tf = tb.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = footer_left
    r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    # right-aligned page label as a second textbox, owning the right half
    tb2 = slide.shapes.add_textbox(sw // 2, sh - Emu(int(0.40 * 914400)),
                                   half, Emu(int(0.32 * 914400)))
    p2 = tb2.text_frame.paragraphs[0]; p2.alignment = PP_ALIGN.RIGHT
    r2 = p2.add_run(); r2.text = footer_right
    r2.font.size = Pt(9); r2.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    return slide


def main():
    prs = Presentation(SRC)
    n_before = len(prs.slides._sldIdLst)
    for img, fl, fr in SLIDES:
        add_image_slide(prs, img, fl, fr)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT)
    print(f"{SRC} ({n_before} slides) -> {OUT} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
