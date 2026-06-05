# Deck Style Guide — how to build a presentation in the keyhole style

Written for the qemu-imx93 / qemu-imx95 sessions to build a "What is qemu-imx9X"
internal engineering deck in the same style as the precision-roadmap / exec decks.
Everything here is reusable; the branding toolkit already exists in this repo.

## TL;DR

Decks are **built in code** (python-pptx), not hand-drawn. Three layers:

1. **Branding toolkit** — `scripts/build_deck.py`: colours (`C`), slide factory
   (`new_slide`), and content helpers (`add_title_subtitle`, `add_text_box`,
   `add_bullet_box`, `add_styled_table`). 16:9 dark theme. **Reuse this — don't
   reinvent it.**
2. **Deck builder** — a script that calls those helpers to emit one slide per
   function, in narrative order, and `prs.save()`s a `.pptx`. Canonical examples:
   - `scripts/build_vla_exec_deck.py` — branded content slides (title, TL;DR,
     tables, bullet lists, section dividers, takeaways).
   - `scripts/build_combined_exec_deck.py` — composes content slides **+**
     full-bleed chart slides + part-dividers into one deck.
   - `scripts/build_precision_roadmap_deck.py` — appends matplotlib chart PNGs
     as full-bleed image slides onto a base deck, with footers.
3. **Charts** — matplotlib figures saved to PNG, dropped onto slides full-bleed.
   Any `scripts/plot_*.py` (e.g. `plot_precision_breadth_v2.py`) is a template:
   dark-friendly colours, one clear message per chart, a caption.

Read those 4 files first — they ARE the spec. This doc is the map.

## Reusing the branding toolkit from another repo

All repos live on one machine, so the fastest path is to point at keyhole's
`build_deck.py` directly (no copy):

```python
import sys
sys.path.insert(0, "/home/kyle/Documents/GitHub/keyhole/scripts")
from build_deck import (
    C, set_deck_size, new_slide, add_text_box, add_title_subtitle,
    add_bullet_box, add_styled_table, CONTENT_LEFT, CONTENT_W,
    SLIDE_W_IN, SLIDE_H_IN,
)
from pptx import Presentation
from pptx.util import Inches
```

(If you'd rather vendor it for portability, copy `build_deck.py` into your repo —
it's a self-contained python-pptx module, no keyhole-specific deps.)

Install once in your venv: `pip install python-pptx matplotlib`.

## Branding cheat-sheet (`build_deck.py`)

- **Canvas**: 16:9, `SLIDE_W_IN=13.333 × SLIDE_H_IN=7.5` in. Call
  `set_deck_size(prs)` once. Content lives in `CONTENT_LEFT=0.5 .. CONTENT_LEFT+CONTENT_W` (12.3 in wide).
- **Colours** (`C.`): `BG_DARK` (title/divider bg), `BG_SLIDE` (content bg),
  `ACCENT_BLUE` (titles), `ACCENT_GREEN` (emphasis/good), `ACCENT_ORANGE`/`ACCENT_RED`
  (warnings), `ACCENT_PURPLE`/`AMBER`/`INDIGO`, `TEXT_WHITE`/`TEXT_BRIGHT`/`TEXT_DIM`.
- **`new_slide(prs, bg_color=None, accent_stripe=True)`** → a blank branded slide
  (dark bg + a thin blue top stripe). Pass `bg_color=C.BG_DARK` for title/divider slides.
- **`add_title_subtitle(slide, title, subtitle=None)`** → blue title + dim subtitle, standard position.
- **`add_text_box(slide, left, top, w, h, text, font_size=18, color=C.TEXT_WHITE, bold=False)`** — raw text (Inches).
- **`add_bullet_box(slide, left_in, top_in, w_in, h_in, items)`** — bullet list. Each
  item is a plain `str`, OR a `(text, color, bold)` tuple for an emphasis line.
- **`add_styled_table(slide, headers, rows, ...)`** — a branded table (alternating rows, header band).

## The deck structure recipe (what made the precision deck read well)

1. **Title slide** — `new_slide(bg=BG_DARK)` + big white title + blue one-line subject + dim sub-points.
2. **TL;DR / thesis** — ONE slide stating the whole argument up front (bullets,
   emphasis lines on the key claims). The reader should get it from this slide alone.
3. **Section dividers** — `new_slide(bg=BG_DARK)` with a small "Part N" kicker + big title + dim subtitle. Use to break parts.
4. **Content slides** — ONE idea per slide. A table, or a bullet box, or a chart.
   Title states the *takeaway*, not the topic ("INT4 wins only decode; FP4 wins both",
   not "Precision results").
5. **Chart slides** — full-bleed PNG + a title + a one-line caption (see next section).
6. **Takeaways** — close with the 3-4 things you want them to remember.

Discipline that matters: **one thesis per slide; titles assert, not label; emphasis
lines (green) carry the load-bearing claims; keep it skimmable.**

## Making a chart slide (the precision-roadmap pattern)

1. Write a `plot_*.py` that builds a matplotlib figure and `fig.savefig("out.png", dpi=130)`.
   Keep it to ONE message; annotate the key number; add a short caption line. Copy the
   look from `scripts/plot_precision_breadth_v2.py` (lollipop) or `plot_fp4_lifecycle.py` (bars).
2. Drop it on a slide full-bleed, preserving aspect ratio. Pattern (from
   `build_combined_exec_deck.py:slide_chart`):

```python
from PIL import Image
def slide_chart(prs, png, title, caption):
    s = new_slide(prs)
    add_text_box(s, Inches(CONTENT_LEFT), Inches(0.45), Inches(CONTENT_W), Inches(0.55),
                 title, font_size=20, color=C.ACCENT_BLUE, bold=True)
    box_l, box_t, box_w, box_h = 0.6, 1.15, SLIDE_W_IN - 1.2, 5.5      # body box
    iw, ih = Image.open(png).size; ar = iw / ih
    w, h = (box_h*ar, box_h) if box_w/box_h > ar else (box_w, box_w/ar)  # fit, keep AR
    s.shapes.add_picture(png, Inches(box_l + (box_w-w)/2), Inches(box_t), Inches(w), Inches(h))
    add_text_box(s, Inches(CONTENT_LEFT), Inches(SLIDE_H_IN-0.55), Inches(CONTENT_W),
                 Inches(0.4), caption, font_size=12, color=C.TEXT_DIM)
    return s
```

For diagrams/screenshots (e.g. a QEMU `screendump`, a block diagram, a boot log), the
SAME `slide_chart` works — any PNG, full-bleed.

## Render + QC (don't skip this — it caught real overlaps for us)

```sh
libreoffice --headless --convert-to pdf --outdir /tmp/check yourdeck.pptx
pdftoppm -r 130 -png /tmp/check/yourdeck.pdf /tmp/check/s
# then open /tmp/check/s-*.png and EYEBALL each slide for:
#   - text boxes overlapping each other or running off the slide
#   - footers colliding (left vs right) — own each its half
#   - chart titles/captions overlapping the image
```
Reading the rendered PNGs back is the only reliable overlap check; python-pptx won't warn you.

## Distribution (how Kyle wants decks delivered)

- Push the `.pptx` (+ a regenerated `.pdf`) to the `my-stuff` repo:
  `cp deck.pptx ~/Documents/GitHub/my-stuff/ && libreoffice --headless --convert-to pdf --outdir ~/Documents/GitHub/my-stuff deck.pptx`, then commit+push.
- And rclone to gdrive for phone review: `rclone copy deck.pptx gdrive:skippy_files/keyhole/` (use your own subfolder if you prefer).
- No `Co-Authored-By` lines in commits (Kyle's standing rule).

## Suggested split for a "What is qemu-imx9X" internal eng deck

You two have the content already (the milestone arc + architecture is all in your bus
history). Coordinate so it's ONE coherent deck, not two:
- Agree a shared title + TL;DR slide ("what these emulators are + why they matter").
- One **Part: i.MX 95** section (95emulator): heterogeneous boot (real M33 SM + SCMI),
  the from-scratch models (FlexCAN, ENETC/NETC, ITS+ECAM), the soak/qtest/e2e validation
  discipline, the 2 external robotics adopters, the 3 upstream patches.
- One **Part: i.MX 93** section (93emulator): greenfield→userspace→networking→**pixels**,
  the LPI2C/edma3/display chain, the SDCLK_AUTO_GATE cross-base reuse (portability witness).
- A shared **methodology** slide (the transferable patterns: register-class triage,
  sub-word MMIO, icount-first, validation bar) — that's the engineering takeaway.
- Close with the upstream status + what's next.
- Use screendumps (the Tux logos!) + boot-log snippets as full-bleed chart slides — they land.

Keep it SIMPLE for an internal eng audience: ~10-14 slides, one idea each, assert the
takeaway in every title. Lead with "what it is + why" before any architecture detail.
