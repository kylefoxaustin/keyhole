"""Build the standalone VLA exec deck: data/output/vla-sizer-exec.pptx.

The parked exec artifact for the VLA bake-off — distinct from the vision-story
keyhole_results.pptx. Tells the VLA edge-sizing story: five published VLAs measured
on an RTX 5090, projected to NXP NPU tiers via the keyhole-sizer engine, organized
around the THREE action-generation topologies (single-loop AR / dual-loop
flow-matching / OFT parallel-chunk) and the three multi-camera cost shapes.

5090 numbers are this repo's measurements (data/output/bakeoff/vla_summary_*.json).
Edge projections are keyhole-sizer's project_vla output (5090-anchored, 3-state
source: green measured / blue calibrated / orange cross_class).

Reuses build_deck.py's python-pptx helpers + branding. Usage:
    python scripts/build_vla_exec_deck.py [--output data/output/vla-sizer-exec.pptx]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_deck import (  # noqa: E402  reuse branding + helpers
    C, set_deck_size, new_slide, add_title_subtitle, add_text_box,
    add_styled_table, add_bullet_box, CONTENT_LEFT, CONTENT_W,
)

OUT_DEFAULT = REPO / "data" / "output" / "vla-sizer-exec.pptx"

# add_bullet_box item formats: plain str = normal bullet; (text, color, bold) = emphasis.
EMPH = C.ACCENT_GREEN   # emphasis-line color


def slide_title(prs):
    s = new_slide(prs, bg_color=C.BG_DARK)
    add_text_box(s, Inches(0.7), Inches(2.4), Inches(12), Inches(1.1),
                 "VLA Edge Sizing — Bake-off Results", font_size=40, bold=True,
                 color=C.TEXT_WHITE)
    add_text_box(s, Inches(0.7), Inches(3.5), Inches(12), Inches(0.6),
                 "Five Vision-Language-Action models, measured on RTX 5090, "
                 "projected to NXP NPU tiers", font_size=18, color=C.ACCENT_BLUE)
    add_text_box(s, Inches(0.7), Inches(4.3), Inches(12), Inches(1.4),
                 "Three action-generation topologies → three distinct edge outcomes.\n"
                 "Topology, not parameter count, decides edge viability.",
                 font_size=15, color=C.TEXT_DIM)
    add_text_box(s, Inches(0.7), Inches(6.7), Inches(12), Inches(0.4),
                 "RTX 5090 measured (Nsight-grade) · NPU projections via keyhole-sizer "
                 "(5090-anchored) · all numbers reproducible", font_size=11, color=C.TEXT_DIM)
    return s


def slide_tldr(prs):
    s = new_slide(prs)
    add_title_subtitle(s, "The question, and the answer",
                       "Can a robot run a VLA control loop on an edge NPU? It depends "
                       "entirely on the action-generation topology.")
    add_bullet_box(s, CONTENT_LEFT, 1.6, CONTENT_W, 4.8, [
        "Single-loop autoregressive (NORA-3B, OpenVLA-7B): every action decoded "
        "token-by-token through the full LLM → hard BANDWIDTH-WALL on edge. "
        "12.6 Hz on 5090 → ~1.8 Hz on NPU Mid. More NPU compute buys ~nothing.",
        "Dual-loop flow-matching (NORA-1.5, π0.5): VLM runs ONCE per action chunk; "
        "a small expert emits the whole chunk → chunk size is the AMORTIZATION KNOB. "
        "π0.5's 50-action chunk → 367 Hz on 5090, 148.9 Hz on NPU High. But the "
        "flow-matching head needs floating point → won't run on INT8-only tiers.",
        "OFT parallel-chunk (BitVLA, ternary): one parallel forward emits the chunk — "
        "no AR loop, no bandwidth-wall. The only model that runs on INT8-only Mid — "
        "and it runs FAST (53 Hz Mid, 118 Hz High); 6 GB footprint (ternary memory win).",
        ("Bottom line: the same edge silicon that strands a 3B single-loop VLA at "
         "1.8 Hz runs a 3B dual-loop VLA at 100+ Hz. Architecture is the lever.", EMPH, True),
    ], font_size=15)
    return s


def slide_topologies(prs):
    s = new_slide(prs)
    add_title_subtitle(s, "Three topologies, three edge behaviors",
                       "The structural backbone of the VLA story")
    headers = ["Topology", "Models", "Per-action mechanism", "Edge bottleneck"]
    rows = [
        ["Single-loop\nautoregressive", "NORA-3B\nOpenVLA-7B",
         "1 VLM prefill, then K action tokens\ndecoded one-by-one through the LLM",
         "BANDWIDTH-WALL — each token\nre-streams all weights (decode\nutil 0.3–0.5%)"],
        ["Dual-loop\nflow-matching", "NORA-1.5\nπ0.5",
         "VLM once → KV cache; small expert\nruns N denoise steps → H-action chunk",
         "VLM amortized over chunk; FP-gated\noff INT8-only tiers; fast loop has\noptimization headroom"],
        ["OFT\nparallel-chunk", "BitVLA\n(ternary)",
         "ONE parallel forward → H actions via\na regression head (no AR, no denoise)",
         "Prefill-shaped (compute-bound) —\nNO AR wall; runs on INT8-only Mid"],
    ]
    add_styled_table(s, Inches(0.4), Inches(1.7), Inches(12.5), Inches(4.2),
                     headers, rows,
                     col_widths=[Inches(2.2), Inches(1.9), Inches(4.6), Inches(3.8)],
                     font_size=12, header_font_size=13)
    add_text_box(s, Inches(0.5), Inches(6.5), Inches(12), Inches(0.5),
                 "Same ~3B scale spans all three — the topology, not the size, sets the "
                 "edge ceiling.", font_size=12, color=C.ACCENT_GREEN)
    return s


def slide_5090_table(prs):
    s = new_slide(prs)
    add_title_subtitle(s, "Five models measured on RTX 5090",
                       "Ground-truth latency anchors (bf16, n=20, p50) — the calibration base")
    headers = ["Model", "Topology", "Params", "Action latency", "Rate", "Peak VRAM"]
    rows = [
        ["NORA-3B", "single-loop AR", "3.6 B", "79 ms e2e", "12.6 Hz", "7.1 GB"],
        ["OpenVLA-7B", "single-loop AR", "7.5 B", "126 ms e2e", "7.9 Hz", "14.4 GB"],
        ["NORA-1.5", "dual-loop", "4.0 B", "183 ms / 5-act chunk", "27 Hz", "7.6 GB"],
        ["π0.5", "dual-loop", "4.1 B", "136 ms / 50-act chunk", "367 Hz", "20.9 GB*"],
        ["BitVLA", "OFT parallel-chunk", "3.0 B", "123 ms / 8-act chunk", "65 Hz", "6.07 GB"],
    ]
    add_styled_table(s, Inches(0.6), Inches(1.7), Inches(12.1), Inches(3.4),
                     headers, rows,
                     col_widths=[Inches(1.9), Inches(2.6), Inches(1.4), Inches(3.0),
                                 Inches(1.5), Inches(1.7)],
                     font_size=13, header_font_size=13, highlight_rows=[4])
    add_text_box(s, Inches(0.6), Inches(5.4), Inches(12), Inches(1.2),
                 "Rate = control Hz (dual-loop/OFT: amortized over the action chunk). "
                 "π0.5 at 367 Hz is the amortization extreme — one VLM forward per 50 "
                 "actions.\n* π0.5 VRAM is float32 master weights (lerobot bf16-AMP path); "
                 "true bf16-weight deploy roughly halves it.", font_size=12, color=C.TEXT_DIM)
    return s


def slide_edge_projection(prs):
    s = new_slide(prs)
    add_title_subtitle(s, "Projected to NXP NPU tiers",
                       "keyhole-sizer engine, 5090-anchored · green measured / blue calibrated / orange cross-class")
    headers = ["Model", "5090", "NPU Mid (INT8-only)", "NPU High", "Edge story"]
    rows = [
        ["NORA-3B", "12.6 Hz", "1.76 Hz (blue)", "1.84 Hz (blue)", "BW-walled; High ≈ Mid"],
        ["OpenVLA-7B", "7.9 Hz", "0.94 Hz (blue)", "0.97 Hz (blue)", "BW-walled (7B, worse)"],
        ["NORA-1.5", "27 Hz", "✗ won't run", "21.0 Hz (blue)", "FP-gated off INT8 Mid"],
        ["π0.5", "367 Hz", "✗ won't run", "148.9 Hz (blue)", "FP-gated; amortizes huge"],
        ["BitVLA", "65 Hz", "53.5 Hz (blue)", "118 Hz (blue)", "int-only — runs FAST on Mid"],
    ]
    add_styled_table(s, Inches(0.5), Inches(1.7), Inches(12.3), Inches(3.4),
                     headers, rows,
                     col_widths=[Inches(1.8), Inches(1.3), Inches(2.8), Inches(2.2), Inches(4.2)],
                     font_size=12, header_font_size=12, highlight_rows=[4])
    add_text_box(s, Inches(0.5), Inches(5.4), Inches(12.3), Inches(1.4),
                 "Two edge gates decide deployability: (1) the BANDWIDTH-WALL strands "
                 "single-loop AR at single-digit Hz regardless of compute; (2) the FP "
                 "GATE blocks the dual-loop flow-matching head from INT8-only tiers. "
                 "Dual-loop on an FP-capable High tier, or OFT/ternary on an INT8 tier, "
                 "are the viable edge paths.", font_size=12, color=C.TEXT_BRIGHT)
    return s


def slide_bw_wall(prs):
    s = new_slide(prs)
    add_title_subtitle(s, "Single-loop: the bandwidth wall",
                       "Why more NPU TOPS doesn't help an autoregressive VLA")
    add_bullet_box(s, CONTENT_LEFT, 1.6, CONTENT_W, 4.6, [
        "Autoregressive decode does ~0.3–0.5% of the 5090's peak compute — it is pure "
        "weight-streaming. Every action token re-reads the entire weight set from memory.",
        "On NPU Mid, NORA-3B falls 12.6 → ~1.8 Hz. NPU High (2× the compute) lands at "
        "~the same ~1.8 Hz — because the loop is bandwidth-bound, not compute-bound.",
        "OpenVLA-7B is worse (~0.95 Hz): more weights to stream per token.",
        ("Implication for NXP: a single-loop VLA cannot be rescued by a higher-TOPS NPU. "
         "It needs an architecture change (dual-loop / OFT) or far higher memory bandwidth.",
         C.ACCENT_ORANGE, True),
    ], font_size=15)
    add_text_box(s, CONTENT_LEFT, 6.3, CONTENT_W, 0.7,
                 "“Needs more bandwidth or a smaller/different model — not more compute.”",
                 font_size=15, color=C.ACCENT_ORANGE, bold=True)
    return s


def slide_amortization(prs):
    s = new_slide(prs)
    add_title_subtitle(s, "Dual-loop: chunk size is the amortization knob",
                       "The VLM runs once; the chunk spreads its cost across many actions")
    add_bullet_box(s, CONTENT_LEFT, 1.6, CONTENT_W, 3.9, [
        "π0.5 (50-action chunk): 136 ms/chunk → 2.7 ms/action → 367 Hz on 5090, "
        "148.9 Hz on NPU High. The biggest amortization in the bake-off.",
        "NORA-1.5 (5-action chunk): similar chunk latency (~183 ms) but 10× fewer "
        "actions per VLM forward → 27 Hz (21 Hz on High).",
        ("Same dual-loop architecture, 10× the throughput — purely from chunk size.",
         EMPH, True),
        "The fast (denoise) loop degrades gracefully on edge: NORA-1.5's is "
        "launch-bound (~1.4× slower on High), π0.5's is partly BW-bound (~4.2×) — "
        "both far from the single-loop wall, with optimization headroom.",
    ], font_size=15)
    add_text_box(s, CONTENT_LEFT, 5.8, CONTENT_W, 1.0,
                 "⚠ Catch: the flow-matching action head requires floating point. On "
                 "INT8-only NPU Mid it won't run at all (hard gate). Dual-loop needs an "
                 "FP-capable tier (NPU High).", font_size=13, color=C.ACCENT_AMBER)
    return s


def slide_oft(prs):
    s = new_slide(prs)
    add_title_subtitle(s, "OFT parallel-chunk + ternary: the INT8 path",
                       "BitVLA — one forward, no AR wall, runs where the others can't")
    add_bullet_box(s, CONTENT_LEFT, 1.6, CONTENT_W, 3.9, [
        "One parallel VLM forward emits all 8 actions via a regression head — no "
        "token-by-token decode, so no AR bandwidth-wall. 123 ms → 65 Hz on 5090.",
        "Prefill-shaped (compute-bound, util ~9–14%) → projects like a prefill, not a "
        "BW-walled decode. ~8× OpenVLA's per-action rate at similar VLM scale.",
        ("Ternary (1-bit) backbone → 6 GB footprint vs OpenVLA-7B's 14.4 GB, and NO FP "
         "gate: BitVLA is the one model that runs on INT8-only NPU Mid — at 53.5 Hz "
         "(118 Hz on High), not a crawl.", EMPH, True),
    ], font_size=15)
    add_text_box(s, CONTENT_LEFT, 5.6, CONTENT_W, 1.3,
                 "⚠ Honesty caveat: in the public bf16 checkpoint the ternary weights run "
                 "as dense bf16 matmuls — ternary buys MEMORY, not compute speed. The "
                 "paper's 4.4× speedup needs specialized (bitblas/LUT) kernels we did not "
                 "run. Treat the ternary speed/bandwidth win as a separate, optimistic, "
                 "kernel-dependent projection.", font_size=13, color=C.ACCENT_AMBER)
    return s


def slide_multicam(prs):
    s = new_slide(prs)
    add_title_subtitle(s, "Multi-camera: three measured cost shapes",
                       "Real deployments use multiple cameras — each pattern scales differently")
    headers = ["Pattern", "Example", "How cost scales with N cameras", "Edge implication"]
    rows = [
        ["Native\nmulti-cam", "π0.5\n(3 cameras)",
         "vision N× AND LLM prefill grows\n(~12.5 ms/added camera) — each cam\ninjects 256 tokens into the prefix",
         "both stages scale; not\n'free extra cameras'"],
        ["Fleet\nreplication", "any single-cam\nVLA × N robots",
         "everything ×N (N independent\ninstances) — measured exactly 2.00×",
         "linear, predictable;\nshare one NPU serially"],
        ["Stitched\npanorama", "OpenVLA\n(N→1 image)",
         "FLAT — fixed-resolution ViT downscales\nthe panorama; vision cost constant in N",
         "free compute, but lossy\n(resolution per cam drops)"],
    ]
    add_styled_table(s, Inches(0.4), Inches(1.7), Inches(12.5), Inches(3.9),
                     headers, rows,
                     col_widths=[Inches(1.7), Inches(1.9), Inches(5.3), Inches(3.6)],
                     font_size=11, header_font_size=12)
    add_text_box(s, Inches(0.5), Inches(6.1), Inches(12), Inches(0.9),
                 "Measurement corrected two intuitive-but-wrong assumptions: native "
                 "multi-cam is NOT 'LLM-flat' (prefix grows with cameras), and a stitched "
                 "panorama does NOT cost more (it's downscaled). The sizer models all "
                 "three shapes from these anchors.", font_size=12, color=C.TEXT_DIM)
    return s


def slide_methodology(prs):
    s = new_slide(prs)
    add_title_subtitle(s, "Methodology & honesty",
                       "Why these numbers hold up under NXP-internal review")
    add_bullet_box(s, CONTENT_LEFT, 1.6, CONTENT_W, 4.9, [
        "5090 = MEASURED: CUDA-event timing, n=20, per-component vision/LLM splits, "
        "physical FLOP + achieved-util read off the loaded model (not assumed).",
        "Edge = PROJECTED via keyhole-sizer, 5090-anchored. 3-state source badge per "
        "cell: green measured (5090) / blue calibrated (NPU, 5090-anchored) / orange cross-class.",
        "Conservative by design — UN-OPTIMIZED stock-framework floor (no CUDA graphs / "
        "compile / specialized kernels). Real deployments are faster; these are lower bounds.",
        "Caveats stated up front: π0.5 is bf16-AMP (float32 master weights → VRAM/BW are "
        "upper bounds); BitVLA ternary is memory-only in the measured path (speed needs "
        "kernels not run); FP-gate and BW-wall are hard, measured constraints.",
        ("Every number reproducible: scripts/bakeoff_vla.py (5090) + keyhole-sizer "
         "project_vla (edge). Full reviewer briefing: docs/VLA_BAKEOFF_REVIEW.md.", EMPH, True),
    ], font_size=14)
    return s


def slide_takeaways(prs):
    s = new_slide(prs)
    add_title_subtitle(s, "Takeaways for NXP edge silicon",
                       "Matching VLA topology to NPU tier")
    add_bullet_box(s, CONTENT_LEFT, 1.6, CONTENT_W, 4.4, [
        "INT8-only tier (NPU Mid): only OFT/ternary VLAs run usefully — BitVLA at "
        "53.5 Hz. Single-loop is BW-walled to ~1–2 Hz; dual-loop is FP-gated out entirely.",
        "FP-capable tier (NPU High): unlocks dual-loop — π0.5 at ~149 Hz control "
        "(3 cameras), NORA-1.5 at ~21 Hz. This is the high-rate robotics story.",
        ("Differentiator: NPU High can run a 3-camera π0.5 control loop AND keep video "
         "vision streams alongside — a multi-camera robotics deployment competitors "
         "(Qualcomm IQ9, TI TDA5) market but few can size credibly. We can, with "
         "measured anchors.", EMPH, True),
        "Single-loop AR VLAs (the most common open models) are the WRONG fit for edge "
        "without FP-capable high-bandwidth memory — quantify before committing silicon.",
    ], font_size=15)
    add_text_box(s, CONTENT_LEFT, 6.3, CONTENT_W, 0.6,
                 "5 models · 3 topologies · 2 NPU tiers · all measured-anchored — "
                 "the credible multi-camera VLA edge story.",
                 font_size=13, color=C.ACCENT_GREEN, bold=True)
    return s


def build(output: Path):
    prs = Presentation()
    set_deck_size(prs)
    for fn in (slide_title, slide_tldr, slide_topologies, slide_5090_table,
               slide_edge_projection, slide_bw_wall, slide_amortization, slide_oft,
               slide_multicam, slide_methodology, slide_takeaways):
        fn(prs)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))
    print(f"Wrote {output}  ({len(prs.slides._sldIdLst)} slides)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default=str(OUT_DEFAULT))
    build(Path(ap.parse_args().output))


if __name__ == "__main__":
    main()
