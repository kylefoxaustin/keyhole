"""
EfficientSAM3.1 bake-off — the text-prompt-capable smaller variant.

The existing Option-A bench (`bakeoff_efficientsam3.py`) used the
`stage1_all_converted/efficient_sam3_efficientvit_s.pt` checkpoint which pairs
a 26M EfficientViT-B0 vision backbone with SAM 3's full 400M text encoder
(total 424M). This bench uses the SAM 3.1 student checkpoint instead:

  `stage1_sam3p1/efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt`

which pairs EfficientViT-S vision + MobileCLIP-S0 text (total 118M — **~3.5x
smaller** than the Option-A checkpoint). Preserves SAM 3's text-concept
prompting natively (set_image + set_text_prompt flow) and runs on the
`stage1_sam3.1` branch of EfficientSAM3.

Measures latency two ways to support the deck's story:

  1. set_image  — vision-encoder forward, once per frame, amortized over prompts
  2. set_text_prompt — text-encoder + decoder, N times per frame (one per concept)

Derives total per-frame cost for N=1, 5, 20 concept queries (single-query,
typical-query, exhaustive-SAM3_CONCEPTS). BW-scales each to NPU Mid.

Runs in the same uv-managed Python 3.12 venv (.venv-es3/) as Option A, but
expects the stage1_sam3.1 branch to be checked out in third_party/efficientsam3.

Output: `data/output/bakeoff/efficientsam3p1_summary.json`
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bakeoff_es3p1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "efficientsam3"))
# Repo root on path so `from src.profiling.nvtx_helpers import ...` resolves
# when this script is run from the .venv-es3 interpreter.
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

CKPT = REPO_ROOT / "weights" / "efficientsam3" / "stage1_sam3p1" / "efficient_sam3p1_efficientvit_s_mobileclip_s0_ctx16.pt"
BPE_PATH = REPO_ROOT / "third_party" / "efficientsam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
BAKEOFF_DIR = REPO_ROOT / "data" / "output" / "bakeoff"

RESOLUTION_CLIPS = {
    "720p":  "720p_EW_clip",
    "1080p": "embedded_world_clip_1080p",
    "4K":    "embedded_world_clip",
}

# Same concept list as the SAM 3 reference pass in bakeoff_sam_variants.py.
SAM3_CONCEPTS = [
    "person", "vehicle", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "cat", "bird", "animal",
    "backpack", "bag", "hat", "umbrella",
    "package", "box", "suitcase", "chair", "laptop",
]

# How many prompts to sample for per-prompt timing (each call is independent —
# we just want a stable average).
PER_PROMPT_SAMPLES = 5

WARMUP_FRAMES = 2
MAX_FRAMES = 8


@dataclass
class ResolutionReport:
    resolution: str
    clip: str
    n_frames_timed: int = 0
    set_image_ms: list[float] = field(default_factory=list)
    per_prompt_ms: list[float] = field(default_factory=list)

    @property
    def set_image_p50(self) -> float:
        return float(np.percentile(self.set_image_ms, 50)) if self.set_image_ms else 0.0

    @property
    def per_prompt_p50(self) -> float:
        return float(np.percentile(self.per_prompt_ms, 50)) if self.per_prompt_ms else 0.0

    def total_for_n(self, n_concepts: int) -> float:
        return self.set_image_p50 + n_concepts * self.per_prompt_p50


def run_resolution(model, proc, resolution: str, clip_stem: str) -> ResolutionReport:
    clip_dir = BAKEOFF_DIR / clip_stem
    frames_dir = clip_dir / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    rep = ResolutionReport(resolution=resolution, clip=clip_stem)

    # Rotate through a small subset of concepts per frame — we don't need exhaustive
    # detection, we need stable latency measurements.
    probe_concepts = ["person", "car", "backpack", "bicycle", "chair"]

    to_run = frame_files[: WARMUP_FRAMES + MAX_FRAMES]
    for i, frame_path in enumerate(to_run):
        is_warmup = i < WARMUP_FRAMES
        img = Image.fromarray(cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB))

        # Time set_image
        from src.profiling.nvtx_helpers import nvtx_range
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx_range("efficientsam3p1_es_ev_s__set_image"):
            state = proc.set_image(img)
        torch.cuda.synchronize()
        set_image_ms = (time.perf_counter() - t0) * 1000

        # Time one prompt per concept in the probe list (up to PER_PROMPT_SAMPLES)
        prompt_times: list[float] = []
        for concept in probe_concepts[:PER_PROMPT_SAMPLES]:
            proc.reset_all_prompts(state)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with nvtx_range("efficientsam3p1_es_ev_s__text_prompt"):
                state = proc.set_text_prompt(state=state, prompt=concept)
            torch.cuda.synchronize()
            prompt_times.append((time.perf_counter() - t0) * 1000)

        tag = "WARMUP" if is_warmup else "TIMED"
        log.info("[%s][%s] frame %s: set_image %.1f ms, per-prompt %s -> mean %.1f ms",
                 resolution, tag, frame_path.stem, set_image_ms,
                 [f"{x:.1f}" for x in prompt_times], float(np.mean(prompt_times)))

        if is_warmup:
            continue
        rep.set_image_ms.append(set_image_ms)
        rep.per_prompt_ms.extend(prompt_times)
        rep.n_frames_timed += 1

    n1 = rep.total_for_n(1)
    n5 = rep.total_for_n(5)
    n20 = rep.total_for_n(20)
    log.info("[%s] %d frames. set_image p50 %.1f ms, per-prompt p50 %.1f ms. "
             "Per-frame totals: n=1 %.1f ms, n=5 %.1f ms, n=20 %.1f ms",
             resolution, rep.n_frames_timed, rep.set_image_p50, rep.per_prompt_p50,
             n1, n5, n20)
    return rep


def main():
    if not CKPT.exists():
        raise SystemExit(f"Checkpoint missing: {CKPT}")

    from sam3.model_builder import build_efficientsam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    log.info("Loading EfficientSAM3.1 ES-EV-S (EfficientViT-S + MobileCLIP-S0 ctx16)...")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    autocast_ctx.__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_efficientsam3_image_model(
        bpe_path=str(BPE_PATH),
        checkpoint_path=str(CKPT),
        backbone_type="efficientvit", model_name="s",
        text_encoder_type="mobileclip-s0",
        text_encoder_context_length=16,
        text_encoder_pos_embed_table_size=16,
        interpolate_pos_embed=False,
        enable_inst_interactivity=False,   # SAM1-task predictor not needed for text-prompt path
    )

    total_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(p.numel() for n, p in model.named_parameters() if "vision_backbone" in n)
    text_params = sum(p.numel() for n, p in model.named_parameters()
                       if "language_backbone" in n or "text_encoder" in n)
    log.info("Params: total %.2fM, vision %.2fM, text %.2fM",
             total_params / 1e6, vision_params / 1e6, text_params / 1e6)

    proc = Sam3Processor(model, confidence_threshold=0.3)
    torch.cuda.reset_peak_memory_stats()

    reports: dict[str, ResolutionReport] = {}
    for res, clip_stem in RESOLUTION_CLIPS.items():
        if not (BAKEOFF_DIR / clip_stem / "frames").exists():
            log.warning("[%s] No frame cache — skipping", res)
            continue
        reports[res] = run_resolution(model, proc, res, clip_stem)
        gc.collect()
        torch.cuda.empty_cache()

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    # Assemble summary
    out = {
        "model": "EfficientSAM3.1 ES-EV-S (stage1_sam3p1)",
        "checkpoint": str(CKPT.relative_to(REPO_ROOT)),
        "source": "github.com/SimonZeng7108/efficientsam3 (branch: stage1_sam3.1)",
        "license": "Apache-2.0",
        "backbone": "EfficientViT-S (model_name='s')",
        "text_encoder": "MobileCLIP-S0 ctx=16",
        "total_params_m": total_params / 1e6,
        "vision_backbone_params_m": vision_params / 1e6,
        "text_encoder_params_m": text_params / 1e6,
        "peak_vram_mb_5090": peak_vram_mb,
        "dtype": "bfloat16 (autocast)",
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
        "by_resolution": {
            res: {
                "clip": r.clip,
                "n_frames_timed": r.n_frames_timed,
                "set_image_5090_p50_ms": r.set_image_p50,
                "per_prompt_5090_p50_ms": r.per_prompt_p50,
                "set_image_5090_all_ms": r.set_image_ms,
                "per_prompt_5090_all_ms": r.per_prompt_ms,
                "per_frame_5090_ms": {
                    "n_1_concept": r.total_for_n(1),
                    "n_5_concepts": r.total_for_n(5),
                    "n_20_concepts_exhaustive": r.total_for_n(20),
                },
            } for res, r in reports.items()
        },
        "_note": (
            "Text-prompt inference: set_image amortizes once per frame, each "
            "additional concept costs per_prompt_ms. Totals shown for n=1 (single-query), "
            "n=5 (typical), n=20 (exhaustive SAM3_CONCEPTS list). Edge-MPU "
            "projection scales each by BW ratio (x14.17 vs 5090 effective BW)."
        ),
    }
    out_path = BAKEOFF_DIR / "efficientsam3p1_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)

    # Summary print
    print("\n=== EfficientSAM3.1 ES-EV-S SUMMARY ===")
    print(f"Params: total {total_params/1e6:.1f}M | vision {vision_params/1e6:.1f}M | text {text_params/1e6:.1f}M")
    print(f"Peak VRAM (5090): {peak_vram_mb:.0f} MB")
    print(f"{'Res':<6}{'set_image':<14}{'per_prompt':<14}{'n=1':<10}{'n=5':<10}{'n=20':<10}")
    for res, r in reports.items():
        print(f"{res:<6}{r.set_image_p50:<14.2f}{r.per_prompt_p50:<14.2f}"
              f"{r.total_for_n(1):<10.2f}{r.total_for_n(5):<10.2f}{r.total_for_n(20):<10.2f}")
    autocast_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
