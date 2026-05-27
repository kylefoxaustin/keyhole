"""VLA (Vision-Language-Action) bake-off — phase 1: NORA 3B latency baseline on RTX 5090.

Establishes the measurement pattern for VLA workloads so subsequent
models (OpenVLA 7B, NORA-1.5, π0.5, BitVLA) can plug into the same
harness. The CSV at `data/inputs/vla_model_data.csv` is the canonical
catalog; this script defaults to the `nora_3b` row.

What we measure for each model (single-loop autoregressive case):

  1. **VLM forward latency**  — vision encoder + LLM prefill on the
     image + text-prompt pair. Run as `generate(..., max_new_tokens=1)`
     so the cost is fully borne by the prefill pass.

  2. **Action forward latency** — autoregressive token-by-token decode
     for N_ACTION_TOKENS new tokens, starting from the cached prefill.
     Reported as ms-per-token (decode rate) and ms-per-action-chunk.

  3. **End-to-end ms-per-action** — VLM forward + action forward,
     matches the CSV's `measured_5090_ms_per_action` column.

  4. **DRAM footprint**  — peak VRAM during inference via
     torch.cuda.max_memory_allocated. Cross-checked against the CSV's
     `inference_dram_gb_*` columns (which are paper-reported / projected).

  5. **FLOPs (analytical)** — derived from total_params × tokens.
     Per-decode-token: bytes_per_token = total_params × bytes_per_param
     (the bandwidth-bound proxy that matters for edge projection). VLM
     prefill: 2 × vlm_params × (n_image_tokens + n_prompt_tokens) FLOPs.

NVTX ranges (`vla_<vla_key>__vlm` and `vla_<vla_key>__action`) wrap each
phase so `profile_all_ncu.sh` can attribute kernels to the right stage
on follow-up DRAM sweeps.

Outputs:
  data/output/bakeoff/vla_summary.json  — schema mirrors the other
                                          bake-off summaries; the model
                                          key + per-phase measurements
                                          land under `result`.

Usage:
  python scripts/bakeoff_vla.py                     # nora_3b @ bf16, defaults
  python scripts/bakeoff_vla.py --dtype bf16        # explicit dtype
  python scripts/bakeoff_vla.py --model-key nora_3b
  python scripts/bakeoff_vla.py --hf-repo declare-lab/nora --action-tokens 7
  python scripts/bakeoff_vla.py --n-trials 50 --warmup 10

First-run note: the `--hf-repo` default is a best-guess based on the
paper authors. **Verify the repo id before relying on the measurement**
— the CSV does not encode the HuggingFace path. If load fails, run the
script with `--hf-repo <correct-id>` and the error trace will clarify
which transformers/processor combo NORA expects.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bakeoff_vla")

# ────────────────────────── Constants ──────────────────────────

CSV_PATH = REPO / "data" / "inputs" / "vla_model_data.csv"
SUMMARY_PATH = REPO / "data" / "output" / "bakeoff" / "vla_summary.json"
CACHED_FRAME_DIR = REPO / "data" / "frames" / "EW_clip_720p"  # mirrors other bake-offs

DEFAULT_MODEL_KEY = "nora_3b"
# Fallback HF repos used only when the CSV row's hf_repo cell is blank.
# Primary source of truth is the `hf_repo` column in vla_model_data.csv —
# new models should be added there, not here.
DEFAULT_HF_REPO_FOR_MODEL_FALLBACK = {
    "nora_3b":               "declare-lab/nora",
    "nora_1p5":              "declare-lab/nora-1.5",
    "openvla_7b_single":     "openvla/openvla-7b",
    "openvla_7b_cached":     "openvla/openvla-7b",
    "pi_0p5":                "lerobot/pi0_5",
}

N_WARMUP_DEFAULT = 3
N_TRIALS_DEFAULT = 20
ACTION_TOKENS_DEFAULT = 7        # 7-DOF action (NORA / OpenVLA convention)
DEFAULT_PROMPT = "Pick up the red object on the table."

# ────────────────────────── CSV / spec ──────────────────────────

@dataclass
class VLAModelSpec:
    """One row of vla_model_data.csv as a typed record. Unknown / blank
    cells preserve None so the consumer can `if spec.measured_5090_ms_per_action`."""
    vla_key: str
    display_name: str
    hf_repo: str                       # may be "" if model weights not yet public
    architecture: str
    total_params_b: float
    vlm_params_b: float
    action_params_m: float
    vision_encoder: str
    llm_backbone: str
    action_head_type: str
    dtype_path_default: str
    dtype_path_alt: str
    default_vlm_hz: float
    default_action_hz: float
    vlm_hz_min: float
    vlm_hz_max: float
    action_hz_min: float
    action_hz_max: float
    measured_5090_ms_per_action: float | None
    inference_dram_gb_bf16: float | None
    inference_dram_gb_int8: float | None
    inference_dram_gb_int4: float | None
    libero_success_pct: float | None
    source_paper: str
    arxiv_id: str
    citation_year: int
    notes: str


def _f(v: str) -> float | None:
    """Empty CSV cell → None; else float."""
    v = v.strip()
    return float(v) if v else None


def load_vla_catalog(csv_path: Path = CSV_PATH) -> dict[str, VLAModelSpec]:
    """Read the VLA catalog CSV into a {vla_key: VLAModelSpec} dict."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"VLA catalog not found at {csv_path}. "
            f"Expected `data/inputs/vla_model_data.csv` — see repo root."
        )
    out: dict[str, VLAModelSpec] = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["vla_key"]] = VLAModelSpec(
                    vla_key=row["vla_key"],
                    display_name=row["display_name"],
                    hf_repo=row.get("hf_repo", ""),
                    architecture=row["architecture"],
                    total_params_b=float(row["total_params_b"]),
                    vlm_params_b=float(row["vlm_params_b"]),
                    action_params_m=float(row["action_params_m"]),
                    vision_encoder=row["vision_encoder"],
                    llm_backbone=row["llm_backbone"],
                    action_head_type=row["action_head_type"],
                    dtype_path_default=row["dtype_path_default"],
                    dtype_path_alt=row["dtype_path_alt"],
                    default_vlm_hz=float(row["default_vlm_hz"]),
                    default_action_hz=float(row["default_action_hz"]),
                    vlm_hz_min=float(row["vlm_hz_min"]),
                    vlm_hz_max=float(row["vlm_hz_max"]),
                    action_hz_min=float(row["action_hz_min"]),
                    action_hz_max=float(row["action_hz_max"]),
                    measured_5090_ms_per_action=_f(row.get("measured_5090_ms_per_action", "")),
                    inference_dram_gb_bf16=_f(row.get("inference_dram_gb_bf16", "")),
                    inference_dram_gb_int8=_f(row.get("inference_dram_gb_int8", "")),
                    inference_dram_gb_int4=_f(row.get("inference_dram_gb_int4", "")),
                    libero_success_pct=_f(row.get("libero_success_pct", "")),
                    source_paper=row["source_paper"],
                    arxiv_id=row["arxiv_id"],
                    citation_year=int(row["citation_year"]),
                    notes=row["notes"],
                )
            except (KeyError, ValueError) as e:
                log.warning("Skipping malformed row %r: %s", row.get("vla_key", "?"), e)
    return out


# ────────────────────────── Model loading ──────────────────────────

DTYPE_TORCH = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def load_nora_model(hf_repo: str, dtype: str):
    """Load NORA 3B (or any Qwen2.5-VL-based VLA) from HuggingFace.

    Returns (model, processor). The processor wraps both the image
    transform and the text tokenizer. On first-run failure, the trace
    will indicate the right transformers class — common variants:
    `Qwen2VLForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration`,
    or a custom NORA-specific class subclassed from Qwen2VL.

    The NORA paper (arXiv 2504.19854) reports the model fits in 8.3 GB
    VRAM at the default precision; the loader uses `device_map="auto"`
    and a torch dtype matching the CLI flag.
    """
    log.info("Loading VLA model from HF: %s (dtype=%s)", hf_repo, dtype)
    from transformers import AutoModel, AutoProcessor  # type: ignore

    torch_dtype = DTYPE_TORCH[dtype]

    # AutoModel + trust_remote_code is the safest entrypoint for VLAs that
    # ship custom config classes. NORA / OpenVLA / π0.5 all need it.
    processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        hf_repo,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


# ────────────────────────── Frame source ──────────────────────────

def load_test_frame(cached_dir: Path = CACHED_FRAME_DIR) -> "np.ndarray":
    """Use one frame from the cached bake-off EW clip as the visual input.

    VLA latency is content-invariant at fixed input size (the model
    graph + token counts dominate, not pixel content), so a single
    representative frame is sufficient for timing.
    """
    import cv2
    candidates = sorted(cached_dir.glob("frame_*.png"))
    if not candidates:
        # Fall back to any 720p-ish image in data/frames or data/test_clips
        for alt in [REPO / "data" / "frames",
                    REPO / "data" / "test_clips"]:
            if alt.exists():
                candidates = sorted(alt.rglob("*.png")) or sorted(alt.rglob("*.jpg"))
                if candidates:
                    break
    if not candidates:
        raise FileNotFoundError(
            f"No cached frame found under {cached_dir}. "
            f"Other bake-offs source frames from data/frames/EW_clip_720p/; "
            f"ensure the EW clip has been pre-rendered."
        )
    frame_path = candidates[0]
    log.info("Visual input: %s", frame_path.relative_to(REPO))
    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"cv2 failed to decode {frame_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ────────────────────────── Measurement ──────────────────────────

def _cuda_event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    """Synchronize + return elapsed ms between two CUDA events."""
    end.synchronize()
    return float(start.elapsed_time(end))


def measure_vlm_forward(model, processor, image, prompt: str,
                        n_warmup: int, n_trials: int,
                        nvtx_label: str) -> dict[str, Any]:
    """VLM forward = vision encoder + LLM prefill, timed via
    generate(max_new_tokens=1). Returns p50/p95 ms + per-trial.

    The first generated token costs essentially the full prefill (no
    cached past kv at trial start), so this isolates the perception
    cost from the action-decode loop.
    """
    from src.profiling.nvtx_helpers import nvtx_range

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    # Warmup (not timed)
    log.info("VLM warmup: %d forwards", n_warmup)
    for _ in range(n_warmup):
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()

    log.info("VLM timed: %d forwards", n_trials)
    per_trial_ms: list[float] = []
    for i in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with nvtx_range(nvtx_label), torch.inference_mode():
            start.record()
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            end.record()
        per_trial_ms.append(_cuda_event_ms(start, end))

    arr = np.array(per_trial_ms)
    return {
        "n_warmup": n_warmup,
        "n_trials": n_trials,
        "per_trial_ms": per_trial_ms,
        "mean_ms": float(arr.mean()),
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "p99_ms":  float(np.percentile(arr, 99)),
    }


def measure_action_forward(model, processor, image, prompt: str,
                           n_action_tokens: int,
                           n_warmup: int, n_trials: int,
                           nvtx_label: str) -> dict[str, Any]:
    """Action forward = autoregressive decode for N action tokens
    starting from the cached prefill.

    For a single-loop VLA (NORA / OpenVLA / BitVLA) the action chunk
    is `n_action_tokens` tokens (typically 7-DOF discretized).

    Reports both ms-per-chunk and decoded ms-per-token (the
    bandwidth-bound rate that scales linearly with the chunk size).
    """
    from src.profiling.nvtx_helpers import nvtx_range

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    # Warmup
    log.info("Action warmup: %d forwards", n_warmup)
    for _ in range(n_warmup):
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=n_action_tokens, do_sample=False)
    torch.cuda.synchronize()

    log.info("Action timed: %d forwards × %d new tokens", n_trials, n_action_tokens)
    per_trial_ms: list[float] = []
    per_trial_decode_ms: list[float] = []   # excludes prefill (subtract VLM-only)
    for i in range(n_trials):
        start_full = torch.cuda.Event(enable_timing=True)
        end_full   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with nvtx_range(nvtx_label), torch.inference_mode():
            start_full.record()
            _ = model.generate(**inputs, max_new_tokens=n_action_tokens, do_sample=False)
            end_full.record()
        per_trial_ms.append(_cuda_event_ms(start_full, end_full))

    arr = np.array(per_trial_ms)
    return {
        "n_warmup": n_warmup,
        "n_trials": n_trials,
        "n_action_tokens": n_action_tokens,
        "per_trial_ms": per_trial_ms,
        "mean_ms": float(arr.mean()),
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "p99_ms":  float(np.percentile(arr, 99)),
    }


def analytical_bytes_per_decode_token(spec: VLAModelSpec, dtype: str) -> float:
    """Bandwidth-bound proxy for the autoregressive decode pass.

    Bytes-per-token = total_params × bytes_per_param. Matches the LLM
    bake-off math at `bakeoff_llm.py:264` — for an autoregressive model
    every decode step touches the full weight set once.

    Args:
      spec:  VLAModelSpec from CSV
      dtype: "bf16" (2 B/param) | "fp16" (2) | "int8" (1) | "int4" (0.5)
    """
    bytes_per_param = {
        "bf16": 2.0, "fp16": 2.0, "fp32": 4.0,
        "int8": 1.0, "int4": 0.5,
    }[dtype]
    return spec.total_params_b * 1e9 * bytes_per_param


# ────────────────────────── Main ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-key", default=DEFAULT_MODEL_KEY,
                   help=f"VLA catalog key (default: {DEFAULT_MODEL_KEY})")
    p.add_argument("--hf-repo", default=None,
                   help="Override the HF repo id (default: best-guess "
                        "per DEFAULT_HF_REPO_FOR_MODEL; verify first run)")
    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_TORCH),
                   help="Torch dtype to load the model in (default: bf16)")
    p.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                   help=f"Timed forwards per phase (default: {N_TRIALS_DEFAULT})")
    p.add_argument("--warmup", type=int, default=N_WARMUP_DEFAULT,
                   help=f"Warmup forwards per phase (default: {N_WARMUP_DEFAULT})")
    p.add_argument("--action-tokens", type=int, default=ACTION_TOKENS_DEFAULT,
                   help="Number of action tokens to decode per chunk "
                        f"(default: {ACTION_TOKENS_DEFAULT}; 7-DOF convention)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Text instruction passed to the VLA")
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 70)
    log.info("VLA bake-off — phase 1: latency baseline on RTX 5090")
    log.info("=" * 70)
    log.info("CUDA device:  %s", torch.cuda.get_device_name(0)
                                  if torch.cuda.is_available() else "<none>")
    log.info("PyTorch:      %s", torch.__version__)

    # ── 1. Catalog ────────────────────────────────────────────────────
    catalog = load_vla_catalog()
    if args.model_key not in catalog:
        log.error("Unknown model key %r. Available: %s",
                  args.model_key, sorted(catalog))
        sys.exit(2)
    spec = catalog[args.model_key]
    log.info("Model:        %s (%s)", spec.display_name, spec.vla_key)
    log.info("Architecture: %s", spec.architecture)
    log.info("Params:       %.2f B total (%.2f B VLM, %d M action)",
             spec.total_params_b, spec.vlm_params_b, int(spec.action_params_m))
    log.info("Reference:    %s (arXiv %s, %d)",
             spec.source_paper, spec.arxiv_id, spec.citation_year)

    # Resolution order: --hf-repo CLI > CSV hf_repo column > script fallback dict.
    hf_repo = (args.hf_repo
               or (spec.hf_repo or None)
               or DEFAULT_HF_REPO_FOR_MODEL_FALLBACK.get(spec.vla_key))
    if not hf_repo:
        log.error("No HF repo configured for %r. Either add to the CSV's "
                  "hf_repo column or pass --hf-repo <id>.", spec.vla_key)
        sys.exit(2)
    if not args.hf_repo:
        source = "CSV hf_repo column" if spec.hf_repo else "script fallback dict"
        log.warning("Using HF repo %r for %s (source: %s) — VERIFY this is "
                    "correct before relying on the measurement. Override with "
                    "--hf-repo to test alternative paths.",
                    hf_repo, spec.vla_key, source)

    # ── 2. Visual input ──────────────────────────────────────────────
    image = load_test_frame()

    # ── 3. Model load + DRAM baseline ────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    pre_load_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    model, processor = load_nora_model(hf_repo, args.dtype)
    post_load_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    weight_vram_mb = post_load_alloc_mb - pre_load_alloc_mb
    log.info("Weight VRAM:  %.0f MB (%.2f GB at %s)",
             weight_vram_mb, weight_vram_mb / 1024, args.dtype)

    # ── 4. Per-phase latency ─────────────────────────────────────────
    vlm_nvtx    = f"vla_{spec.vla_key}__vlm"
    action_nvtx = f"vla_{spec.vla_key}__action"

    vlm = measure_vlm_forward(
        model, processor, image, args.prompt,
        n_warmup=args.warmup, n_trials=args.n_trials, nvtx_label=vlm_nvtx,
    )
    action = measure_action_forward(
        model, processor, image, args.prompt,
        n_action_tokens=args.action_tokens,
        n_warmup=args.warmup, n_trials=args.n_trials, nvtx_label=action_nvtx,
    )

    # End-to-end = VLM (one new token, ≈ pure prefill) + extra decode for
    # remaining action_tokens-1 tokens. The action_forward measurement
    # already includes prefill + N tokens, so it IS the e2e number.
    e2e_ms = action["p50_ms"]
    decode_only_ms = max(0.0, action["p50_ms"] - vlm["p50_ms"])
    ms_per_decode_token = decode_only_ms / max(1, args.action_tokens - 1)

    # ── 5. DRAM during inference ─────────────────────────────────────
    peak_inference_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ── 6. Bandwidth proxy (analytical) ──────────────────────────────
    bpt = analytical_bytes_per_decode_token(spec, args.dtype)
    decode_tok_s_bw = (peak_inference_mb * 1e6) / bpt if bpt > 0 else 0  # informational only

    # ── 7. Report + write ────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Results for %s @ %s on RTX 5090:", spec.display_name, args.dtype)
    log.info("  VLM forward p50:           %.2f ms", vlm["p50_ms"])
    log.info("  Action chunk p50:          %.2f ms  (%d tokens)",
             action["p50_ms"], args.action_tokens)
    log.info("  Decode ms/token (derived): %.2f ms", ms_per_decode_token)
    log.info("  End-to-end ms/action p50:  %.2f ms", e2e_ms)
    log.info("  Action rate:               %.1f Hz", 1000.0 / e2e_ms if e2e_ms > 0 else 0)
    log.info("  Peak VRAM (incl. weights): %.0f MB (%.2f GB)",
             peak_inference_mb, peak_inference_mb / 1024)
    log.info("  Bytes/decode-token (BW):   %.1f MB  (%s precision)",
             bpt / 1e6, args.dtype)
    if spec.measured_5090_ms_per_action:
        log.info("  CSV reference (5090):      %.1f ms/action",
                 spec.measured_5090_ms_per_action)
    log.info("=" * 70)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": "RTX 5090",
        "torch": torch.__version__,
        "dtype": args.dtype,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_path": str(CSV_PATH.relative_to(REPO)),
        "model_key": spec.vla_key,
        "hf_repo": hf_repo,
        "model_spec": asdict(spec),
        "result": {
            "vlm_forward": vlm,
            "action_forward": action,
            "derived": {
                "ms_per_decode_token": round(ms_per_decode_token, 3),
                "e2e_ms_per_action_p50": round(e2e_ms, 3),
                "action_rate_hz_p50": round(1000.0 / e2e_ms, 2) if e2e_ms > 0 else 0,
            },
            "dram": {
                "weight_vram_mb": round(weight_vram_mb, 1),
                "weight_vram_gb": round(weight_vram_mb / 1024, 2),
                "peak_inference_mb": round(peak_inference_mb, 1),
                "peak_inference_gb": round(peak_inference_mb / 1024, 2),
                "bytes_per_decode_token_analytical": round(bpt, 0),
                "bytes_per_decode_token_unit": "bytes",
            },
            "csv_reference": {
                "measured_5090_ms_per_action": spec.measured_5090_ms_per_action,
                "inference_dram_gb_at_default": getattr(
                    spec, f"inference_dram_gb_{spec.dtype_path_default}", None
                ),
            },
            "nvtx_labels": {
                "vlm":    vlm_nvtx,
                "action": action_nvtx,
            },
        },
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, default=str))
    log.info("Wrote %s", SUMMARY_PATH.relative_to(REPO))

    # ── 8. ncu sweep follow-up note ──────────────────────────────────
    log.info("")
    log.info("Next step — DRAM measurement via ncu:")
    log.info("  Add a block to scripts/profile_all_ncu.sh that runs:")
    log.info("    python scripts/bakeoff_vla.py --model-key %s --n-trials 5", spec.vla_key)
    log.info("  with NVTX-range filters for `%s` and `%s`.",
             vlm_nvtx, action_nvtx)
    log.info("  Output JSON will land at data/output/ncu/vla_<key>.json.")


if __name__ == "__main__":
    main()
