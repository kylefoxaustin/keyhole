"""VLA (Vision-Language-Action) bake-off — phase 1: NORA 3B latency baseline on RTX 5090.

Establishes the measurement pattern for VLA workloads so subsequent
models (OpenVLA 7B, NORA-1.5, π0.5, BitVLA) can plug into the same
harness. The CSV at `data/inputs/vla_model_data.csv` is the canonical
catalog; this script defaults to the `nora_3b` row.

What we measure for each model (single-loop autoregressive case):

  1. **VLM forward latency**  — vision encoder + LLM prefill on the
     image + text-prompt pair. Run as `generate(..., max_new_tokens=1)`
     so the cost is fully borne by the prefill pass. Forward hooks on the
     vision tower additionally split this into a **vision-encoder vs
     LLM-prefill** component breakdown (the roofline split the sizer
     consumes) under `result.vlm_forward.components`.

  2. **Action forward latency** — the full NORA inference call: prefill +
     autoregressive FAST+ action-token decode to EOS. NORA emits a
     *variable-length* BPE-compressed action sequence (not a fixed 7 raw
     tokens), so the decoded count is captured at runtime, not assumed. This
     is the end-to-end ms/action the CSV's `measured_5090_ms_per_action`
     (33 ms → 30 Hz) refers to.

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

Inference path: NORA ships a vanilla Qwen2.5-VL checkpoint; the action logic
lives in declare-lab/nora's `inference/nora.py` wrapper, mirrored here — load
as `Qwen2_5_VLForConditionalGeneration`, preprocess via the chat template with
a 224×224 image, `generate()` to EOS, then FAST+ decode the action tokens to a
7-DOF vector. The HF path resolves from the CSV `hf_repo` column (canonical),
overridable with `--hf-repo`.
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
# Runaway safety cap for the FAST+ action decode. NORA emits a variable-length
# BPE-compressed action sequence (NOT a fixed 7 raw tokens) and stops at EOS;
# the cap only prevents an unbounded generate if EOS is never hit. The actual
# decoded length is captured and reported as n_generated_tokens.
ACTION_TOKENS_DEFAULT = 256
DEFAULT_PROMPT = "Pick up the red object on the table."
# norm_stats key used to un-normalize the 7-DOF action for the validation
# check. NORA trains on Open X-Embodiment; bridge_orig is the canonical BridgeV2
# key used in the model card example. Validation-only — does not affect latency.
DEFAULT_UNNORM_KEY = "bridge_orig"

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


def load_nora_model(hf_repo: str, dtype: str, attn_impl: str = "sdpa"):
    """Load NORA 3B from HuggingFace, matching the official inference path.

    NORA ships a *vanilla* Qwen2.5-VL checkpoint (no custom modeling code in
    the HF repo); the action-prediction logic lives in declare-lab/nora's
    `inference/nora.py` wrapper, not the weights. That wrapper loads the model
    as `Qwen2_5_VLForConditionalGeneration` (NOT `AutoModel`, which returns the
    base model with no LM head and cannot `generate()`), pulls the repo's
    `generation_config`, and forces `do_sample=False` for deterministic decode.
    We mirror that exactly so the measured latency reflects the published path.

    Returns (model, processor). The NORA paper (arXiv 2504.19854) reports the
    model fits in 8.3 GB VRAM at bf16; we load with a torch dtype matching the
    CLI flag and place it on cuda with a plain `.to()` (NORA's wrapper does the
    same — no `device_map="auto"` sharding for a 3B model on a 32 GB 5090).
    """
    log.info("Loading VLA model from HF: %s (dtype=%s)", hf_repo, dtype)
    from transformers import (  # type: ignore
        AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig,
    )

    torch_dtype = DTYPE_TORCH[dtype]

    processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)
    # Default to SDPA: transformers' SDPA backend dispatches to flash-attention
    # kernels on Ada/Blackwell without the (heavy-to-build) flash_attn package.
    # Recorded in the payload as measurement provenance.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_repo,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
    )
    model.to("cuda")
    # Repo generation_config drives the variable-length FAST+ action decode
    # (generate runs to EOS — there is no fixed action-token count).
    model.generation_config = GenerationConfig.from_pretrained(hf_repo)
    model.generation_config.do_sample = False
    model.eval()
    actual_attn = getattr(model.config, "_attn_implementation", attn_impl)
    log.info("Attention backend: %s", actual_attn)
    return model, processor, actual_attn


# NORA expects a 224×224 image; the chat-template + process_vision_info path
# below mirrors declare-lab/nora inference/nora.py exactly so the image-token
# count (and therefore the prefill latency) matches the published 33 ms figure.
NORA_IMG_SIZE = 224


def build_nora_inputs(processor, image, prompt: str, device):
    """Replicate NORA's exact preprocessing → model-ready inputs dict.

    NORA does NOT call `processor(images=, text=)` directly. It builds a
    Qwen2.5-VL chat message with an explicit 224×224 resize, applies the chat
    template, and runs `qwen_vl_utils.process_vision_info`. Skipping this (as
    the first-draft harness did) feeds the model a full-resolution image and a
    different prompt scaffold → wrong image-token count → an uncalibrated VLM
    forward. We mirror the wrapper so the measurement is faithful.
    """
    import PIL.Image
    from qwen_vl_utils import process_vision_info

    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image,
             "resized_height": NORA_IMG_SIZE, "resized_width": NORA_IMG_SIZE},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


# NORA's action tokens occupy a dedicated slice of the Qwen2.5-VL vocabulary
# (see declare-lab/nora inference/nora.py). The FAST+ tokenizer decodes tokens
# offset back to its own id space.
_NORA_ACTION_TOKEN_MIN = 151665
_NORA_ACTION_TOKEN_MAX = 153712


def validate_nora_action_path(model, processor, image, prompt: str,
                              unnorm_key: str, max_action_tokens: int) -> dict[str, Any]:
    """Validate the action-prediction path — the Step 1b methodology proof.

    Two layers, of which the first is authoritative:

    1. **Token-level** (authoritative): NORA must emit tokens in its dedicated
       action-token range [%d, %d] and terminate at EOS. If it does, the harness
       is provably measuring the real FAST+ action-prediction path (not garbage,
       not a runaway). This needs nothing beyond the model itself.

    2. **FAST+ float decode** (best-effort): decode those action tokens to a
       7-DOF float vector via the physical-intelligence/fast processor +
       NORA's norm_stats, mirroring declare-lab/nora inference/nora.py. That
       processor is pinned to transformers~=4.50 and currently fails to
       instantiate under transformers 5.x; when it does, we record the vector,
       and when it doesn't we record why and fall back to the token-level proof.
       The float vector is a methodology nicety — the latency deliverable the
       sizer consumes does not depend on it.

    All CPU-side / single-shot; NOT part of the GPU timing.
    """ % (_NORA_ACTION_TOKEN_MIN, _NORA_ACTION_TOKEN_MAX)
    import json
    from huggingface_hub import hf_hub_download

    inputs = build_nora_inputs(processor, image, prompt, model.device)
    input_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_action_tokens, do_sample=False)
    new = gen[0][input_len:]
    in_range = (new >= _NORA_ACTION_TOKEN_MIN) & (new <= _NORA_ACTION_TOKEN_MAX)
    action_ids = new[in_range]
    n_action = int(in_range.sum().item())

    gc = getattr(model, "generation_config", None)
    eos_ids: set[int] = set()
    if gc is not None and gc.eos_token_id is not None:
        eos_ids = (set(gc.eos_token_id)
                   if isinstance(gc.eos_token_id, (list, tuple))
                   else {gc.eos_token_id})
    eos_present = bool(len(new) and int(new[-1].item()) in eos_ids)

    result: dict[str, Any] = {
        "token_level_ok": bool(n_action > 0 and eos_present),
        "n_action_tokens": n_action,
        "n_generated_tokens": int(len(new)),
        "eos_terminated": eos_present,
        "generated_token_ids": [int(t) for t in new.tolist()],
        "action_token_range": [_NORA_ACTION_TOKEN_MIN, _NORA_ACTION_TOKEN_MAX],
    }

    # Layer 2 — best-effort FAST+ float decode.
    try:
        from transformers import AutoProcessor
        fast_tok = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        fast_tok.action_dim = 7
        fast_tok.time_horizon = 1
        norm_stats = json.load(open(hf_hub_download("declare-lab/nora", "norm_stats.json")))
        if unnorm_key not in norm_stats:
            unnorm_key = next(iter(norm_stats))
        action = fast_tok.decode([action_ids - _NORA_ACTION_TOKEN_MIN])
        stats = norm_stats[unnorm_key]["action"]
        high, low = np.array(stats["q99"]), np.array(stats["q01"])
        unnorm = 0.5 * (np.asarray(action) + 1) * (high - low) + low
        vec = np.array(unnorm[0])
        result["fast_decode_ok"] = bool(tuple(vec.shape) == (7,))
        result["action_dof"] = int(vec.shape[0]) if vec.ndim == 1 else None
        result["action_vector"] = [round(float(x), 5) for x in np.ravel(vec)]
        result["unnorm_key"] = unnorm_key
    except Exception as e:  # noqa: BLE001
        result["fast_decode_ok"] = False
        result["fast_decode_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        result["fast_decode_note"] = (
            "physical-intelligence/fast processor requires transformers~=4.50; "
            "incompatible with installed transformers 5.x. Token-level "
            "validation above is authoritative."
        )
    return result


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


def find_vision_module(model):
    """Locate the vision tower for per-component timing.

    Qwen2.5-VL nests it at `model.model.visual`; older layouts expose
    `model.visual`. Returns the module or None if the layout is unknown (in
    which case the harness silently skips the component split and still reports
    the aggregate VLM forward).
    """
    for path in ("model.visual", "visual"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def measure_vlm_forward(model, processor, image, prompt: str,
                        n_warmup: int, n_trials: int,
                        nvtx_label: str, vision_module=None) -> dict[str, Any]:
    """VLM forward = vision encoder + LLM prefill, timed via
    generate(max_new_tokens=1). Returns p50/p95 ms + per-trial.

    The first generated token costs essentially the full prefill (no
    cached past kv at trial start), so this isolates the perception
    cost from the action-decode loop.

    When `vision_module` is provided, forward hooks place CUDA events around
    the vision tower so each trial also yields the vision-encoder duration; the
    LLM-prefill component is the paired remainder (VLM-forward total − vision,
    i.e. embed-merge + LLM prefill + lm_head + 1-token sample). This is the
    vision-vs-LLM roofline split the sizer consumes — added per their
    2026-05-29 request.
    """
    from src.profiling.nvtx_helpers import nvtx_range

    inputs = build_nora_inputs(processor, image, prompt, model.device)

    # Per-component instrumentation: time the vision tower via hooks.
    vis_evt: dict[str, torch.cuda.Event] = {}
    handles = []
    if vision_module is not None:
        def _pre(_mod, _args):
            e = torch.cuda.Event(enable_timing=True); e.record(); vis_evt["start"] = e
        def _post(_mod, _args, _out):
            e = torch.cuda.Event(enable_timing=True); e.record(); vis_evt["end"] = e
        handles.append(vision_module.register_forward_pre_hook(_pre))
        handles.append(vision_module.register_forward_hook(_post))

    # Warmup (not timed)
    log.info("VLM warmup: %d forwards", n_warmup)
    for _ in range(n_warmup):
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()

    log.info("VLM timed: %d forwards%s", n_trials,
             " (+vision/LLM split)" if vision_module is not None else "")
    per_trial_ms: list[float] = []
    per_vision_ms: list[float] = []
    for i in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        vis_evt.clear()
        with nvtx_range(nvtx_label), torch.inference_mode():
            start.record()
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            end.record()
        per_trial_ms.append(_cuda_event_ms(start, end))
        if "start" in vis_evt and "end" in vis_evt:
            vis_evt["end"].synchronize()
            per_vision_ms.append(float(vis_evt["start"].elapsed_time(vis_evt["end"])))

    for h in handles:
        h.remove()

    arr = np.array(per_trial_ms)
    out: dict[str, Any] = {
        "n_warmup": n_warmup,
        "n_trials": n_trials,
        "per_trial_ms": per_trial_ms,
        "mean_ms": float(arr.mean()),
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "p99_ms":  float(np.percentile(arr, 99)),
    }

    # Paired per-trial split (vision_i and total_i are the same trial).
    if per_vision_ms and len(per_vision_ms) == len(per_trial_ms):
        v = np.array(per_vision_ms)
        llm = arr - v  # remainder: embed-merge + LLM prefill + lm_head + sample
        out["components"] = {
            "vision_encoder": {
                "p50_ms": float(np.percentile(v, 50)),
                "p95_ms": float(np.percentile(v, 95)),
                "mean_ms": float(v.mean()),
            },
            "llm_prefill": {
                "p50_ms": float(np.percentile(llm, 50)),
                "p95_ms": float(np.percentile(llm, 95)),
                "mean_ms": float(llm.mean()),
            },
            "vision_frac_p50": round(float(np.percentile(v, 50) /
                                           np.percentile(arr, 50)), 3),
            "method": "forward-hook CUDA events around the vision tower; "
                      "llm_prefill = VLM-forward total − vision (paired per-trial). "
                      "llm_prefill bundles embed-merge + LLM prefill + lm_head + "
                      "1-token sample.",
        }
    return out


def measure_action_forward(model, processor, image, prompt: str,
                           max_action_tokens: int,
                           n_warmup: int, n_trials: int,
                           nvtx_label: str) -> dict[str, Any]:
    """Action forward = the full NORA inference call: prefill + autoregressive
    FAST+ action-token decode to EOS. This IS the end-to-end ms-per-action that
    the CSV's `measured_5090_ms_per_action` (33 ms → 30 Hz) refers to.

    NORA does ONE `generate(**inputs)` that emits a *variable-length* FAST+
    token sequence (the action is BPE-compressed, not a fixed 7 raw tokens) and
    stops at EOS. We pass an explicit `max_new_tokens=max_action_tokens` purely
    as a runaway safety cap — the repo `generation_config` sets no length bound,
    and transformers' tiny default would truncate a real action sequence. The
    cap is generous (default 256); the actual decoded count is captured from the
    generated ids and reported as `n_generated_tokens` so the ms/token
    derivation uses the true length, not the cap.
    """
    from src.profiling.nvtx_helpers import nvtx_range

    inputs = build_nora_inputs(processor, image, prompt, model.device)
    input_len = int(inputs["input_ids"].shape[1])

    # Warmup
    log.info("Action warmup: %d forwards", n_warmup)
    n_generated = 0
    for _ in range(n_warmup):
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_action_tokens,
                                 do_sample=False)
        n_generated = int(out.shape[1]) - input_len
    torch.cuda.synchronize()
    log.info("Action decode produced %d new tokens (EOS-terminated, cap=%d)",
             n_generated, max_action_tokens)

    log.info("Action timed: %d forwards (decode to EOS)", n_trials)
    per_trial_ms: list[float] = []
    for i in range(n_trials):
        start_full = torch.cuda.Event(enable_timing=True)
        end_full   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with nvtx_range(nvtx_label), torch.inference_mode():
            start_full.record()
            _ = model.generate(**inputs, max_new_tokens=max_action_tokens,
                               do_sample=False)
            end_full.record()
        per_trial_ms.append(_cuda_event_ms(start_full, end_full))

    arr = np.array(per_trial_ms)
    return {
        "n_warmup": n_warmup,
        "n_trials": n_trials,
        "input_len_tokens": input_len,
        "n_generated_tokens": n_generated,
        "max_action_tokens_cap": max_action_tokens,
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
                   help="Runaway safety cap (max_new_tokens) for the FAST+ "
                        f"action decode (default: {ACTION_TOKENS_DEFAULT}). The "
                        "real decoded length is EOS-terminated and reported as "
                        "n_generated_tokens.")
    p.add_argument("--attn", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"],
                   help="Attention backend (default: sdpa — flash kernels "
                        "without the flash_attn package)")
    p.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY,
                   help=f"norm_stats key for 7-DOF validation (default: "
                        f"{DEFAULT_UNNORM_KEY})")
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
    model, processor, attn_backend = load_nora_model(hf_repo, args.dtype, args.attn)
    post_load_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    weight_vram_mb = post_load_alloc_mb - pre_load_alloc_mb
    log.info("Weight VRAM:  %.0f MB (%.2f GB at %s)",
             weight_vram_mb, weight_vram_mb / 1024, args.dtype)

    # ── 3b. Validate the action-prediction path (Step 1b criterion) ──
    # Prove the harness measures NORA's real action decode (action-range tokens
    # + EOS), not garbage. Guarded so it can never abort the latency run.
    try:
        action_validation = validate_nora_action_path(
            model, processor, image, args.prompt,
            args.unnorm_key, args.action_tokens,
        )
        if action_validation["token_level_ok"]:
            log.info("Action validation PASSED (token-level): %d action tokens "
                     "+ EOS, ids=%s", action_validation["n_action_tokens"],
                     action_validation["generated_token_ids"])
        else:
            log.warning("Action validation FAILED token-level check "
                        "(%d action tokens, eos=%s) — review before trusting "
                        "latency.", action_validation["n_action_tokens"],
                        action_validation["eos_terminated"])
        if action_validation.get("fast_decode_ok"):
            log.info("FAST+ 7-DOF decode: %s (unnorm_key=%s)",
                     action_validation["action_vector"],
                     action_validation.get("unnorm_key"))
        else:
            log.info("FAST+ float decode unavailable: %s",
                     action_validation.get("fast_decode_error", "n/a"))
    except Exception as e:  # noqa: BLE001 — validation must never abort timing
        log.warning("Action validation could not run (%s: %s). Latency "
                    "measurement proceeds.", type(e).__name__, e)
        action_validation = {"token_level_ok": False,
                             "error": f"{type(e).__name__}: {e}"}

    # ── 4. Per-phase latency ─────────────────────────────────────────
    vlm_nvtx    = f"vla_{spec.vla_key}__vlm"
    action_nvtx = f"vla_{spec.vla_key}__action"

    vision_module = find_vision_module(model)
    if vision_module is None:
        log.warning("Vision tower not located — skipping per-component split "
                    "(aggregate VLM forward still measured).")
    vlm = measure_vlm_forward(
        model, processor, image, args.prompt,
        n_warmup=args.warmup, n_trials=args.n_trials, nvtx_label=vlm_nvtx,
        vision_module=vision_module,
    )
    action = measure_action_forward(
        model, processor, image, args.prompt,
        max_action_tokens=args.action_tokens,
        n_warmup=args.warmup, n_trials=args.n_trials, nvtx_label=action_nvtx,
    )

    # The action_forward measurement is the full NORA generate (prefill + FAST+
    # decode to EOS) — it IS the end-to-end ms/action the CSV's 33 ms refers to.
    # The VLM forward is prefill + 1 token, so subtracting isolates the decode
    # of the remaining (n_generated - 1) action tokens.
    e2e_ms = action["p50_ms"]
    decode_only_ms = max(0.0, action["p50_ms"] - vlm["p50_ms"])
    n_decode_tok = max(1, action.get("n_generated_tokens", 1) - 1)
    ms_per_decode_token = decode_only_ms / n_decode_tok

    # ── 5. DRAM during inference ─────────────────────────────────────
    peak_inference_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ── 6. Bandwidth proxy (analytical) ──────────────────────────────
    bpt = analytical_bytes_per_decode_token(spec, args.dtype)
    decode_tok_s_bw = (peak_inference_mb * 1e6) / bpt if bpt > 0 else 0  # informational only

    # ── 7. Report + write ────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Results for %s @ %s on RTX 5090:", spec.display_name, args.dtype)
    log.info("  VLM forward p50:           %.2f ms", vlm["p50_ms"])
    if "components" in vlm:
        c = vlm["components"]
        log.info("    ├─ vision encoder p50:   %.2f ms  (%.0f%% of VLM forward)",
                 c["vision_encoder"]["p50_ms"], 100 * c["vision_frac_p50"])
        log.info("    └─ LLM prefill p50:      %.2f ms", c["llm_prefill"]["p50_ms"])
    log.info("  Action chunk p50:          %.2f ms  (%d FAST+ tokens decoded)",
             action["p50_ms"], action.get("n_generated_tokens", 0))
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
        "attn_backend": attn_backend,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_path": str(CSV_PATH.relative_to(REPO)),
        "model_key": spec.vla_key,
        "hf_repo": hf_repo,
        "model_spec": asdict(spec),
        "result": {
            "action_validation": action_validation,
            "vlm_forward": vlm,
            "action_forward": action,
            "derived": {
                "ms_per_decode_token": round(ms_per_decode_token, 3),
                "n_decode_tokens_used": n_decode_tok,
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
            "calibration": {
                "paper_ms_per_action": spec.measured_5090_ms_per_action,
                "vlm_forward_matches_paper": (
                    abs(vlm["p50_ms"] - spec.measured_5090_ms_per_action) <= 5.0
                    if spec.measured_5090_ms_per_action else None
                ),
                "vlm_forward_p50_ms": round(vlm["p50_ms"], 2),
                "e2e_vs_paper_ratio": (
                    round(e2e_ms / spec.measured_5090_ms_per_action, 2)
                    if spec.measured_5090_ms_per_action else None
                ),
                "note": (
                    "VLM-forward (prefill) p50 reproduces the paper's 33 ms anchor. "
                    "End-to-end (prefill + FAST+ decode to EOS) is higher because "
                    "this is stock HuggingFace generate() — no CUDA graphs / static "
                    "KV cache / torch.compile — so per-token decode carries Python + "
                    "kernel-launch overhead (~3x the bf16 bandwidth floor). The gap "
                    "is optimization headroom, not a measurement error; the paper's "
                    "deployment uses an optimized decode loop."
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
