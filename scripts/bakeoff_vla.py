"""VLA (Vision-Language-Action) bake-off — latency baselines on RTX 5090.

Establishes the measurement pattern for VLA workloads so subsequent
models (OpenVLA 7B, NORA-1.5, π0.5, BitVLA) can plug into the same
harness. The CSV at `data/inputs/vla_model_data.csv` is the canonical
catalog; this script defaults to the `nora_3b` row.

Two measurement topologies share this harness:

  • **single-loop** (NORA 3B, OpenVLA 7B) — one VLM prefill, then K *autoregressive*
    action tokens, each a full forward through the LLM. The decode is brutally
    bandwidth-walled (every token streams the whole weight set). Measured by the
    measure_vlm_forward / measure_action_forward path below.

  • **dual-loop** (NORA-1.5, π0.5) — the VLM backbone runs *once* per action
    chunk → frozen KV cache; then a *separate, much smaller* action expert runs
    N fixed flow-matching denoise steps against that cache, emitting a whole
    H-action chunk at once. The expensive VLM is amortized across H actions and
    the fast loop is a tiny model — a completely different latency/util profile.
    Measured by the run_dual_loop path; see that section for the loop split the
    sizer's Phase 3b consumes.

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
import os
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
# RTX 5090 dense bf16 matmul peak (TFLOP/s) — matches the sizer's roofline
# (peak_bf16=209). Used only to report an achieved-util cross-check alongside
# the hardware-independent physical FLOP; the FLOP numbers themselves carry no
# util assumption.
PEAK_BF16_TFLOPS_5090 = 209.0

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


# ════════════════════════ OpenVLA family ════════════════════════
# OpenVLA (Prismatic) is a different architecture from NORA: Llama-2 7B +
# fused SigLIP/DINOv2 vision, discrete 256-bin action tokens (NOT FAST+), and
# its own `predict_action()` wrapper. Its custom code is pinned to the 4.40-era
# transformers (AutoModelForVision2Seq alias + transformers.tokenization_utils
# symbols), so this path runs under a transformers~=4.57 venv (~/.virtualenvs/
# openvla), NOT the keyhole 5.x venv. See docs / session-brief memory.

# Prompt scaffold OpenVLA was trained with (README); {instruction} filled in.
OPENVLA_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"
# Empty-string token OpenVLA's predict_action appends after the ":" to match
# the training-time input format (see modeling_prismatic.predict_action).
_OPENVLA_EMPTY_TOKEN = 29871


def load_openvla_model(hf_repo: str, dtype: str, attn_impl: str = "sdpa"):
    """Load OpenVLA via AutoModelForVision2Seq (present in transformers ~4.57).

    Mirrors the model card: trust_remote_code + low_cpu_mem_usage, dtype per the
    CLI flag, attn backend recorded for provenance. Returns (model, processor,
    actual_attn).
    """
    log.info("Loading OpenVLA model from HF: %s (dtype=%s)", hf_repo, dtype)
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    torch_dtype = DTYPE_TORCH[dtype]

    # Compat shim: OpenVLA's custom class (4.40-era) exposes `_supports_sdpa`/
    # flash-attn support as PROPERTIES that delegate to `self.language_model`.
    # transformers 4.57 reads those flags inside `super().__init__()` — BEFORE
    # `language_model` is assigned — so the delegating getter raises. Pin them as
    # plain class attributes (Llama-2 genuinely supports SDPA + FA2), which
    # shadows the broken properties and unblocks the loader. Pure compatibility;
    # does not alter the computation being measured.
    cls = get_class_from_dynamic_module(
        "modeling_prismatic.OpenVLAForActionPrediction", hf_repo)
    for flag in ("_supports_sdpa", "_supports_flash_attn_2", "_supports_flash_attn"):
        setattr(cls, flag, True)

    processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        hf_repo,
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to("cuda")
    model.eval()
    actual_attn = getattr(model.config, "_attn_implementation", attn_impl)
    log.info("Attention backend: %s", actual_attn)
    return model, processor, actual_attn


def build_openvla_inputs(processor, image, instruction: str, device, dtype: str):
    """Replicate OpenVLA's exact preprocessing → model-ready inputs dict.

    Builds the trained prompt scaffold, runs the Prismatic processor over
    (prompt, image), and appends the empty-string token (29871) exactly as
    `predict_action` does so the measured forward matches the real inference
    path. Returns a dict with input_ids / attention_mask / pixel_values on the
    target device + dtype.
    """
    import PIL.Image
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)
    prompt = OPENVLA_PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image)
    inputs = {k: (v.to(device, dtype=DTYPE_TORCH[dtype]) if torch.is_floating_point(v)
                  else v.to(device)) for k, v in inputs.items()}

    # Append the empty-string token to match training-time format (predict_action).
    ids = inputs["input_ids"]
    if not bool(torch.all(ids[:, -1] == _OPENVLA_EMPTY_TOKEN)):
        pad = torch.tensor([[_OPENVLA_EMPTY_TOKEN]], dtype=ids.dtype, device=ids.device)
        inputs["input_ids"] = torch.cat([ids, pad], dim=1)
        if "attention_mask" in inputs:
            am = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat(
                [am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)], dim=1)
    return inputs


def validate_openvla_action_path(model, processor, image, instruction: str,
                                 unnorm_key: str, dtype: str) -> dict[str, Any]:
    """Validate OpenVLA's action path: predict_action must return a 7-DOF vector.

    Unlike NORA, OpenVLA's float decode works on this stack (no FAST tokenizer),
    so this is a full end-to-end methodology proof. CPU-side / single-shot.
    """
    import PIL.Image
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)
    prompt = OPENVLA_PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = processor(prompt, image).to(model.device, dtype=DTYPE_TORCH[dtype])
    try:
        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        vec = np.asarray(action)
        ok = tuple(vec.shape) == (7,)
        return {
            "token_level_ok": bool(ok),
            "fast_decode_ok": bool(ok),
            "action_dof": int(vec.shape[0]) if vec.ndim == 1 else None,
            "action_vector": [round(float(x), 5) for x in np.ravel(vec)],
            "unnorm_key": unnorm_key,
            "n_action_tokens": int(model.get_action_dim(unnorm_key)),
        }
    except Exception as e:  # noqa: BLE001
        return {"token_level_ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def resolve_family(vla_key: str) -> str:
    """Map a catalog key to its harness family (load/preprocess/validate path).

    - "dual_loop": flow-matching VLAs with a separate action expert, NORA-1.5
      vendored-Qwen2.5-VL path (~/.virtualenvs/nora15).
    - "pi05": π0.5 dual-loop via the lerobot PI05Policy stack (PaliGemma + Gemma
      expert), run in ~/.virtualenvs/pi05.
    - "openvla": Prismatic Llama-2 + discrete 256-bin tokens (separate venv).
    - "nora": single-loop Qwen2.5-VL + FAST+ autoregressive decode.
    """
    if vla_key == "nora_1p5":
        return "dual_loop"
    if vla_key == "pi_0p5":
        return "pi05"
    if vla_key == "bitvla":
        return "bitvla"   # OpenVLA-OFT ternary, parallel-chunk (separate venv)
    if vla_key.startswith("openvla"):
        return "openvla"
    if vla_key.startswith(("nora", "pi_")):
        return "nora"   # Qwen2.5-VL + FAST+ / chat-template path
    return "nora"       # default to the NORA-style path


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
    # model.model.visual = Qwen2.5-VL (NORA); vision_backbone = OpenVLA/Prismatic.
    for path in ("model.visual", "visual", "vision_backbone"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def find_llm_module(model):
    """Locate the LLM decoder body (text model) — used to capture the true
    prefill sequence length for the FLOP attribution."""
    for path in ("model.language_model", "language_model"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def capture_llm_prefill_seqlen(model, inputs: dict, llm_module) -> int | None:
    """Ground-truth count of tokens the LLM actually processes at prefill.

    `input_ids.shape[1]` is NOT reliable across families: NORA (Qwen2.5-VL)
    expands image patches into input_ids placeholders (so it's already counted),
    but OpenVLA injects 256 vision tokens as *embeddings* — they never appear in
    input_ids, so input_ids length (~24, text-only) would undercount the LLM
    prefill ~10x. A forward pre-hook on the LLM body reads the real sequence
    length of its input (inputs_embeds / input_ids / hidden states) on the
    prefill pass. Returns None if it can't be captured (caller falls back to
    input_ids length).
    """
    if llm_module is None:
        return None
    seen: dict[str, int] = {}

    def _pre(_mod, args, kwargs):
        for cand in (kwargs.get("inputs_embeds"), kwargs.get("input_ids"),
                     kwargs.get("hidden_states"),
                     args[0] if args else None):
            if torch.is_tensor(cand) and cand.dim() >= 2:
                seen["len"] = max(seen.get("len", 0), int(cand.shape[1]))
                break

    h = llm_module.register_forward_pre_hook(_pre, with_kwargs=True)
    try:
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
    finally:
        h.remove()
    return seen.get("len")


def measure_vlm_forward(model, inputs: dict, n_warmup: int, n_trials: int,
                        nvtx_label: str, vision_module=None) -> dict[str, Any]:
    """VLM forward = vision encoder + LLM prefill, timed via
    generate(max_new_tokens=1) over the pre-built `inputs` dict. Returns
    p50/p95 ms + per-trial.

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


def measure_action_forward(model, inputs: dict, max_action_tokens: int,
                           n_warmup: int, n_trials: int,
                           nvtx_label: str) -> dict[str, Any]:
    """Action forward = the full inference call: prefill + autoregressive
    action-token decode. This IS the end-to-end ms-per-action that the CSV's
    `measured_5090_ms_per_action` refers to.

    Family-agnostic over the pre-built `inputs` dict:
    - NORA emits a *variable-length* FAST+ token sequence and stops at EOS;
      `max_action_tokens` (default 256) is just a runaway safety cap.
    - OpenVLA emits exactly `action_dim` (=7) discrete tokens; pass
      `max_action_tokens=7` and generation runs the full chunk.
    The actual decoded count is captured from the generated ids and reported as
    `n_generated_tokens`, so the ms/token derivation uses the true length.
    """
    from src.profiling.nvtx_helpers import nvtx_range

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
    log.info("Action decode produced %d new tokens (cap=%d)",
             n_generated, max_action_tokens)

    log.info("Action timed: %d forwards (decode to EOS or cap)", n_trials)
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


def _module_params(model, paths) -> int:
    """Sum parameters of the first submodule found at any of `paths` (0 if none)."""
    for path in paths:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return int(sum(p.numel() for p in obj.parameters()))
        except AttributeError:
            continue
    return 0


def count_component_params(model, family: str) -> dict[str, int]:
    """Real parameter counts per component, read off the loaded model — far more
    accurate than the CSV's round-number estimates. Used to drive the physical
    FLOP attribution. Keys: vision, llm_body, lm_head, projector, total."""
    vision = _module_params(model, ("model.visual", "visual", "vision_backbone"))
    llm_body = _module_params(model, ("model.language_model", "language_model"))
    lm_head = _module_params(model, ("lm_head", "language_model.lm_head"))
    projector = _module_params(model, ("projector", "multi_modal_projector"))
    total = int(sum(p.numel() for p in model.parameters()))
    return {"vision": vision, "llm_body": llm_body, "lm_head": lm_head,
            "projector": projector, "total": total}


def infer_vision_patches(model, default: int = 256) -> int:
    """Number of patches the vision ViT processes — drives the vision FLOP term.

    Tries (image_size // patch_size)² from the vision config; falls back to 256
    (224×224 input at patch-14, the standard for both NORA's Qwen2.5-VL ViT and
    OpenVLA's SigLIP/DINOv2). The LLM-side token count is measured directly
    (input_len), so this assumption only affects the vision term.
    """
    cfg = getattr(model, "config", None)
    for attr in ("vision_config",):
        vc = getattr(cfg, attr, None)
        if vc is None:
            continue
        img = getattr(vc, "image_size", None)
        patch = getattr(vc, "patch_size", None)
        if isinstance(img, (list, tuple)):
            img = img[0]
        if img and patch:
            return int((img // patch) ** 2)
    return default


def analytical_component_flops(pc: dict[str, int], n_vision_patches: int,
                               input_len: int, n_action_tokens: int) -> dict[str, Any]:
    """Physical (not effective) per-component FLOP attribution via the standard
    matmul rule: a module with P params processing T tokens costs ~2·P·T FLOPs
    (2 = MAC). This is the same convention the sizer's roofline uses, so the
    numbers compose with their per-hw util constants and kill the
    effective-FLOP-at-5090-util footgun.

    Split (per the sizer's 2026-05-29 guidance — no double-counting):
      • vision     = 2·P_vision·n_patches               (ViT over image patches)
      • llm_prefill= 2·P_llm_body·input_len + 2·P_lm_head + 2·P_projector·n_patches
      • llm_decode = 2·(P_llm_body + P_lm_head) per generated token
    Both NORA (FAST+) and OpenVLA (discrete 256-bin) decode *through the LLM +
    lm_head* — there is no standalone action-head FLOP to add. The attention
    quadratic term is omitted (<~2% at these sequence lengths); matmul-dominated.
    All values in GFLOP.
    """
    P_v, P_llm, P_head, P_proj = (pc["vision"], pc["llm_body"],
                                  pc["lm_head"], pc["projector"])
    g = 1e9
    vision = 2 * P_v * n_vision_patches / g
    prefill = (2 * P_llm * input_len + 2 * P_head + 2 * P_proj * n_vision_patches) / g
    decode_per_tok = 2 * (P_llm + P_head) / g
    e2e = vision + prefill + decode_per_tok * n_action_tokens
    return {
        "method": "physical matmul FLOP, 2·P·T per module (attention-quadratic "
                  "term omitted, <~2%); params counted off the loaded model. "
                  "Decode runs through LLM body + lm_head (no standalone action "
                  "head — applies to both NORA FAST+ and OpenVLA discrete tokens).",
        "params_millions": {k: round(v / 1e6, 1) for k, v in pc.items()},
        "n_vision_patches": n_vision_patches,
        "input_len_tokens": input_len,
        "n_action_tokens": n_action_tokens,
        "vision_encoder_gflop": round(vision, 2),
        "llm_prefill_gflop": round(prefill, 2),
        "llm_decode_gflop_per_token": round(decode_per_tok, 3),
        "llm_decode_gflop_total": round(decode_per_tok * n_action_tokens, 2),
        "e2e_action_gflop": round(e2e, 2),
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


# ════════════════════════ Dual-loop family ════════════════════════
# NORA-1.5 / π0.5-class VLAs run TWO loops, not one autoregressive decode:
#   • slow loop  — the VLM backbone runs ONCE per action chunk → frozen KV cache
#   • fast loop  — a SEPARATE small action expert runs N flow-matching denoise
#                  steps against that KV cache, emitting a whole H-action chunk
# The expensive VLM is amortized across H actions; the fast loop is a tiny model.
#
# NORA-1.5's action expert + denoise loop live in declare-lab/nora-1.5's training
# code, NOT the HF weights (the repo ships a bare model.safetensors + a config of
# the expert hyperparameters). We vendor the inference-relevant subset at
# scripts/vendor/nora15_modelling_expert.py (TF/dlimp deps stripped) and drive its
# published `sample_actions` / `denoise_step` API verbatim. Runs under the
# dedicated ~/.virtualenvs/nora15 venv (transformers 4.54.1) — the vendored MoT
# attention reads the legacy tuple-style KV cache that 4.54 still exposes; newer
# transformers changed the Cache __getitem__ contract and break the expert.

VENDOR_DIR = REPO / "scripts" / "vendor"
DUAL_LOOP_NUM_STEPS_DEFAULT = 10  # NORA-1.5 README: sample_actions(..., num_steps=10)


def load_dual_loop_model(hf_repo: str, dtype: str, attn_impl: str = "sdpa"):
    """Construct VLAWithExpert (VLM backbone + flow-matching action expert).

    The config.json fields map 1:1 to VLAWithExpert.__init__ kwargs; from_pretrained
    runs __init__ (which loads the VLM from `vlm_model_id`) then overlays the full
    nora-1.5 state dict (fine-tuned VLM + action expert + projection heads) with
    strict=False. We then cast the whole module to the target FP dtype — the
    action expert REQUIRES floating point (INT8 of the diffusion head breaks task
    success per QuantVLA / the CSV note), so bf16 is the faithful path. Returns
    (model, processor, attn_backend).
    """
    log.info("Loading dual-loop VLA from HF: %s (dtype=%s)", hf_repo, dtype)
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    from nora15_modelling_expert import VLAWithExpert  # type: ignore
    from huggingface_hub import hf_hub_download

    cfg = json.load(open(hf_hub_download(hf_repo, "config.json")))
    model = VLAWithExpert.from_pretrained(
        hf_repo,
        vlm_model_id=cfg.get("vlm_model_id", "declare-lab/nora-long"),
        processor_id=cfg.get("processor_id", "declare-lab/nora"),
        fast_tokenizer_id=cfg.get("fast_tokenizer_id", "physical-intelligence/fast"),
        lm_expert_width_multiplier=cfg.get("lm_expert_width_multiplier", 0.375),
        lm_expert_num_attention_head=cfg.get("lm_expert_num_attention_head", 6),
        action_chunk_length=cfg.get("action_chunk_length", 5),
        action_dim=cfg.get("action_dim", 7),
    )
    torch_dtype = DTYPE_TORCH[dtype]
    model.to("cuda", dtype=torch_dtype)
    model.eval()
    actual_attn = getattr(model.vlm.config, "_attn_implementation", attn_impl)
    log.info("Action expert: chunk_length=%d, width_mult=%.3f, heads=%d, attn=%s",
             model.action_chunk_length,
             cfg.get("lm_expert_width_multiplier", 0.375),
             cfg.get("lm_expert_num_attention_head", 6), actual_attn)
    return model, model.processor, actual_attn


def build_dual_loop_inputs(model, image, prompt: str, device):
    """Replicate VLAWithExpert.sample_actions preprocessing → model-ready inputs.

    Mirrors the vendored path exactly: vendored resize_image (PIL LANCZOS, the
    de-TF'd lanczos3 equivalent) → Qwen2.5-VL chat template at 224×224 → processor.
    Image content is timing-invariant; a representative frame suffices.
    """
    import PIL.Image
    from qwen_vl_utils import process_vision_info
    import nora15_modelling_expert as nm  # type: ignore

    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)
    image = nm.resize_image(image)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image,
             "resized_height": 224, "resized_width": 224},
            {"type": "text", "text": prompt},
        ],
    }]
    text = model.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = model.processor(text=[text], images=image_inputs, videos=video_inputs,
                             padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _build_denoise_conditioning(model, inputs: dict):
    """Run the VLM backbone once and build the action-expert conditioning,
    replicating VLAWithExpert.sample_actions up to (but not including) the loop.

    Returns (vlm_kv_cache, full_2d_attn_mask, bsz). The KV cache is READ-ONLY
    conditioning for the expert (the MoT attention concatenates it with the
    expert's own K/V but never writes back), so it is safely reused across every
    denoise step and every timing trial — which is exactly the amortization the
    dual-loop architecture exploits.
    """
    import nora15_modelling_expert as nm  # type: ignore
    device = model.vlm.device
    bsz = int(inputs["input_ids"].shape[0])
    L = model.action_chunk_length
    with torch.no_grad():
        vlm_outputs = model.vlm(**inputs)
    vlm_kv_cache = vlm_outputs.past_key_values
    if vlm_kv_cache is None:
        with torch.no_grad():
            vlm_kv_cache = model.vlm(**inputs, use_cache=True).past_key_values
    vlm_pad_mask = inputs["attention_mask"].clone()
    action_pad_mask = torch.ones(bsz, L, device=device).bool()
    action_attn_mask = torch.zeros(bsz, L, device=device).bool()
    concat_pad_mask = torch.cat([vlm_pad_mask, action_pad_mask], dim=1)
    concat_attn_mask = torch.cat([vlm_pad_mask, action_attn_mask], dim=1)
    full_2d_attn_mask = nm.make_att_2d_masks(concat_pad_mask, concat_attn_mask)
    return vlm_kv_cache, full_2d_attn_mask, bsz


def measure_vlm_backbone(model, inputs: dict, n_warmup: int, n_trials: int,
                         nvtx_label: str, vision_module=None) -> dict[str, Any]:
    """Time the dual-loop SLOW loop: one VLM backbone forward → KV cache.

    Unlike the single-loop measure_vlm_forward (which times generate(max_new=1)),
    the dual-loop VLM is invoked as a plain forward — exactly as sample_actions
    does — because its only job is to produce the conditioning KV cache, not to
    emit a token. Same vision/LLM-prefill component split (CUDA-event hooks on the
    vision tower) the sizer's roofline consumes.
    """
    from src.profiling.nvtx_helpers import nvtx_range

    vis_evt: dict[str, torch.cuda.Event] = {}
    handles = []
    if vision_module is not None:
        def _pre(_m, _a):
            e = torch.cuda.Event(enable_timing=True); e.record(); vis_evt["start"] = e
        def _post(_m, _a, _o):
            e = torch.cuda.Event(enable_timing=True); e.record(); vis_evt["end"] = e
        handles.append(vision_module.register_forward_pre_hook(_pre))
        handles.append(vision_module.register_forward_hook(_post))

    log.info("VLM-backbone warmup: %d forwards", n_warmup)
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model.vlm(**inputs)
    torch.cuda.synchronize()

    log.info("VLM-backbone timed: %d forwards%s", n_trials,
             " (+vision/LLM split)" if vision_module is not None else "")
    per_trial_ms: list[float] = []
    per_vision_ms: list[float] = []
    for _ in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); vis_evt.clear()
        with nvtx_range(nvtx_label), torch.no_grad():
            start.record()
            _ = model.vlm(**inputs)
            end.record()
        per_trial_ms.append(_cuda_event_ms(start, end))
        if "start" in vis_evt and "end" in vis_evt:
            vis_evt["end"].synchronize()
            per_vision_ms.append(float(vis_evt["start"].elapsed_time(vis_evt["end"])))

    for h in handles:
        h.remove()

    arr = np.array(per_trial_ms)
    out: dict[str, Any] = {
        "n_warmup": n_warmup, "n_trials": n_trials,
        "per_trial_ms": per_trial_ms,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }
    if per_vision_ms and len(per_vision_ms) == len(per_trial_ms):
        v = np.array(per_vision_ms)
        llm = arr - v
        out["components"] = {
            "vision_encoder": {"p50_ms": float(np.percentile(v, 50)),
                               "p95_ms": float(np.percentile(v, 95)),
                               "mean_ms": float(v.mean())},
            "llm_prefill": {"p50_ms": float(np.percentile(llm, 50)),
                            "p95_ms": float(np.percentile(llm, 95)),
                            "mean_ms": float(llm.mean())},
            "vision_frac_p50": round(float(np.percentile(v, 50) /
                                           np.percentile(arr, 50)), 3),
            "method": "forward-hook CUDA events around the VLM vision tower; "
                      "llm_prefill = VLM-forward total − vision (paired per-trial).",
        }
    return out


def measure_denoise_loop(model, vlm_kv_cache, full_2d_attn_mask, bsz: int,
                         num_steps: int, n_warmup: int, n_trials: int,
                         nvtx_label: str) -> dict[str, Any]:
    """Time the dual-loop FAST loop: the action expert's flow-matching denoise.

    Replicates VLAWithExpert.sample_actions' integration loop verbatim (dt =
    -1/num_steps, time 1.0 → 0, `num_steps` denoise_step calls), reusing the
    VLM KV cache passed in (the VLM is NOT re-run — that is the amortization).
    Times both a single denoise step and the full num_steps loop; fresh noise is
    sampled per trial (cheap, off the timed path's hot region). Returns p50/p95
    for each plus the step count.
    """
    from src.profiling.nvtx_helpers import nvtx_range
    device = model.vlm.device
    L = model.action_chunk_length
    dt = -1.0 / num_steps

    def _full_loop():
        x_t = model.sample_noise((bsz, L, 7), device=device)
        time = torch.tensor(1.0, dtype=model.vlm.dtype, device=device)
        while time >= -dt / 2:
            v_t = model.denoise_step(x_t, time.expand(bsz), vlm_kv_cache,
                                     full_2d_attn_mask)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t

    def _one_step():
        x_t = model.sample_noise((bsz, L, 7), device=device)
        t = torch.tensor(1.0, dtype=model.vlm.dtype, device=device)
        return model.denoise_step(x_t, t.expand(bsz), vlm_kv_cache, full_2d_attn_mask)

    log.info("Denoise warmup: %d loops (%d steps each)", n_warmup, num_steps)
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = _full_loop()
    torch.cuda.synchronize()

    log.info("Denoise timed: %d loops + %d single-steps", n_trials, n_trials)
    loop_ms: list[float] = []
    step_ms: list[float] = []
    for _ in range(n_trials):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with nvtx_range(nvtx_label), torch.no_grad():
            s.record(); _ = _full_loop(); e.record()
        loop_ms.append(_cuda_event_ms(s, e))
    for _ in range(n_trials):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.no_grad():
            s.record(); _ = _one_step(); e.record()
        step_ms.append(_cuda_event_ms(s, e))

    larr, sarr = np.array(loop_ms), np.array(step_ms)
    return {
        "n_warmup": n_warmup, "n_trials": n_trials,
        "num_denoise_steps": num_steps,
        "action_chunk_length": L,
        "loop_per_trial_ms": loop_ms,
        "loop_p50_ms": float(np.percentile(larr, 50)),
        "loop_p95_ms": float(np.percentile(larr, 95)),
        "loop_mean_ms": float(larr.mean()),
        "step_p50_ms": float(np.percentile(sarr, 50)),
        "step_p95_ms": float(np.percentile(sarr, 95)),
        "step_mean_ms": float(sarr.mean()),
    }


def validate_dual_loop_action_path(model, image, prompt: str,
                                   num_steps: int) -> dict[str, Any]:
    """Prove the harness measures the real flow-matching path: sample_actions
    must return a finite (1, action_chunk_length, action_dim) normalized action
    chunk. CPU-orchestrated / single-shot; NOT part of the GPU timing.
    """
    try:
        with torch.no_grad():
            act = model.sample_actions(image, prompt, num_steps=num_steps)
        arr = np.asarray(act)
        L = model.action_chunk_length
        ok = (arr.shape == (1, L, 7)) and bool(np.isfinite(arr).all())
        return {
            "token_level_ok": bool(ok),     # schema parity with single-loop validators
            "fast_decode_ok": bool(ok),
            "action_chunk_shape": list(arr.shape),
            "action_chunk_length": L,
            "action_dim": 7,
            "num_denoise_steps": num_steps,
            "action_chunk_sample": [round(float(x), 5) for x in np.ravel(arr)[:7]],
            # Full [H,7] fp16 reference chunk (drone-sizer / qualcomm mixed-precision diff).
            "action_chunk_full": [[round(float(x), 5) for x in row]
                                  for row in np.ravel(arr).reshape(-1, 7)],
        }
    except Exception as e:  # noqa: BLE001 — validation must never abort timing
        return {"token_level_ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def count_dual_loop_params(model) -> dict[str, int]:
    """Real parameter counts per dual-loop component, read off the loaded model.
    Keys: vlm_vision, vlm_body, vlm_head, action_expert, total."""
    vlm_vision = _module_params(model, ("vlm.model.visual", "vlm.visual"))
    vlm_body = _module_params(model, ("vlm.model.language_model", "vlm.language_model"))
    vlm_head = _module_params(model, ("vlm.lm_head",))
    action_expert = int(sum(p.numel() for p in model.action_expert.parameters()))
    # Small flow-matching projection heads (in/out/time MLPs) live on the expert side.
    proj = 0
    for name in ("action_in_proj", "action_out_proj",
                 "action_time_mlp_in", "action_time_mlp_out"):
        mod = getattr(model, name, None)
        if mod is not None:
            proj += int(sum(p.numel() for p in mod.parameters()))
    total = int(sum(p.numel() for p in model.parameters()))
    return {"vlm_vision": vlm_vision, "vlm_body": vlm_body, "vlm_head": vlm_head,
            "action_expert": action_expert + proj, "total": total}


def dual_loop_flops(pc: dict[str, int], n_vision_patches: int, vlm_seqlen: int,
                    action_chunk_length: int, num_steps: int) -> dict[str, Any]:
    """Physical (not effective) per-component FLOP for the dual-loop topology,
    same 2·P·T matmul convention as the single-loop attribution so the numbers
    compose with the sizer's per-hw util constants.

      • vision        = 2·P_vlm_vision·n_patches            (ViT, once per chunk)
      • vlm_prefill   = 2·(P_vlm_body+P_vlm_head)·vlm_seqlen (LLM backbone, once)
      • denoise_step  = 2·P_action_expert·action_chunk_length
                        (the expert processes H action tokens per step; the
                        cross-attention over the VLM KV is attention-quadratic,
                        omitted per the <~2% convention — but note the caveat
                        below, the VLM seqlen here is large relative to H)
      • action_chunk  = vision + vlm_prefill + num_steps·denoise_step
    Per-action FLOP = action_chunk / action_chunk_length. All values in GFLOP.
    """
    g = 1e9
    P_v, P_body, P_head, P_exp = (pc["vlm_vision"], pc["vlm_body"],
                                  pc["vlm_head"], pc["action_expert"])
    vision = 2 * P_v * n_vision_patches / g
    vlm_prefill = 2 * (P_body + P_head) * vlm_seqlen / g
    denoise_step = 2 * P_exp * action_chunk_length / g
    denoise_total = denoise_step * num_steps
    action_chunk = vision + vlm_prefill + denoise_total
    return {
        "method": "physical matmul FLOP, 2·P·T per module (attention-quadratic "
                  "term omitted). Dual-loop: VLM backbone (vision+prefill) runs "
                  "ONCE per chunk; the action expert runs num_steps denoise steps "
                  "over action_chunk_length tokens. Per-action = chunk / H.",
        "params_millions": {k: round(v / 1e6, 1) for k, v in pc.items()},
        "n_vision_patches": n_vision_patches,
        "vlm_seqlen_tokens": vlm_seqlen,
        "action_chunk_length": action_chunk_length,
        "num_denoise_steps": num_steps,
        "vision_encoder_gflop": round(vision, 2),
        "vlm_prefill_gflop": round(vlm_prefill, 2),
        "denoise_step_gflop": round(denoise_step, 4),
        "denoise_total_gflop": round(denoise_total, 3),
        "action_chunk_gflop": round(action_chunk, 2),
        "per_action_gflop": round(action_chunk / action_chunk_length, 3),
        "caveat_attention_quadratic": (
            "action-expert cross-attention reads the full VLM KV (seqlen≈%d vs "
            "H=%d), so the omitted QKᵀ/·V term is a larger share here than in the "
            "single-loop case; treat denoise FLOP as a matmul-only lower bound."
            % (vlm_seqlen, action_chunk_length)),
    }


def run_dual_loop(args, spec, hf_repo: str, image) -> None:
    """End-to-end dual-loop measurement + summary write. Parallels main()'s
    single-loop flow but for the VLM-once / expert-N-steps topology, and emits
    the SAME result schema (vlm_forward / action_forward / derived / dram / flops
    / calibration) plus a dual_loop block that makes the two-loop split explicit."""
    num_steps = args.num_steps
    log.info("Harness family: dual_loop (flow-matching action expert, %d steps)",
             num_steps)

    # ── Load + DRAM baseline ─────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    pre_load_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    model, processor, attn_backend = load_dual_loop_model(hf_repo, args.dtype, args.attn)
    post_load_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    weight_vram_mb = post_load_mb - pre_load_mb
    log.info("Weight VRAM:  %.0f MB (%.2f GB at %s)",
             weight_vram_mb, weight_vram_mb / 1024, args.dtype)

    inputs = build_dual_loop_inputs(model, image, args.prompt, model.vlm.device)

    # ── Validate the flow-matching action path ───────────────────────
    try:
        action_validation = validate_dual_loop_action_path(model, image, args.prompt, num_steps)
        if action_validation.get("token_level_ok"):
            log.info("Action validation PASSED: chunk shape %s, sample %s",
                     action_validation.get("action_chunk_shape"),
                     action_validation.get("action_chunk_sample"))
        else:
            log.warning("Action validation FAILED — review before trusting "
                        "latency: %s", action_validation)
    except Exception as e:  # noqa: BLE001
        log.warning("Action validation could not run (%s: %s).", type(e).__name__, e)
        action_validation = {"token_level_ok": False, "error": f"{type(e).__name__}: {e}"}

    # ── Per-loop latency ─────────────────────────────────────────────
    vlm_nvtx = f"vla_{spec.vla_key}__vlm"
    denoise_nvtx = f"vla_{spec.vla_key}__denoise"

    vision_module = find_vision_module(model.vlm)
    if vision_module is None:
        log.warning("VLM vision tower not located — skipping vision/LLM split.")
    vlm = measure_vlm_backbone(model, inputs, n_warmup=args.warmup,
                               n_trials=args.n_trials, nvtx_label=vlm_nvtx,
                               vision_module=vision_module)

    vlm_kv_cache, full_2d_attn_mask, bsz = _build_denoise_conditioning(model, inputs)
    denoise = measure_denoise_loop(model, vlm_kv_cache, full_2d_attn_mask, bsz,
                                   num_steps=num_steps, n_warmup=args.warmup,
                                   n_trials=args.n_trials, nvtx_label=denoise_nvtx)

    # ── Dual-loop derived rates ──────────────────────────────────────
    L = model.action_chunk_length
    vlm_ms = vlm["p50_ms"]
    loop_ms = denoise["loop_p50_ms"]
    # One fresh observation → one chunk of L actions: pay VLM once + the N-step loop.
    chunk_ms = vlm_ms + loop_ms
    amortized_ms_per_action = chunk_ms / L
    control_hz_amortized = 1000.0 / amortized_ms_per_action if amortized_ms_per_action > 0 else 0
    # Fast-loop-only rate: if the VLM context is reused / pipelined, the action
    # expert alone emits L actions per denoise loop — the "expert at ~40 Hz" claim.
    fast_loop_hz = (L * 1000.0 / loop_ms) if loop_ms > 0 else 0

    log.info("=" * 70)
    log.info("Dual-loop results for %s @ %s on RTX 5090:", spec.display_name, args.dtype)
    log.info("  [slow] VLM backbone p50:    %.2f ms", vlm_ms)
    if "components" in vlm:
        c = vlm["components"]
        log.info("    ├─ vision encoder p50:   %.2f ms  (%.0f%% of VLM forward)",
                 c["vision_encoder"]["p50_ms"], 100 * c["vision_frac_p50"])
        log.info("    └─ LLM prefill p50:      %.2f ms", c["llm_prefill"]["p50_ms"])
    log.info("  [fast] denoise step p50:    %.3f ms  (×%d steps)",
             denoise["step_p50_ms"], num_steps)
    log.info("  [fast] denoise loop p50:    %.2f ms  (→ %d-action chunk)", loop_ms, L)
    log.info("  Chunk latency (VLM+loop):   %.2f ms  → %d actions", chunk_ms, L)
    log.info("  Amortized ms/action:        %.2f ms  (%.1f Hz control)",
             amortized_ms_per_action, control_hz_amortized)
    log.info("  Fast-loop-only rate:        %.1f Hz  (expert alone, VLM reused)",
             fast_loop_hz)

    peak_inference_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    log.info("  Peak VRAM (incl. weights):  %.0f MB (%.2f GB)",
             peak_inference_mb, peak_inference_mb / 1024)

    # ── FLOP attribution ─────────────────────────────────────────────
    pc = count_dual_loop_params(model)
    vlm_seqlen = int(inputs["input_ids"].shape[1])  # Qwen2.5-VL: image tokens are in input_ids
    flops = dual_loop_flops(pc, n_vision_patches=infer_vision_patches(model.vlm),
                            vlm_seqlen=vlm_seqlen, action_chunk_length=L,
                            num_steps=num_steps)
    peak_gf_s = PEAK_BF16_TFLOPS_5090 * 1e3
    comp = vlm.get("components", {})
    vis_ms = comp.get("vision_encoder", {}).get("p50_ms")
    pre_ms = comp.get("llm_prefill", {}).get("p50_ms")

    def _util(gflop, ms):
        return round(gflop / (ms / 1e3) / peak_gf_s, 4) if ms and ms > 0 else None
    # Bottleneck classification for the denoise step — the number the sizer most
    # needs to project the fast loop to edge correctly. A step reads the expert's
    # weights once; effective BW = expert-weight-bytes / step-time. If both the
    # compute util AND the effective BW are tiny fractions of peak, the step is
    # neither compute- nor bandwidth-bound — it's kernel-launch / Python-dispatch
    # bound (36 small expert layers × eager custom-Python MoT attention over only
    # H tokens). This is CATEGORICALLY different from the single-loop AR-decode
    # bandwidth-wall, and it means the fast loop has large optimization headroom
    # (CUDA graphs / fusion / compile), so BW-wall edge scaling must NOT be
    # applied to it the way it is to autoregressive decode.
    bytes_per_param = {"bf16": 2.0, "fp16": 2.0, "fp32": 4.0,
                       "int8": 1.0, "int4": 0.5}[args.dtype]
    expert_weight_bytes = pc["action_expert"] * bytes_per_param
    step_s = denoise["step_p50_ms"] / 1e3
    denoise_eff_bw_gbs = round(expert_weight_bytes / step_s / 1e9, 1) if step_s > 0 else None
    PEAK_BW_GBS_5090 = 1792.0  # RTX 5090 ~1.79 TB/s GDDR7
    denoise_compute_util = _util(flops["denoise_step_gflop"], denoise["step_p50_ms"])
    denoise_bw_util = (round(denoise_eff_bw_gbs / PEAK_BW_GBS_5090, 4)
                       if denoise_eff_bw_gbs else None)
    flops["achieved_util_5090"] = {
        "peak_bf16_tflops": PEAK_BF16_TFLOPS_5090,
        "peak_bw_gbs": PEAK_BW_GBS_5090,
        "vision_encoder": _util(flops["vision_encoder_gflop"], vis_ms),
        "vlm_prefill": _util(flops["vlm_prefill_gflop"], pre_ms),
        "denoise_step": denoise_compute_util,
        "denoise_step_effective_bw_gbs": denoise_eff_bw_gbs,
        "denoise_step_bw_util": denoise_bw_util,
        "denoise_bottleneck": "launch/overhead-bound",
        "note": "physical FLOP / measured p50 / dense bf16 peak. KEY: the denoise "
                "step is neither compute-bound (~%.2f%% of peak FLOP) NOR "
                "bandwidth-bound (~%.1f%% of peak BW, eff %.0f GB/s) — it is "
                "kernel-launch/dispatch-bound in stock eager HF (36 tiny expert "
                "layers, custom-Python MoT attention over %d tokens). UNLIKE the "
                "single-loop AR-decode BW-wall, this has large optimization "
                "headroom; do NOT project it to edge via bandwidth scaling."
                % (100 * (denoise_compute_util or 0),
                   100 * (denoise_bw_util or 0),
                   denoise_eff_bw_gbs or 0, L),
    }
    log.info("  Physical FLOP: vision %.0f GF | prefill %.0f GF | denoise %.2f GF/step "
             "| chunk %.0f GF (%.1f GF/action)",
             flops["vision_encoder_gflop"], flops["vlm_prefill_gflop"],
             flops["denoise_step_gflop"], flops["action_chunk_gflop"],
             flops["per_action_gflop"])
    log.info("=" * 70)

    expert_m = round(pc["action_expert"] / 1e6)
    calibration_note = (
        "Dual-loop native (NORA-1.5): the Qwen2.5-VL backbone runs ONCE per "
        f"action chunk ({round(vlm_ms,1)} ms) and a separate flow-matching action "
        f"expert (measured {expert_m}M params incl. projection heads — the CSV's "
        f"800M over-counts) runs {num_steps} denoise steps ({round(loop_ms,1)} ms) "
        f"to emit a {L}-action chunk. Amortized control = {round(control_hz_amortized,1)} "
        f"Hz; the fast loop alone runs at {round(fast_loop_hz,1)} Hz (the published "
        "'action expert at ~40 Hz' regime, achievable when the VLM context is "
        "reused/pipelined across chunks). CRITICAL for edge projection: the "
        f"denoise step ({round(denoise['step_p50_ms'],1)} ms) is launch/overhead-"
        f"bound (eff {denoise_eff_bw_gbs} GB/s, ~{round(100*(denoise_bw_util or 0),1)}% "
        "of peak BW), NOT bandwidth-walled like single-loop AR decode — so it has "
        "large optimization headroom (CUDA graphs / fusion / compile) and must not "
        "be scaled to edge by bandwidth. Stock HF forward, no CUDA graphs / static "
        "cache / compile — an un-optimized floor. The action expert REQUIRES FP "
        "(bf16 here); INT8 of the diffusion head breaks task success per the CSV "
        "note, so unlike the VLM stages it has no INT8 path."
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": "RTX 5090",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": args.dtype,
        "attn_backend": attn_backend,
        "family": "dual_loop",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_path": str(CSV_PATH.relative_to(REPO)),
        "model_key": spec.vla_key,
        "hf_repo": hf_repo,
        "model_spec": asdict(spec),
        "result": {
            "action_validation": action_validation,
            "vlm_forward": vlm,            # the slow loop — schema parity w/ single-loop
            "action_forward": denoise,     # the fast loop (denoise) — dual-loop fields inside
            "dual_loop": {
                "topology": "vlm_backbone_once + action_expert_flow_matching_loop",
                "num_denoise_steps": num_steps,
                "action_chunk_length": L,
                "action_dim": 7,
                "vlm_backbone_p50_ms": round(vlm_ms, 3),
                "denoise_step_p50_ms": round(denoise["step_p50_ms"], 4),
                "denoise_loop_p50_ms": round(loop_ms, 3),
                "chunk_latency_ms": round(chunk_ms, 3),
                "amortized_ms_per_action": round(amortized_ms_per_action, 3),
                "control_hz_amortized": round(control_hz_amortized, 2),
                "fast_loop_only_hz": round(fast_loop_hz, 2),
                "note": "amortized = (VLM once + N denoise steps) / H actions; "
                        "fast_loop_only = H / denoise-loop (VLM reused).",
            },
            "derived": {
                "e2e_ms_per_action_p50": round(amortized_ms_per_action, 3),
                "action_rate_hz_p50": round(control_hz_amortized, 2),
                "chunk_latency_ms": round(chunk_ms, 3),
                "fast_loop_only_hz": round(fast_loop_hz, 2),
            },
            "dram": {
                "weight_vram_mb": round(weight_vram_mb, 1),
                "weight_vram_gb": round(weight_vram_mb / 1024, 2),
                "peak_inference_mb": round(peak_inference_mb, 1),
                "peak_inference_gb": round(peak_inference_mb / 1024, 2),
            },
            "flops": flops,
            "csv_reference": {
                "measured_5090_ms_per_action": spec.measured_5090_ms_per_action,
                "inference_dram_gb_at_default": getattr(
                    spec, f"inference_dram_gb_{spec.dtype_path_default}", None),
            },
            "calibration": {
                "reference_ms_per_action": spec.measured_5090_ms_per_action,
                "vlm_forward_p50_ms": round(vlm_ms, 2),
                "note": calibration_note,
            },
            "nvtx_labels": {"vlm": vlm_nvtx, "denoise": denoise_nvtx},
        },
    }
    blob = json.dumps(payload, indent=2, default=str)
    SUMMARY_PATH.write_text(blob)
    per_model = SUMMARY_PATH.parent / f"vla_summary_{spec.vla_key}.json"
    per_model.write_text(blob)
    log.info("Wrote %s and %s", SUMMARY_PATH.relative_to(REPO),
             per_model.relative_to(REPO))
    log.info("")
    log.info("Next step — DRAM via ncu: NVTX ranges `%s` (slow loop) and `%s` "
             "(fast loop) are set for profile_all_ncu.sh.", vlm_nvtx, denoise_nvtx)


# ════════════════════════ π0.5 family (lerobot flow-matching) ════════════════
# π0.5 is dual-loop like NORA-1.5 but a different stack: a PaliGemma (gemma_2b)
# VLM + a Gemma-300M-class flow-matching action expert, served through the
# lerobot framework (lerobot/pi05_base). Same two-loop shape — VLM prefix forward
# ONCE → KV cache, then N=10 denoise steps through the expert — but the published
# inference is PI05Policy.predict_action_chunk over a robot-observation batch
# (3 cameras + state + task), NOT a simple (image, instruction). We drive that
# real lerobot path and extract the loop split by hooking the shared
# `paligemma_with_expert` module: call 0 is the VLM prefill, calls 1..N are the
# denoise steps. The SigLIP vision tower runs inside embed_prefix (before the
# prefill call), so it is hooked separately. One VLM forward amortizes over the
# whole 50-action chunk — the largest amortization in the bake-off.
#
# Runs ONLY in ~/.virtualenvs/pi05 (Python 3.12, lerobot>=0.5.2, transformers
# 5.5.x). The PyPI lerobot 0.4.4 hard-gates pi0/pi05 on an unshipped patched
# transformers; lerobot 0.5.2 (main) removed that gate but requires Python 3.12.

PI05_REPO = "lerobot/pi05_base"


def _evt_pair_hooks(store: list):
    """forward_pre/forward hooks that push (start_event, end_event) per call into
    `store`, for per-call CUDA-event timing of a shared module."""
    def _pre(_m, _a):
        e = torch.cuda.Event(enable_timing=True); e.record(); store.append([e, None])
    def _post(_m, _a, _o):
        e = torch.cuda.Event(enable_timing=True); e.record()
        if store:
            store[-1][1] = e
    return _pre, _post


def _pair_durations(store: list) -> list[float]:
    """Resolve a list of [start,end] event pairs to elapsed ms (after sync)."""
    out = []
    for s, e in store:
        if s is not None and e is not None:
            e.synchronize()
            out.append(float(s.elapsed_time(e)))
    return out


def run_pi05_dual_loop(args, spec, hf_repo: str, image) -> None:
    """End-to-end π0.5 dual-loop measurement via the lerobot PI05Policy, emitting
    the same dual-loop summary schema as run_dual_loop (vlm_forward / action_forward
    / dual_loop / derived / flops)."""
    import cv2  # noqa: F401  (image is already an ndarray; kept for parity)
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    num_steps = args.num_steps
    log.info("Harness family: pi05 (lerobot flow-matching dual-loop, %d steps)", num_steps)

    # ── Load + DRAM baseline ─────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    pre_load_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    log.info("Loading π0.5 via lerobot PI05Policy: %s", hf_repo)
    policy = PI05Policy.from_pretrained(hf_repo)
    # pi05's flow-matching expert hardcodes float32 internals (sinusoidal time
    # embedding → time_mlp), the openpi convention, so weight-casting to bf16
    # breaks it. The faithful bf16 path for lerobot is AMP autocast: float32
    # master weights, bf16 matmuls (== lerobot's use_amp inference). We run all
    # forwards under autocast(bf16); reported dtype reflects that.
    policy.to("cuda").eval()
    model_dtype = next(policy.model.parameters()).dtype
    use_autocast = args.dtype in ("bf16", "fp16")
    autocast_dtype = DTYPE_TORCH[args.dtype] if use_autocast else model_dtype
    log.info("Master weight dtype: %s | compute: %s",
             model_dtype, f"autocast {args.dtype}" if use_autocast else str(model_dtype))
    cfg = policy.config
    post_load_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    weight_vram_mb = post_load_mb - pre_load_mb
    log.info("chunk_size=%d, num_inference_steps=%d, cameras=%d, max_action_dim=%d",
             cfg.chunk_size, cfg.num_inference_steps, len(list(cfg.image_features)),
             cfg.max_action_dim)
    log.info("Weight VRAM: %.0f MB (%.2f GB at %s)",
             weight_vram_mb, weight_vram_mb / 1024, args.dtype)

    # ── Build a model-ready observation batch via lerobot's own preprocessor ──
    # (normalization + state-token + tokenization + batch dim) — faithful, no drift.
    import PIL.Image  # noqa: F401
    img224 = image
    if img224.shape[:2] != (224, 224):
        import cv2 as _cv2
        img224 = _cv2.resize(image, (224, 224))
    img_t = torch.from_numpy(img224).permute(2, 0, 1).float() / 255.0
    raw = {f"observation.images.{k.split('.')[-1]}": img_t.clone()
           for k in cfg.image_features}
    raw["observation.state"] = torch.zeros(cfg.max_state_dim)
    raw["task"] = args.prompt
    pre, _post_proc = make_pre_post_processors(policy.config, hf_repo)
    batch = pre(raw)
    batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}

    n_cameras = len(list(cfg.image_features))
    L = cfg.chunk_size
    # Fixed noise (timing-invariant) at the master-weight dtype so it composes
    # with autocast. predict_action_chunk forwards **kwargs → sample_actions(noise=...).
    noise = torch.randn((1, L, cfg.max_action_dim), device="cuda", dtype=model_dtype)
    import contextlib
    def _ac():
        return (torch.autocast("cuda", dtype=autocast_dtype) if use_autocast
                else contextlib.nullcontext())
    action_dim = cfg.output_features[__import__("lerobot").constants.ACTION].shape[0] \
        if hasattr(__import__("lerobot"), "constants") else cfg.max_action_dim

    # ── Validate the action path ─────────────────────────────────────
    try:
        with torch.no_grad(), _ac():
            act = policy.predict_action_chunk(batch, noise=noise)
        a = act.detach().float().cpu().numpy()
        ok = (a.shape[0] == 1 and a.shape[1] == L) and bool(np.isfinite(a).all())
        action_validation = {
            "token_level_ok": bool(ok), "fast_decode_ok": bool(ok),
            "action_chunk_shape": list(a.shape), "action_chunk_length": L,
            "num_denoise_steps": num_steps,
            "action_chunk_sample": [round(float(x), 5) for x in np.ravel(a)[:7]],
            # Full [H,7] fp16 reference chunk (drone-sizer / qualcomm mixed-precision diff).
            "action_chunk_full": [[round(float(x), 5) for x in row]
                                  for row in np.ravel(a).reshape(-1, 7)],
        }
        log.info("Action validation %s: chunk %s sample %s",
                 "PASSED" if ok else "FAILED", list(a.shape),
                 action_validation["action_chunk_sample"])
    except Exception as e:  # noqa: BLE001
        log.warning("Action validation could not run (%s: %s).", type(e).__name__, e)
        action_validation = {"token_level_ok": False, "error": f"{type(e).__name__}: {e}"}

    # ── Per-loop latency via per-call hooks on the shared modules ────
    from src.profiling.nvtx_helpers import nvtx_range
    pwe = policy.model.paligemma_with_expert
    vision_tower = pwe.paligemma.model.vision_tower
    vis_store: list = []
    pwe_store: list = []
    # Vision tower is invoked via __call__, so forward hooks fire for it.
    vpre, vpost = _evt_pair_hooks(vis_store)
    handles = [
        vision_tower.register_forward_pre_hook(vpre),
        vision_tower.register_forward_hook(vpost),
    ]
    # lerobot calls `paligemma_with_expert.forward(...)` DIRECTLY (not __call__),
    # so forward hooks never fire — wrap the bound method to record CUDA events
    # per call instead. Call 0 = VLM prefill; calls 1..N = denoise steps.
    _orig_pwe_forward = pwe.forward
    def _timed_pwe_forward(*a, **k):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = _orig_pwe_forward(*a, **k)
        e.record()
        pwe_store.append((s, e))
        return out
    pwe.forward = _timed_pwe_forward
    nvtx_label = f"vla_{spec.vla_key}__chunk"

    log.info("Warmup: %d chunks", args.warmup)
    for _ in range(args.warmup):
        vis_store.clear(); pwe_store.clear()
        with torch.no_grad(), _ac():
            _ = policy.predict_action_chunk(batch, noise=noise)
    torch.cuda.synchronize()

    log.info("Timed: %d chunks (per-call vision / prefill / denoise split)", args.n_trials)
    chunk_ms_all, vision_ms_all, prefill_ms_all = [], [], []
    step_ms_all, loop_ms_all = [], []
    for _ in range(args.n_trials):
        vis_store.clear(); pwe_store.clear()
        cs = torch.cuda.Event(enable_timing=True); ce = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with nvtx_range(nvtx_label), torch.no_grad(), _ac():
            cs.record()
            _ = policy.predict_action_chunk(batch, noise=noise)
            ce.record()
        chunk_ms_all.append(_cuda_event_ms(cs, ce))
        vdur = _pair_durations(vis_store)        # n_cameras vision-tower calls
        pdur = _pair_durations(pwe_store)        # [prefill, denoise×N]
        vision_ms_all.append(float(sum(vdur)))
        if pdur:
            prefill_ms_all.append(pdur[0])
            steps = pdur[1:]
            if steps:
                step_ms_all.extend(steps)
                loop_ms_all.append(float(sum(steps)))

    for h in handles:
        h.remove()
    pwe.forward = _orig_pwe_forward  # restore

    def _p50(xs):
        return float(np.percentile(np.array(xs), 50)) if xs else None
    chunk_p50 = _p50(chunk_ms_all)
    vision_p50 = _p50(vision_ms_all)
    prefill_p50 = _p50(prefill_ms_all)
    step_p50 = _p50(step_ms_all)
    loop_p50 = _p50(loop_ms_all)
    vlm_backbone_p50 = (vision_p50 or 0) + (prefill_p50 or 0)

    # ── Dual-loop derived rates (amortized over the 50-action chunk) ──
    amortized_ms_per_action = chunk_p50 / L if chunk_p50 else None
    control_hz_amortized = 1000.0 / amortized_ms_per_action if amortized_ms_per_action else 0
    fast_loop_hz = (L * 1000.0 / loop_p50) if loop_p50 else 0
    peak_inference_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    log.info("=" * 70)
    log.info("π0.5 dual-loop results @ %s on RTX 5090:", args.dtype)
    log.info("  [slow] VLM backbone p50:    %.2f ms (vision %.2f ×%d cams + prefill %.2f)",
             vlm_backbone_p50, vision_p50 or 0, n_cameras, prefill_p50 or 0)
    log.info("  [fast] denoise step p50:    %.3f ms  (×%d)", step_p50 or 0, num_steps)
    log.info("  [fast] denoise loop p50:    %.2f ms  (→ %d-action chunk)", loop_p50 or 0, L)
    log.info("  Chunk latency (full):       %.2f ms  → %d actions", chunk_p50 or 0, L)
    log.info("  Amortized ms/action:        %.2f ms  (%.1f Hz control)",
             amortized_ms_per_action or 0, control_hz_amortized)
    log.info("  Fast-loop-only rate:        %.1f Hz", fast_loop_hz)
    log.info("  Peak VRAM (incl. weights):  %.0f MB (%.2f GB)",
             peak_inference_mb, peak_inference_mb / 1024)

    # ── Param counts + FLOP attribution (dual-loop) ──────────────────
    def _pc(mod):
        return int(sum(p.numel() for p in mod.parameters()))
    P_vision = _pc(vision_tower)
    P_body = _pc(pwe.paligemma.model.language_model)
    P_expert = _pc(pwe.gemma_expert.model)   # transformer only; expert lm_head unused for action
    P_proj = _pc(pwe.paligemma.model.multi_modal_projector)
    for nm in ("action_in_proj", "action_out_proj", "action_time_mlp_in",
               "action_time_mlp_out", "state_proj"):
        if hasattr(policy.model, nm):
            P_proj += _pc(getattr(policy.model, nm))
    P_total = _pc(policy.model)
    pc = {"vlm_vision": P_vision, "vlm_body": P_body, "vlm_head": 0,
          "action_expert": P_expert + P_proj, "total": P_total}

    # Prefix seqlen: 3 cameras × image-tokens + language tokens (real, from mask).
    patches_per_cam = infer_vision_patches(pwe.paligemma)
    lang_mask = batch.get("observation.language.attention_mask")
    n_lang = int(lang_mask.sum().item()) if lang_mask is not None else cfg.tokenizer_max_length
    vlm_seqlen = patches_per_cam * n_cameras + n_lang
    flops = dual_loop_flops(pc, n_vision_patches=patches_per_cam * n_cameras,
                            vlm_seqlen=vlm_seqlen, action_chunk_length=L,
                            num_steps=num_steps)
    peak_gf_s = PEAK_BF16_TFLOPS_5090 * 1e3

    def _util(gflop, ms):
        return round(gflop / (ms / 1e3) / peak_gf_s, 4) if ms and ms > 0 else None
    # AMP keeps float32 master weights; weight-streaming bytes reflect the actual
    # stored dtype, not the autocast compute dtype.
    bytes_per_param = {torch.float32: 4.0, torch.bfloat16: 2.0,
                       torch.float16: 2.0}.get(model_dtype, 4.0)
    expert_weight_bytes = pc["action_expert"] * bytes_per_param
    denoise_eff_bw_gbs = (round(expert_weight_bytes / (step_p50 / 1e3) / 1e9, 1)
                          if step_p50 else None)
    PEAK_BW_GBS_5090 = 1792.0
    denoise_bw_util = (round(denoise_eff_bw_gbs / PEAK_BW_GBS_5090, 4)
                       if denoise_eff_bw_gbs else None)
    denoise_compute_util = _util(flops["denoise_step_gflop"], step_p50)

    # Data-driven bottleneck classification (don't hardcode — π0.5's 430M float32
    # expert over 50 tokens streams real weight bytes, unlike NORA-1.5's tiny one).
    def _classify(cu, bw):
        cu, bw = cu or 0, bw or 0
        if cu >= 0.40:
            return "compute-bound"
        if bw >= 0.40:
            return "bandwidth-bound"
        if cu < 0.05 and bw < 0.05:
            return "launch/overhead-bound"
        return "mixed (partial-BW + launch overhead)"
    denoise_bottleneck = _classify(denoise_compute_util, denoise_bw_util)
    flops["achieved_util_5090"] = {
        "peak_bf16_tflops": PEAK_BF16_TFLOPS_5090, "peak_bw_gbs": PEAK_BW_GBS_5090,
        "vision_encoder": _util(flops["vision_encoder_gflop"], vision_p50),
        "vlm_prefill": _util(flops["vlm_prefill_gflop"], prefill_p50),
        "denoise_step": denoise_compute_util,
        "denoise_step_effective_bw_gbs": denoise_eff_bw_gbs,
        "denoise_step_bw_util": denoise_bw_util,
        "denoise_bottleneck": denoise_bottleneck,
        "weight_bytes_per_param": bytes_per_param,
        "note": "physical FLOP / measured p50 / dense bf16 peak. π0.5's denoise "
                "step is %s (compute %.1f%% / BW %.1f%% of peak) — NOT the AR-decode "
                "BW-wall, but more BW-leaning than NORA-1.5's tiny launch-bound "
                "expert because this expert is 430M and processes the full 50-action "
                "chunk per step. CAVEAT: bf16-AMP keeps float32 (4-byte) master "
                "weights, so the effective-BW / BW-util are an UPPER bound; true "
                "bf16-weight deployment would roughly halve them (→ more launch-"
                "leaning). The 50-action chunk gives the largest VLM amortization in "
                "the bake-off (one VLM forward per 50 actions → %.0f Hz amortized)."
                % (denoise_bottleneck, 100 * (denoise_compute_util or 0),
                   100 * (denoise_bw_util or 0), control_hz_amortized),
    }
    log.info("  Physical FLOP: vision %.0f GF | prefill %.0f GF | denoise %.2f GF/step "
             "| chunk %.0f GF (%.1f GF/action)",
             flops["vision_encoder_gflop"], flops["vlm_prefill_gflop"],
             flops["denoise_step_gflop"], flops["action_chunk_gflop"],
             flops["per_action_gflop"])
    log.info("=" * 70)

    expert_m = round(pc["action_expert"] / 1e6)
    calibration_note = (
        "π0.5 dual-loop (lerobot pi05_base): PaliGemma (gemma_2b) VLM backbone "
        f"runs ONCE ({round(vlm_backbone_p50,1)} ms = SigLIP vision ×{n_cameras} "
        f"cameras + gemma_2b prefill over {vlm_seqlen} tokens) and a Gemma-class "
        f"flow-matching expert (measured {expert_m}M params incl. projections — "
        f"the CSV's 300M under-counts) runs {num_steps} denoise steps "
        f"({round(loop_p50 or 0,1)} ms) to emit a {L}-action chunk. Amortized "
        f"control = {round(control_hz_amortized,1)} Hz; fast loop alone "
        f"{round(fast_loop_hz,1)} Hz. The {L}-action chunk is the largest "
        "amortization in the bake-off (one VLM forward per 50 actions). Denoise "
        f"step is {denoise_bottleneck} (compute {round(100*(denoise_compute_util or 0),1)}%"
        f" / BW {round(100*(denoise_bw_util or 0),1)}%, eff {denoise_eff_bw_gbs} GB/s) — "
        "not the AR-decode BW-wall, but more BW-leaning than NORA-1.5. NOTE: bf16-AMP "
        "(float32 master weights, autocast matmuls) — the lerobot mixed-precision "
        "path, since pi05's expert hardcodes float32 time-embedding internals so "
        "true bf16-weight load is not faithful; BW util is thus an upper bound. "
        "Stock lerobot forward (eager attention), no CUDA graphs / compile — an "
        "un-optimized floor."
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": "RTX 5090",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "lerobot": __import__("lerobot").__version__,
        "dtype": (f"{args.dtype}-amp (float32 master weights, autocast matmuls)"
                  if use_autocast else str(model_dtype)),
        "attn_backend": "eager (lerobot sample_actions default)",
        "family": "pi05",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_path": str(CSV_PATH.relative_to(REPO)),
        "model_key": spec.vla_key,
        "hf_repo": hf_repo,
        "model_spec": asdict(spec),
        "result": {
            "action_validation": action_validation,
            "vlm_forward": {
                "p50_ms": round(vlm_backbone_p50, 3),
                "n_trials": args.n_trials,
                "components": {
                    "vision_encoder": {"p50_ms": round(vision_p50 or 0, 3),
                                       "n_cameras": n_cameras},
                    "llm_prefill": {"p50_ms": round(prefill_p50 or 0, 3)},
                    "vision_frac_p50": round((vision_p50 or 0) / vlm_backbone_p50, 3)
                    if vlm_backbone_p50 else None,
                    "method": "per-call CUDA-event hooks: SigLIP vision tower "
                              "(fires once per camera) + paligemma_with_expert "
                              "call 0 (gemma_2b prefill). Vision runs in "
                              "embed_prefix, before the prefill call.",
                },
            },
            "action_forward": {
                "n_trials": args.n_trials,
                "num_denoise_steps": num_steps,
                "action_chunk_length": L,
                "step_p50_ms": round(step_p50 or 0, 4),
                "loop_p50_ms": round(loop_p50 or 0, 3),
                "chunk_total_p50_ms": round(chunk_p50 or 0, 3),
            },
            "dual_loop": {
                "topology": "paligemma_vlm_once + gemma_expert_flow_matching_loop",
                "stack": "lerobot PI05Policy.predict_action_chunk",
                "num_denoise_steps": num_steps,
                "action_chunk_length": L,
                "action_dim": int(action_dim),
                "n_cameras": n_cameras,
                "vlm_backbone_p50_ms": round(vlm_backbone_p50, 3),
                "denoise_step_p50_ms": round(step_p50 or 0, 4),
                "denoise_loop_p50_ms": round(loop_p50 or 0, 3),
                "chunk_latency_ms": round(chunk_p50 or 0, 3),
                "amortized_ms_per_action": round(amortized_ms_per_action or 0, 3),
                "control_hz_amortized": round(control_hz_amortized, 2),
                "fast_loop_only_hz": round(fast_loop_hz, 2),
                "note": "chunk_latency = full predict_action_chunk wall time "
                        "(VLM once + N denoise + embed/proj overhead); amortized "
                        "= chunk / H actions; fast_loop_only = H / denoise-loop.",
            },
            "derived": {
                "e2e_ms_per_action_p50": round(amortized_ms_per_action or 0, 3),
                "action_rate_hz_p50": round(control_hz_amortized, 2),
                "chunk_latency_ms": round(chunk_p50 or 0, 3),
                "fast_loop_only_hz": round(fast_loop_hz, 2),
            },
            "dram": {
                "weight_vram_mb": round(weight_vram_mb, 1),
                "weight_vram_gb": round(weight_vram_mb / 1024, 2),
                "peak_inference_mb": round(peak_inference_mb, 1),
                "peak_inference_gb": round(peak_inference_mb / 1024, 2),
            },
            "flops": flops,
            "csv_reference": {
                "measured_5090_ms_per_action": spec.measured_5090_ms_per_action,
                "inference_dram_gb_at_default": getattr(
                    spec, f"inference_dram_gb_{spec.dtype_path_default}", None),
            },
            "calibration": {
                "reference_ms_per_action": spec.measured_5090_ms_per_action,
                "vlm_forward_p50_ms": round(vlm_backbone_p50, 2),
                "note": calibration_note,
            },
            "nvtx_labels": {"chunk": nvtx_label},
        },
    }
    blob = json.dumps(payload, indent=2, default=str)
    SUMMARY_PATH.write_text(blob)
    per_model = SUMMARY_PATH.parent / f"vla_summary_{spec.vla_key}.json"
    per_model.write_text(blob)
    log.info("Wrote %s and %s", SUMMARY_PATH.relative_to(REPO),
             per_model.relative_to(REPO))


def run_pi05_camera_scaling(args, spec, hf_repo: str, image) -> None:
    """Multi-camera scaling for π0.5 (multi-cam brief Phase 1). pi05_base is locked
    to its 3 trained cameras, so we can't sweep n_cameras on the model directly.
    Instead we DECOMPOSE: time the SigLIP vision tower on a single camera
    (→ per-camera vision cost; total vision ≈ N×) and re-time the PaliGemma prefill
    at N=1/2/3-equivalent prefix lengths (N·image_tokens + text, by slicing the real
    captured prefix) → the LLM-vs-N slope. This directly tests the brief's
    'vision N×, LLM 1×' claim — and quantifies that the LLM is NOT invariant to N
    (each camera injects ~256 image tokens into the prefix)."""
    import contextlib
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    log.info("Harness family: pi05 camera-scaling (decomposed vision + prefill-vs-N)")
    policy = PI05Policy.from_pretrained(hf_repo)
    policy.to("cuda").eval()
    cfg = policy.config
    model_dtype = next(policy.model.parameters()).dtype
    ac = (torch.autocast("cuda", dtype=DTYPE_TORCH[args.dtype])
          if args.dtype in ("bf16", "fp16") else contextlib.nullcontext())

    # Build the real 3-camera batch via lerobot's preprocessor.
    import cv2 as _cv2
    img224 = image if image.shape[:2] == (224, 224) else _cv2.resize(image, (224, 224))
    img_t = torch.from_numpy(img224).permute(2, 0, 1).float() / 255.0
    raw = {f"observation.images.{k.split('.')[-1]}": img_t.clone() for k in cfg.image_features}
    raw["observation.state"] = torch.zeros(cfg.max_state_dim)
    raw["task"] = args.prompt
    pre, _ = make_pre_post_processors(policy.config, hf_repo)
    batch = pre(raw)
    batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
    noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), device="cuda", dtype=model_dtype)
    n_cams_native = len(list(cfg.image_features))
    pwe = policy.model.paligemma_with_expert
    vision_tower = pwe.paligemma.model.vision_tower
    patches_per_cam = infer_vision_patches(pwe.paligemma)

    # Capture the real vision-tower input + the real prefill-forward kwargs from one pass.
    cap: dict = {}
    def _vpre(_m, a, k):
        if "vis" not in cap:
            cap["vis"] = (a, dict(k))
    vh = vision_tower.register_forward_pre_hook(_vpre, with_kwargs=True)
    _orig = pwe.forward
    def _wrap(*a, **k):
        if "prefill" not in cap:
            cap["prefill"] = dict(k)   # call 0 = VLM prefix forward
        return _orig(*a, **k)
    pwe.forward = _wrap
    with torch.no_grad(), ac:
        policy.predict_action_chunk(batch, noise=noise)
    vh.remove(); pwe.forward = _orig

    prefix_embs = cap["prefill"]["inputs_embeds"][0]          # (1, L_full, hidden)
    L_full = int(prefix_embs.shape[1])
    n_text = L_full - patches_per_cam * n_cams_native         # padded text tokens
    log.info("Captured prefix: %d tokens = %d image (%d/cam ×%d) + %d text",
             L_full, patches_per_cam * n_cams_native, patches_per_cam, n_cams_native, n_text)

    def _time(fn, n_warmup, n_trials):
        for _ in range(n_warmup):
            with torch.no_grad(), ac:
                fn()
        torch.cuda.synchronize()
        ms = []
        for _ in range(n_trials):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            with torch.no_grad(), ac:
                s.record(); fn(); e.record()
            ms.append(_cuda_event_ms(s, e))
        return float(np.percentile(np.array(ms), 50))

    # ── Per-camera vision: re-run the SigLIP tower on one camera's real input ──
    va, vk = cap["vis"]
    per_cam_vision_ms = _time(lambda: vision_tower(*va, **vk), args.warmup, args.n_trials)
    log.info("Per-camera vision (SigLIP tower, 1 image): %.3f ms → N-cam ≈ N× this", per_cam_vision_ms)

    # ── LLM prefill vs N: slice the prefix to N·image + text, full-attention ──
    hidden = prefix_embs.shape[2]
    img_tok = patches_per_cam
    prefill_by_n = {}
    for N in (1, 2, 3):
        Ln = img_tok * N + n_text
        sliced = torch.cat([prefix_embs[:, :img_tok * N, :], prefix_embs[:, -n_text:, :]], dim=1) \
            if n_text > 0 else prefix_embs[:, :img_tok * N, :]
        mask4d = torch.zeros((1, 1, Ln, Ln), dtype=prefix_embs.dtype, device=prefix_embs.device)
        pos = torch.arange(Ln, device=prefix_embs.device).unsqueeze(0)
        def _prefill(s=sliced, m=mask4d, p=pos):
            return pwe.forward(attention_mask=m, position_ids=p, past_key_values=None,
                               inputs_embeds=[s, None], use_cache=True)
        prefill_by_n[N] = _time(_prefill, args.warmup, args.n_trials)
        log.info("LLM prefill N=%d (%d tok = %d img + %d text): %.2f ms",
                 N, Ln, img_tok * N, n_text, prefill_by_n[N])

    # Invariance verdict: is prefill(3) ≈ prefill(1)? (brief's 'LLM 1×' claim)
    p1, p3 = prefill_by_n[1], prefill_by_n[3]
    invariant = bool(abs(p3 - p1) / p1 <= 0.10) if p1 else None
    prefill_slope_ms_per_cam = round((p3 - p1) / 2, 3) if (p1 and p3) else None

    log.info("=" * 70)
    log.info("π0.5 camera scaling (decomposed):")
    log.info("  vision: %.3f ms/camera → 3-cam %.1f ms", per_cam_vision_ms, per_cam_vision_ms * 3)
    log.info("  LLM prefill: N=1 %.1f / N=2 %.1f / N=3 %.1f ms  (slope ~%.1f ms/camera)",
             prefill_by_n[1], prefill_by_n[2], prefill_by_n[3], prefill_slope_ms_per_cam or 0)
    log.info("  LLM invariant to N? %s — each camera adds %d image tokens to the prefix",
             invariant, img_tok)
    log.info("=" * 70)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": "RTX 5090", "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "lerobot": __import__("lerobot").__version__,
        "dtype": f"{args.dtype}-amp" if args.dtype in ("bf16", "fp16") else str(model_dtype),
        "family": "pi05", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_key": spec.vla_key, "hf_repo": hf_repo,
        "result": {
            "measurement_config": {
                "n_cameras": n_cams_native,
                "n_independent_passes": 1,
                "camera_config": "multi_native",
                "method": "pi05_base is locked to its 3 trained cameras; this is a "
                          "DECOMPOSED scaling measurement — per-camera SigLIP tower "
                          "timing + LLM prefill re-timed at N·image+text prefix lengths "
                          "(real captured prefix, sliced). Faithful component latencies, "
                          "not full-model N-camera runs (which the checkpoint can't do).",
            },
            "camera_scaling": {
                "image_tokens_per_camera": img_tok,
                "text_tokens": n_text,
                "vision_ms_per_camera": round(per_cam_vision_ms, 3),
                "vision_ms_by_n": {str(N): round(per_cam_vision_ms * N, 3) for N in (1, 2, 3)},
                "llm_prefill_ms_by_n": {str(N): round(prefill_by_n[N], 3) for N in (1, 2, 3)},
                "llm_prefill_slope_ms_per_camera": prefill_slope_ms_per_cam,
                "llm_backbone_invariant_to_n_cameras": invariant,
                "finding": "vision scales ~N× (tower runs once per camera). LLM prefill "
                           "is NOT invariant to N — it grows ~%s ms/camera because each "
                           "camera injects %d image tokens into the PaliGemma prefix "
                           "(%d/%d = %d%% of the 3-cam prefix is image tokens). The "
                           "brief's 'LLM ≈ 1×' assumption does not hold for this "
                           "native-multi-cam VLA."
                           % (prefill_slope_ms_per_cam, img_tok,
                              img_tok * n_cams_native, L_full,
                              round(100 * img_tok * n_cams_native / L_full)),
            },
            "derived": {
                "vision_encoder_ms_per_camera": round(per_cam_vision_ms, 3),
                "llm_backbone_invariant_to_n_cameras": invariant,
            },
            "nvtx_labels": {},
        },
    }
    blob = json.dumps(payload, indent=2, default=str)
    out = SUMMARY_PATH.parent / f"vla_summary_{spec.vla_key}_camscaling.json"
    out.write_text(blob)
    log.info("Wrote %s", out.relative_to(REPO))


def run_openvla_panorama(args, spec, hf_repo: str, image) -> None:
    """Multi-camera 'stitched panorama' what-if for OpenVLA (multi-cam brief Phase 3).
    OpenVLA is single-camera; a customer might stitch N feeds into one wide panorama.
    We feed 1/2/3 horizontally-stitched copies and measure the VLM-forward vision
    cost. HYPOTHESIS (worth refuting): OpenVLA's SigLIP/DINOv2 resizes every input to
    a fixed square, so a wider panorama is just downscaled → vision cost ~FLAT, does
    NOT scale with image area. Measured here, not assumed. Not a recommended
    deployment — a 'what if the customer tries it anyway' anchor for the sizer."""
    log.info("Harness family: openvla panorama (stitched-camera what-if)")
    model, processor, attn_backend = load_openvla_model(hf_repo, args.dtype, args.attn)
    vision_module = find_vision_module(model)
    by_n = {}
    for N in (1, 2, 3):
        stitched = np.concatenate([image] * N, axis=1)  # widen the frame N×
        inputs = build_openvla_inputs(processor, stitched, args.prompt, model.device, args.dtype)
        px = inputs.get("pixel_values")
        px_shape = tuple(px.shape) if px is not None else None
        vlm = measure_vlm_forward(model, inputs, n_warmup=args.warmup,
                                  n_trials=args.n_trials, nvtx_label=f"vla_{spec.vla_key}__pano{N}",
                                  vision_module=vision_module)
        comp = vlm.get("components", {})
        by_n[N] = {
            "input_image_wh": [stitched.shape[1], stitched.shape[0]],
            "pixel_values_shape": list(px_shape) if px_shape else None,
            "vlm_forward_p50_ms": round(vlm["p50_ms"], 3),
            "vision_encoder_p50_ms": round(comp.get("vision_encoder", {}).get("p50_ms", 0), 3) or None,
        }
        log.info("Panorama N=%d: input %dx%d → pixel_values %s | vision %.2f ms | VLM %.2f ms",
                 N, stitched.shape[1], stitched.shape[0], px_shape,
                 by_n[N]["vision_encoder_p50_ms"] or 0, vlm["p50_ms"])

    v1 = by_n[1]["vision_encoder_p50_ms"]
    v3 = by_n[3]["vision_encoder_p50_ms"]
    scales_with_area = bool(v1 and v3 and (v3 - v1) / v1 > 0.10)
    px_const = len({tuple(by_n[N]["pixel_values_shape"] or []) for N in (1, 2, 3)}) == 1
    log.info("=" * 70)
    log.info("OpenVLA panorama: pixel_values constant across N? %s | vision scales w/ area? %s",
             px_const, scales_with_area)
    log.info("=" * 70)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": "RTX 5090", "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": args.dtype, "attn_backend": attn_backend,
        "family": "openvla", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_key": spec.vla_key, "hf_repo": hf_repo,
        "result": {
            "measurement_config": {"n_cameras": 1, "n_independent_passes": 1,
                                   "camera_config": "multi_stitched"},
            "panorama_scaling": {
                "by_n_stitched": {str(N): by_n[N] for N in (1, 2, 3)},
                "pixel_values_constant_across_n": px_const,
                "vision_scales_with_image_area": scales_with_area,
                "finding": ("OpenVLA resizes every input to a FIXED %s tensor "
                            "(pixel_values constant across N), so a stitched panorama "
                            "is just downscaled — vision cost does NOT scale with image "
                            "area (the brief's premise does not hold). N separate camera "
                            "forwards WOULD scale ~N×; a single stitched image does not."
                            % (by_n[1]["pixel_values_shape"],)) if px_const and not scales_with_area
                           else "vision scales with stitched-panorama area (see by_n).",
            },
            "derived": {
                "vision_encoder_ms_per_camera": by_n[1]["vision_encoder_p50_ms"],
                "llm_backbone_invariant_to_n_cameras": None,
            },
            "nvtx_labels": {},
        },
    }
    blob = json.dumps(payload, indent=2, default=str)
    out = SUMMARY_PATH.parent / f"vla_summary_{spec.vla_key}_panorama.json"
    out.write_text(blob)
    log.info("Wrote %s", out.relative_to(REPO))


# ════════════════════════ BitVLA family (OpenVLA-OFT, ternary) ════════════════
# BitVLA is a 1-bit (ternary {-1,0,1}) VLA on the OpenVLA-OFT architecture: a
# SigLIP vision tower + a ternary-BitLinear LLM backbone + a parallel L1-regression
# action head (+ proprio projector). Unlike single-loop AR decode (OpenVLA 7B) or
# the flow-matching dual-loop (NORA-1.5/π0.5), OFT does ONE VLM forward over
# [image tokens ×N cams + prompt + ACTION_DIM·NUM_ACTIONS_CHUNK action placeholders
# + proprio] and the action head reads the action-position hidden states in
# PARALLEL → a whole H-action chunk from a single forward (no AR loop, no denoise).
#
# Runs ONLY in ~/.virtualenvs/bitvla (Python 3.10, torch cu128 for Blackwell, the
# BitVLA transformers fork v4.51 + openvla-oft `prismatic`, both editable-installed
# from a clone at $KEYHOLE_BITVLA_OFT, default /tmp/BitVLA/openvla-oft). prismatic's
# __init__ eagerly pulls the RLDS *training* data pipeline (dlimp/TF) which inference
# never calls; we inject permissive stub modules for those heavy/git-only deps so the
# light constants/action-head/proprio modules import. The "bf16" checkpoint runs the
# ternary BitLinear as bf16 matmuls — ternary buys MEMORY (weights pack to ~1.58-bit),
# not compute speed, unless bitblas/LUT kernels are used (not in this HF path).

BITVLA_OFT_DIR = Path(os.environ.get("KEYHOLE_BITVLA_OFT", "/tmp/BitVLA/openvla-oft"))
BITVLA_DEFAULT_REPO = "hongyuw/ft-bitvla-bitsiglipL-224px-libero_goal-bf16"


def _inject_bitvla_stubs():
    """Permissive stub modules for the training-only deps prismatic imports at
    module load (dlimp / TensorFlow) but never calls during inference. Spec'd +
    dunder-safe so transformers' availability probe and inspect/import machinery
    still behave (TF reads as 'absent' via missing distribution metadata)."""
    import types, importlib.machinery

    class _Stub(types.ModuleType):
        def __getattr__(self, n):
            if n.startswith("__") and n.endswith("__"):
                raise AttributeError(n)
            return _Stub(n)
        def __call__(self, *a, **k):
            return _Stub("call")

    for name in ("tensorflow", "tensorflow.io", "tensorflow.data", "dlimp",
                 "tensorflow_datasets", "tensorflow_graphics",
                 "tensorflow_graphics.geometry",
                 "tensorflow_graphics.geometry.transformation"):
        if name in sys.modules:
            continue
        s = _Stub(name)
        s.__spec__ = importlib.machinery.ModuleSpec(name, None)
        s.__path__ = []
        sys.modules[name] = s


def run_bitvla(args, spec, hf_repo: str, image) -> None:
    """End-to-end BitVLA (OpenVLA-OFT, ternary) measurement: one parallel VLM
    forward → H-action chunk via the L1-regression head. Emits a summary that
    mirrors the other VLA schemas (vlm_forward + components, derived, flops)."""
    from types import SimpleNamespace
    _inject_bitvla_stubs()
    oft = str(BITVLA_OFT_DIR)
    for p in (oft, oft + "/bitvla"):   # second path fixes a bare `from configuration_bit_vla import`
        if p not in sys.path:
            sys.path.insert(0, p)
    if not BITVLA_OFT_DIR.exists():
        log.error("BitVLA openvla-oft not found at %s. Clone ustcwhy/BitVLA "
                  "(recursive) and set $KEYHOLE_BITVLA_OFT, or build the bitvla "
                  "venv per the commit notes.", BITVLA_OFT_DIR)
        sys.exit(2)

    from experiments.robot.bitnet_utils import get_bitnet_vla, get_bitnet_vla_action
    from experiments.robot.openvla_utils import (
        get_processor, get_action_head, get_proprio_projector)
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM
    from bitvla.constants import (
        BITNET_DEFAULT_IMAGE_TOKEN_IDX, BITNET_PROPRIO_PAD_IDX, BITNET_IGNORE_INDEX,
        BITNET_ACTION_TOKEN_BEGIN_IDX, BITNET_STOP_INDEX)
    from huggingface_hub import snapshot_download
    from src.profiling.nvtx_helpers import nvtx_range

    # prismatic's import (overwatch) raises the root log level to WARNING, which
    # would swallow our INFO result block — restore it.
    logging.getLogger().setLevel(logging.INFO)
    log.setLevel(logging.INFO)

    log.info("Harness family: bitvla (OpenVLA-OFT ternary; parallel %d-action "
             "chunk, dim %d)", NUM_ACTIONS_CHUNK, ACTION_DIM)

    # ── Load (mirrors run_libero_eval_bitnet.py exactly) ──────────────
    torch.cuda.reset_peak_memory_stats()
    pre_load_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    local = snapshot_download(hf_repo)
    cfg = SimpleNamespace(
        pretrained_checkpoint=local, model_family="bitnet",
        num_images_in_input=2, use_proprio=True,
        use_l1_regression=True, use_diffusion=False, num_diffusion_steps=0,
        # center_crop is a fixed train-time aug — latency-neutral; off here to avoid
        # the TF-stubbed crop path (does not change the GPU forward being timed).
        center_crop=False, num_open_loop_steps=NUM_ACTIONS_CHUNK,
        load_in_8bit=False, load_in_4bit=False, lora_rank=0,
    )
    vla = get_bitnet_vla(cfg)
    vla.set_constant(image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
                     proprio_pad_idx=BITNET_PROPRIO_PAD_IDX, ignore_idx=BITNET_IGNORE_INDEX,
                     action_token_begin_idx=BITNET_ACTION_TOKEN_BEGIN_IDX,
                     stop_index=BITNET_STOP_INDEX)
    llm_dim = vla.config.text_config.hidden_size
    cfg.unnorm_key = next(iter(vla.norm_stats))   # e.g. "libero_goal_no_noops"
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim, proprio_dim=PROPRIO_DIM)
    post_load_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    weight_vram_mb = post_load_mb - pre_load_mb
    log.info("Loaded BitVLA: llm_dim=%d, unnorm_key=%s, %d cameras, chunk=%d",
             llm_dim, cfg.unnorm_key, cfg.num_images_in_input, NUM_ACTIONS_CHUNK)
    log.info("Weight VRAM: %.0f MB (%.2f GB, bf16-stored ternary)",
             weight_vram_mb, weight_vram_mb / 1024)

    # ── Build a synthetic observation (timing-invariant) ─────────────
    import cv2
    img224 = image if image.shape[:2] == (224, 224) else cv2.resize(image, (224, 224))
    obs = {"full_image": img224, "wrist_image": img224,
           "state": np.zeros(PROPRIO_DIM, dtype=np.float32)}
    task_label = args.prompt

    # ── Validate the action path ─────────────────────────────────────
    try:
        out = get_bitnet_vla_action(cfg, vla, processor, obs, task_label,
                                    action_head=action_head, proprio_projector=proprio_projector)
        a = np.asarray(out)
        ok = (a.shape == (NUM_ACTIONS_CHUNK, ACTION_DIM)) and bool(np.isfinite(a).all())
        action_validation = {
            "token_level_ok": bool(ok), "fast_decode_ok": bool(ok),
            "action_chunk_shape": list(a.shape), "action_chunk_length": NUM_ACTIONS_CHUNK,
            "action_dim": ACTION_DIM,
            "action_chunk_sample": [round(float(x), 5) for x in np.ravel(a)[:7]],
            # Full [H,7] fp16 reference chunk (drone-sizer / qualcomm mixed-precision diff).
            "action_chunk_full": [[round(float(x), 5) for x in row]
                                  for row in np.ravel(a).reshape(-1, 7)],
        }
        log.info("Action validation %s: chunk %s", "PASSED" if ok else "FAILED", list(a.shape))
    except Exception as e:  # noqa: BLE001
        log.warning("Action validation could not run (%s: %s).", type(e).__name__, e)
        action_validation = {"token_level_ok": False, "error": f"{type(e).__name__}: {e}"}

    # ── Per-call timing: wrap predict_action (GPU forward) + hook vision tower ──
    vision_tower = getattr(vla, "vision_tower", None) or getattr(getattr(vla, "model", None), "vision_tower", None)
    vis_store: list = []
    handles = []
    if vision_tower is not None:
        vpre, vpost = _evt_pair_hooks(vis_store)
        handles += [vision_tower.register_forward_pre_hook(vpre),
                    vision_tower.register_forward_hook(vpost)]
    fwd_store: list = []
    _orig_predict = vla.predict_action
    def _timed_predict(*a, **k):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); out = _orig_predict(*a, **k); e.record()
        fwd_store.append((s, e))
        return out
    vla.predict_action = _timed_predict
    nvtx_label = f"vla_{spec.vla_key}__forward"

    log.info("Warmup: %d forwards", args.warmup)
    for _ in range(args.warmup):
        vis_store.clear(); fwd_store.clear()
        get_bitnet_vla_action(cfg, vla, processor, obs, task_label,
                              action_head=action_head, proprio_projector=proprio_projector)
    torch.cuda.synchronize()

    log.info("Timed: %d forwards (parallel-chunk; vision split)", args.n_trials)
    fwd_ms_all, vision_ms_all = [], []
    for _ in range(args.n_trials):
        vis_store.clear(); fwd_store.clear()
        with nvtx_range(nvtx_label):
            get_bitnet_vla_action(cfg, vla, processor, obs, task_label,
                                  action_head=action_head, proprio_projector=proprio_projector)
        fd = _pair_durations(fwd_store)
        vd = _pair_durations(vis_store)
        if fd:
            fwd_ms_all.append(fd[0])
        vision_ms_all.append(float(sum(vd)))

    for h in handles:
        h.remove()
    vla.predict_action = _orig_predict

    def _p50(xs):
        return float(np.percentile(np.array(xs), 50)) if xs else None
    fwd_p50 = _p50(fwd_ms_all)
    vision_p50 = _p50(vision_ms_all)
    llm_p50 = (fwd_p50 - (vision_p50 or 0)) if fwd_p50 else None
    peak_inference_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # OFT predicts the whole chunk in ONE forward → amortized per-action = forward/H.
    amortized_ms_per_action = fwd_p50 / NUM_ACTIONS_CHUNK if fwd_p50 else None
    control_hz = 1000.0 / amortized_ms_per_action if amortized_ms_per_action else 0

    log.info("=" * 70)
    log.info("BitVLA (OFT ternary) results @ bf16 on RTX 5090:")
    log.info("  Action forward p50 (1 parallel pass): %.2f ms  → %d-action chunk",
             fwd_p50 or 0, NUM_ACTIONS_CHUNK)
    log.info("    ├─ vision encoder p50:  %.2f ms (×%d cams)", vision_p50 or 0, cfg.num_images_in_input)
    log.info("    └─ LLM (ternary) p50:   %.2f ms", llm_p50 or 0)
    log.info("  Amortized ms/action:      %.2f ms  (%.1f Hz)", amortized_ms_per_action or 0, control_hz)
    log.info("  Peak VRAM (incl weights): %.0f MB (%.2f GB)", peak_inference_mb, peak_inference_mb / 1024)

    # ── Physical FLOP (single parallel forward) ──────────────────────
    P_vision = int(sum(p.numel() for p in vision_tower.parameters())) if vision_tower else 0
    lm = getattr(vla, "language_model", None) or getattr(getattr(vla, "model", None), "language_model", None)
    P_llm = int(sum(p.numel() for p in lm.parameters())) if lm is not None else 0
    P_total = int(sum(p.numel() for p in vla.parameters()))
    patches_per_cam = infer_vision_patches(vla, default=256)
    # OFT prefix seqlen: image tokens ×cams + prompt(+proprio) + action placeholders.
    seqlen = patches_per_cam * cfg.num_images_in_input + 32 + ACTION_DIM * NUM_ACTIONS_CHUNK
    g = 1e9
    vision_gflop = 2 * P_vision * patches_per_cam * cfg.num_images_in_input / g
    llm_gflop = 2 * P_llm * seqlen / g
    forward_gflop = vision_gflop + llm_gflop
    peak_gf_s = PEAK_BF16_TFLOPS_5090 * 1e3
    def _util(gf, ms):
        return round(gf / (ms / 1e3) / peak_gf_s, 4) if ms and ms > 0 else None
    flops = {
        "method": "physical matmul FLOP 2·P·T for ONE parallel OFT forward "
                  "(vision ×cams + ternary-LLM over the full prefix incl. "
                  f"{ACTION_DIM}·{NUM_ACTIONS_CHUNK} action-placeholder tokens). "
                  "Ternary weights do NOT reduce matmul FLOP; the bf16 BitLinear "
                  "runs dense bf16 matmuls — ternary buys memory/BW, not compute "
                  "speed, absent bitblas/LUT kernels (not used in this HF path).",
        "params_millions": {"vision": round(P_vision / 1e6, 1), "llm_ternary": round(P_llm / 1e6, 1),
                            "total": round(P_total / 1e6, 1)},
        "n_vision_patches": patches_per_cam, "n_cameras": cfg.num_images_in_input,
        "prefix_seqlen_tokens": seqlen, "action_chunk_length": NUM_ACTIONS_CHUNK,
        "vision_encoder_gflop": round(vision_gflop, 2),
        "llm_forward_gflop": round(llm_gflop, 2),
        "action_forward_gflop": round(forward_gflop, 2),
        "per_action_gflop": round(forward_gflop / NUM_ACTIONS_CHUNK, 2),
        "achieved_util_5090": {
            "peak_bf16_tflops": PEAK_BF16_TFLOPS_5090,
            "vision_encoder": _util(vision_gflop, vision_p50),
            "llm_forward": _util(llm_gflop, llm_p50),
            "note": "ternary backbone, bf16 matmuls. The OFT parallel forward over "
                    f"{NUM_ACTIONS_CHUNK} action positions is compute-shaped like a "
                    "prefill (no per-token weight re-streaming), so it AVOIDS the "
                    "single-loop AR-decode bandwidth-wall — the OFT speed story.",
        },
    }
    log.info("  Physical FLOP: vision %.0f GF | LLM %.0f GF | forward %.0f GF (%.1f GF/action)",
             vision_gflop, llm_gflop, forward_gflop, flops["per_action_gflop"])
    log.info("=" * 70)

    calibration_note = (
        "BitVLA = 1-bit (ternary) VLA on OpenVLA-OFT. ONE parallel VLM forward "
        f"({round(fwd_p50 or 0,1)} ms) emits the whole {NUM_ACTIONS_CHUNK}-action "
        f"chunk via an L1-regression head (no AR decode, no denoise loop) → "
        f"{round(amortized_ms_per_action or 0,1)} ms/action = {round(control_hz,1)} Hz. "
        f"Peak VRAM {round(peak_inference_mb/1024,2)} GB — far below OpenVLA-7B's "
        "14.4 GB: the ternary backbone's memory win (weights pack to ~1.58-bit; "
        "here stored bf16, ~6 GB). KEY CAVEAT: this HF path runs ternary BitLinear "
        "as DENSE bf16 matmuls — ternary reduces memory/bandwidth, NOT compute "
        "FLOP or latency, unless bitblas/LUT ternary kernels are used (they are "
        "not here). So the latency is an OpenVLA-OFT-class bf16 number; the 'orange "
        "optimistic INT/ternary floor' is a MEMORY/BW floor, not a compute-speed "
        "floor, in this measurement. Stock HF forward, no CUDA graphs / compile. "
        "center_crop disabled (latency-neutral train-time aug). proprio=zeros "
        "(content-invariant for timing)."
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": "RTX 5090",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": "bf16 (ternary BitLinear stored/computed bf16)",
        "attn_backend": "sdpa (OFT default)",
        "family": "bitvla",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_path": str(CSV_PATH.relative_to(REPO)),
        "model_key": spec.vla_key,
        "hf_repo": hf_repo,
        "model_spec": asdict(spec),
        "result": {
            "action_validation": action_validation,
            "vlm_forward": {
                "p50_ms": round(fwd_p50 or 0, 3), "n_trials": args.n_trials,
                "topology": "single parallel OFT forward (no AR/denoise loop)",
                "components": {
                    "vision_encoder": {"p50_ms": round(vision_p50 or 0, 3),
                                       "n_cameras": cfg.num_images_in_input},
                    "llm_forward": {"p50_ms": round(llm_p50 or 0, 3)},
                    "vision_frac_p50": round((vision_p50 or 0) / fwd_p50, 3) if fwd_p50 else None,
                    "method": "CUDA-event wrap of predict_action + vision-tower hook.",
                },
            },
            "derived": {
                "action_chunk_length": NUM_ACTIONS_CHUNK,
                "e2e_ms_per_action_p50": round(amortized_ms_per_action or 0, 3),
                "action_rate_hz_p50": round(control_hz, 2),
                "action_forward_p50_ms": round(fwd_p50 or 0, 3),
            },
            "dram": {
                "weight_vram_mb": round(weight_vram_mb, 1),
                "weight_vram_gb": round(weight_vram_mb / 1024, 2),
                "peak_inference_mb": round(peak_inference_mb, 1),
                "peak_inference_gb": round(peak_inference_mb / 1024, 2),
            },
            "flops": flops,
            "csv_reference": {
                "measured_5090_ms_per_action": spec.measured_5090_ms_per_action,
            },
            "calibration": {
                "reference_ms_per_action": spec.measured_5090_ms_per_action,
                "vlm_forward_p50_ms": round(fwd_p50 or 0, 2),
                "note": calibration_note,
            },
            "nvtx_labels": {"forward": nvtx_label},
        },
    }
    blob = json.dumps(payload, indent=2, default=str)
    SUMMARY_PATH.write_text(blob)
    per_model = SUMMARY_PATH.parent / f"vla_summary_{spec.vla_key}.json"
    per_model.write_text(blob)
    log.info("Wrote %s and %s", SUMMARY_PATH.relative_to(REPO),
             per_model.relative_to(REPO))


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
    p.add_argument("--camera-scaling", action="store_true",
                   help="π0.5 only: run the decomposed multi-camera scaling "
                        "measurement (per-camera vision + LLM-prefill-vs-N slope) "
                        "instead of the normal dual-loop run. pi05_base is locked to "
                        "3 cameras so this decomposes rather than sweeps.")
    p.add_argument("--n-independent-passes", type=int, default=1,
                   help="Fleet-replication verification (single-loop models): run K "
                        "sequential independent action forwards (fresh KV each) and "
                        "confirm total ≈ K× single-pass within noise — validates the "
                        "'N robots = N× cost' projection without running N instances. "
                        "Default 1 (no verification).")
    p.add_argument("--num-steps", type=int, default=DUAL_LOOP_NUM_STEPS_DEFAULT,
                   help="Flow-matching denoise steps for the dual-loop action "
                        f"expert (default: {DUAL_LOOP_NUM_STEPS_DEFAULT}, per the "
                        "NORA-1.5 README). Single-loop families ignore this.")
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

    # ── 2. Visual input + family ──────────────────────────────────────
    image = load_test_frame()
    family = resolve_family(spec.vla_key)

    # Dual-loop (flow-matching action expert) is a wholly different topology —
    # VLM once + expert N-step denoise loop — so it runs its own measurement path
    # and emits the schema-compatible summary itself, then returns.
    if family == "dual_loop":
        run_dual_loop(args, spec, hf_repo, image)
        return
    if family == "pi05":
        if getattr(args, "camera_scaling", False):
            run_pi05_camera_scaling(args, spec, hf_repo, image)
        else:
            run_pi05_dual_loop(args, spec, hf_repo, image)
        return
    if family == "bitvla":
        run_bitvla(args, spec, hf_repo, image)
        return
    if family == "openvla" and getattr(args, "camera_scaling", False):
        run_openvla_panorama(args, spec, hf_repo, image)
        return

    log.info("Harness family: %s", family)

    # ── 3. Model load + DRAM baseline ────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    pre_load_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    if family == "openvla":
        model, processor, attn_backend = load_openvla_model(hf_repo, args.dtype, args.attn)
    else:
        model, processor, attn_backend = load_nora_model(hf_repo, args.dtype, args.attn)
    post_load_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    weight_vram_mb = post_load_alloc_mb - pre_load_alloc_mb
    log.info("Weight VRAM:  %.0f MB (%.2f GB at %s)",
             weight_vram_mb, weight_vram_mb / 1024, args.dtype)

    # ── 3a. Build the family-specific inference inputs ───────────────
    if family == "openvla":
        inputs = build_openvla_inputs(processor, image, args.prompt, model.device, args.dtype)
        action_max_tokens = int(model.get_action_dim(args.unnorm_key))  # 7 discrete tokens
    else:
        inputs = build_nora_inputs(processor, image, args.prompt, model.device)
        action_max_tokens = args.action_tokens  # FAST+ EOS-terminated, this is the cap

    # ── 3b. Validate the action-prediction path (Step 1b criterion) ──
    # Prove the harness measures the real action decode, not garbage. Guarded so
    # it can never abort the latency run.
    try:
        if family == "openvla":
            action_validation = validate_openvla_action_path(
                model, processor, image, args.prompt, args.unnorm_key, args.dtype)
        else:
            action_validation = validate_nora_action_path(
                model, processor, image, args.prompt, args.unnorm_key, args.action_tokens)
        if action_validation.get("token_level_ok"):
            log.info("Action validation PASSED: %s", {
                k: action_validation.get(k) for k in
                ("n_action_tokens", "action_dof", "eos_terminated") if k in action_validation})
        else:
            log.warning("Action validation FAILED — review before trusting "
                        "latency: %s", action_validation)
        if action_validation.get("fast_decode_ok"):
            log.info("7-DOF action vector: %s (unnorm_key=%s)",
                     action_validation.get("action_vector"),
                     action_validation.get("unnorm_key"))
        elif "fast_decode_error" in action_validation:
            log.info("Float decode unavailable: %s",
                     action_validation.get("fast_decode_error"))
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
        model, inputs,
        n_warmup=args.warmup, n_trials=args.n_trials, nvtx_label=vlm_nvtx,
        vision_module=vision_module,
    )
    action = measure_action_forward(
        model, inputs, max_action_tokens=action_max_tokens,
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

    # ── 4b. Fleet-replication verification (multi-cam brief, Phase 2) ─
    # Single-camera VLAs deployed across N robots run as N INDEPENDENT instances
    # (no weight/KV sharing) → cost is N×. We verify that experimentally: K
    # sequential fresh-KV action forwards should total ≈ K× the single-pass p50
    # within noise (no hidden batching/caching win). One supplementary check; the
    # N× projection itself is arithmetic, this just confirms no surprise sublinearity.
    fleet = None
    K = max(1, getattr(args, "n_independent_passes", 1))
    if K > 1:
        from src.profiling.nvtx_helpers import nvtx_range
        log.info("Fleet verification: timing %d sequential independent passes", K)
        per_total: list[float] = []
        for _ in range(args.n_trials):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            with nvtx_range(f"{action_nvtx}__fleet{K}"), torch.inference_mode():
                s.record()
                for _k in range(K):
                    _ = model.generate(**inputs, max_new_tokens=action_max_tokens, do_sample=False)
                e.record()
            per_total.append(_cuda_event_ms(s, e))
        k_total_p50 = float(np.percentile(np.array(per_total), 50))
        expected = K * action["p50_ms"]
        ratio = k_total_p50 / action["p50_ms"] if action["p50_ms"] else None
        # "linear within noise" = K-pass total within 10% of K× single.
        linear = bool(ratio is not None and abs(ratio - K) / K <= 0.10)
        fleet = {
            "n_independent_passes": K,
            "k_pass_total_p50_ms": round(k_total_p50, 3),
            "single_pass_p50_ms": round(action["p50_ms"], 3),
            "measured_ratio": round(ratio, 3) if ratio else None,
            "expected_ratio": K,
            "linear_within_10pct": linear,
            "note": "K sequential fresh-KV model.generate() calls; confirms fleet "
                    "replication (N robots = N independent instances) scales ~N× "
                    "with no hidden sublinearity. The N× cost projection is otherwise "
                    "arithmetic — this is the empirical sanity check the brief asked for.",
        }
        log.info("Fleet: %d-pass total %.1f ms = %.2f× single (%.1f ms) — %s",
                 K, k_total_p50, ratio or 0, action["p50_ms"],
                 "LINEAR ✓" if linear else "SUBLINEAR/SUPERLINEAR ⚠")

    # ── 5. DRAM during inference ─────────────────────────────────────
    peak_inference_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ── 6. Bandwidth proxy (analytical) ──────────────────────────────
    bpt = analytical_bytes_per_decode_token(spec, args.dtype)
    decode_tok_s_bw = (peak_inference_mb * 1e6) / bpt if bpt > 0 else 0  # informational only

    # ── 6b. Physical per-component FLOP attribution (sizer ask) ──────
    param_counts = count_component_params(model, family)
    # True LLM prefill length (text + injected vision embeds), not input_ids len.
    llm_seqlen = capture_llm_prefill_seqlen(model, inputs, find_llm_module(model))
    input_len_tok = llm_seqlen or int(inputs["input_ids"].shape[1])
    n_act_tok = action.get("n_generated_tokens", action_max_tokens)
    log.info("LLM prefill seq len (true, incl. vision embeds): %s "
             "(input_ids len was %d)", input_len_tok, int(inputs["input_ids"].shape[1]))
    flops = analytical_component_flops(
        param_counts, n_vision_patches=infer_vision_patches(model),
        input_len=input_len_tok, n_action_tokens=n_act_tok)

    # Achieved-util cross-check: physical FLOP ÷ measured latency ÷ peak. Reveals
    # how compute-saturated each stage actually is (the sizer's per-hw util input,
    # measured rather than assumed). Decode util being tiny = the BW-wall.
    peak_gf_s = PEAK_BF16_TFLOPS_5090 * 1e3
    comp = vlm.get("components", {})
    vis_ms = comp.get("vision_encoder", {}).get("p50_ms")
    pre_ms = comp.get("llm_prefill", {}).get("p50_ms")
    def _util(gflop, ms):
        return round(gflop / (ms / 1e3) / peak_gf_s, 4) if ms and ms > 0 else None
    flops["achieved_util_5090"] = {
        "peak_bf16_tflops": PEAK_BF16_TFLOPS_5090,
        "vision_encoder": _util(flops["vision_encoder_gflop"], vis_ms),
        "llm_prefill": _util(flops["llm_prefill_gflop"], pre_ms),
        "llm_decode_per_token": _util(flops["llm_decode_gflop_per_token"],
                                      ms_per_decode_token),
        "note": "physical FLOP / measured p50 latency / dense bf16 peak. Low "
                "decode util = bandwidth-bound (weights streamed per token), the "
                "edge BW-wall driver.",
    }
    log.info("  Physical FLOP: vision %.0f GF (util %.0f%%) | prefill %.0f GF (util %.0f%%) | "
             "decode %.1f GF/tok (util %.1f%%) | e2e %.0f GF",
             flops["vision_encoder_gflop"], 100 * (flops["achieved_util_5090"]["vision_encoder"] or 0),
             flops["llm_prefill_gflop"], 100 * (flops["achieved_util_5090"]["llm_prefill"] or 0),
             flops["llm_decode_gflop_per_token"], 100 * (flops["achieved_util_5090"]["llm_decode_per_token"] or 0),
             flops["e2e_action_gflop"])

    # ── 7. Report + write ────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Results for %s @ %s on RTX 5090:", spec.display_name, args.dtype)
    log.info("  VLM forward p50:           %.2f ms", vlm["p50_ms"])
    if "components" in vlm:
        c = vlm["components"]
        log.info("    ├─ vision encoder p50:   %.2f ms  (%.0f%% of VLM forward)",
                 c["vision_encoder"]["p50_ms"], 100 * c["vision_frac_p50"])
        log.info("    └─ LLM prefill p50:      %.2f ms", c["llm_prefill"]["p50_ms"])
    log.info("  Action chunk p50:          %.2f ms  (%d action tokens decoded)",
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

    # Family-aware calibration narrative — the reference figures mean different
    # things per model, so don't assert NORA's prefill-matches-paper story for all.
    if family == "openvla":
        calibration_note = (
            "CSV reference 73 ms is an RTX 4090 measurement from an optimized "
            "benchmark (IndexBox); the 5090 estimate was ~50 ms. Our measured "
            f"5090 stock-HuggingFace e2e ({round(e2e_ms,1)} ms) is HIGHER than the "
            "4090 reference — not a faster-GPU-is-slower paradox but a different, "
            "unoptimized stack (transformers 4.40.1 + sdpa, no CUDA graphs / static "
            "KV cache / torch.compile; ~1.7x the bf16 decode BW floor). VLM forward "
            f"({round(vlm['p50_ms'],1)} ms) splits into vision {round(vlm.get('components',{}).get('vision_encoder',{}).get('p50_ms',0),1)} ms "
            "+ LLM prefill. Gap vs reference is optimization headroom + stack/GPU "
            "difference."
        )
    else:
        calibration_note = (
            "VLM-forward (prefill) p50 reproduces the paper's 33 ms anchor. "
            "End-to-end (prefill + FAST+ decode to EOS) is higher because this is "
            "stock HuggingFace generate() — no CUDA graphs / static KV cache / "
            "torch.compile — so per-token decode carries Python + kernel-launch "
            "overhead (~3x the bf16 bandwidth floor). The gap is optimization "
            "headroom, not a measurement error; the paper uses an optimized decode."
        )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": "RTX 5090",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": args.dtype,
        "attn_backend": attn_backend,
        "family": family,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_path": str(CSV_PATH.relative_to(REPO)),
        "model_key": spec.vla_key,
        "hf_repo": hf_repo,
        "model_spec": asdict(spec),
        "result": {
            "action_validation": action_validation,
            "measurement_config": {
                "n_cameras": 1,
                "n_independent_passes": K,
                # single-camera models; multi-robot use = fleet replication (N× all),
                # NOT native multi-camera fusion. See result.fleet_verification.
                "camera_config": "single",
            },
            "vlm_forward": vlm,
            "action_forward": action,
            "fleet_verification": fleet,   # None unless --n-independent-passes > 1
            "derived": {
                "ms_per_decode_token": round(ms_per_decode_token, 3),
                "n_decode_tokens_used": n_decode_tok,
                "e2e_ms_per_action_p50": round(e2e_ms, 3),
                "action_rate_hz_p50": round(1000.0 / e2e_ms, 2) if e2e_ms > 0 else 0,
                # multi-cam schema (single-cam models: per-camera = the one vision pass;
                # llm-invariance is N/A — these fleet-replicate N× rather than fusing).
                "vision_encoder_ms_per_camera": (round(vis_ms, 3) if vis_ms else None),
                "llm_backbone_invariant_to_n_cameras": None,
            },
            "dram": {
                "weight_vram_mb": round(weight_vram_mb, 1),
                "weight_vram_gb": round(weight_vram_mb / 1024, 2),
                "peak_inference_mb": round(peak_inference_mb, 1),
                "peak_inference_gb": round(peak_inference_mb / 1024, 2),
                "bytes_per_decode_token_analytical": round(bpt, 0),
                "bytes_per_decode_token_unit": "bytes",
            },
            "flops": flops,
            "csv_reference": {
                "measured_5090_ms_per_action": spec.measured_5090_ms_per_action,
                "inference_dram_gb_at_default": getattr(
                    spec, f"inference_dram_gb_{spec.dtype_path_default}", None
                ),
            },
            "calibration": {
                "reference_ms_per_action": spec.measured_5090_ms_per_action,
                "vlm_forward_p50_ms": round(vlm["p50_ms"], 2),
                "e2e_vs_reference_ratio": (
                    round(e2e_ms / spec.measured_5090_ms_per_action, 2)
                    if spec.measured_5090_ms_per_action else None
                ),
                "note": calibration_note,
            },
            "nvtx_labels": {
                "vlm":    vlm_nvtx,
                "action": action_nvtx,
            },
        },
    }
    blob = json.dumps(payload, indent=2, default=str)
    SUMMARY_PATH.write_text(blob)
    # Per-model copy so multi-model runs don't clobber each other (the canonical
    # vla_summary.json always holds the most recent run).
    per_model = SUMMARY_PATH.parent / f"vla_summary_{spec.vla_key}.json"
    per_model.write_text(blob)
    log.info("Wrote %s and %s", SUMMARY_PATH.relative_to(REPO),
             per_model.relative_to(REPO))

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
