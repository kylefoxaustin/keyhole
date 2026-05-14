"""Loader for private NPU + CNN measured-silicon anchors.

Reads `.streamlit/secrets.toml` (gitignored). Same schema as
keyhole-sizer + PAI sizer — cross-project compatible by design. See spec
in personal-ai-framework/docs/private_anchor_secrets_spec.md.

DISCIPLINE: this module must never print, log, or surface measured
values in error messages. Loaders return None on missing / zero values;
log statements (when added) reference KEYS only. The chat-safe-code +
private-runtime split this module enforces lets `scripts/build_deck.py
--include-private` consume measured values at build time without the
values ever entering the source tree, the bus, or any log readable by
Claude.

Schema:

    [npu_llm_anchors.<tier_precision>.<model_key>]
    tokps = 0.0                   # headline; zero means "not measured"
    peak_bw_gbps = 134.4          # LPDDR5X-8.4 × 128-bit (both Mid + High)
    bw_share_frac = 0.75          # default share reserved for NPU
    bw_efficiency_frac = 0.70     # matches keyhole BW-efficiency methodology
    source = "measured" | "vendor_spec" | "projected"

    [cnn_anchors.<tier_precision>.<cnn_key>]
    ms_per_inference = 0.0        # headline; zero means "not measured"
    peak_bw_gbps = 134.4
    bw_share_frac = 0.75
    bw_efficiency_frac = 0.70
    source = "measured" | "vendor_spec" | "projected"
    input_resolution = "640x640"  # descriptive

Tier-precision values:
  - LLM: "mid_int8", "high_int8", "high_fp"
  - CNN: "mid_int8", "high_int8"   (CNN measured INT-only on NPU High)

Model keys (LLM): "qwen3_30b_a3b_moe", "qwen25_32b_dense", "qwen25_7b_dense"
CNN keys: "resnet50_w4", "yolov8n_w4", "yolov8n_w8"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# tomllib is stdlib on Python 3.11+; fall back to tomli on 3.10
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
SECRETS_PATH = REPO_ROOT / ".streamlit" / "secrets.toml"

# Provenance badge mapping — matches keyhole-sizer's _render_source_banner
# convention. Used by deck/UI consumers; loader returns the raw `source`
# string and the consumer picks a badge.
BADGE_FOR_SOURCE = {
    "measured":    "🟢",   # direct measurement on real silicon
    "vendor_spec": "🟡",   # vendor-published specification
    "projected":   "🟠",   # placeholder / projected
}


@dataclass
class LLMAnchor:
    """Single LLM measurement cell from `npu_llm_anchors.<tier>.<model>`."""
    tier: str                           # e.g. "mid_int8"
    model_key: str                      # e.g. "qwen3_30b_a3b_moe"
    tokps: float                        # decode tokens/sec
    peak_bw_gbps: float
    bw_share_frac: float
    bw_efficiency_frac: float
    source: str                         # "measured" | "vendor_spec" | "projected"

    def achieved_bw_gbps(self, share_override: float | None = None) -> float:
        """Effective BW available to the NPU under (optional) share override."""
        share = share_override if share_override is not None else self.bw_share_frac
        return self.peak_bw_gbps * share * self.bw_efficiency_frac

    def bytes_per_token(self, share_override: float | None = None) -> float:
        """Implied bytes-per-token from the measurement + BW model.

        bytes_per_token = achieved_bw_gbps × 10^9 / tokps   →   bytes
        Returns 0.0 if tokps is zero (avoid divide-by-zero); callers should
        check this before treating the value as meaningful.
        """
        if self.tokps <= 0:
            return 0.0
        bw_bytes_per_sec = self.achieved_bw_gbps(share_override) * 1e9
        return bw_bytes_per_sec / self.tokps

    @property
    def badge(self) -> str:
        return BADGE_FOR_SOURCE.get(self.source, "⚪")


@dataclass
class CNNAnchor:
    """Single CNN measurement cell from `cnn_anchors.<tier>.<cnn_key>`."""
    tier: str
    cnn_key: str                        # e.g. "resnet50_w4"
    ms_per_inference: float
    peak_bw_gbps: float
    bw_share_frac: float
    bw_efficiency_frac: float
    source: str
    input_resolution: str = ""          # descriptive, e.g. "640x640"

    def achieved_bw_gbps(self, share_override: float | None = None) -> float:
        share = share_override if share_override is not None else self.bw_share_frac
        return self.peak_bw_gbps * share * self.bw_efficiency_frac

    @property
    def fps(self) -> float:
        return 1000.0 / self.ms_per_inference if self.ms_per_inference > 0 else 0.0

    @property
    def badge(self) -> str:
        return BADGE_FOR_SOURCE.get(self.source, "⚪")


# ───────────────────────────────────────────────────────────────────────
# Loader
# ───────────────────────────────────────────────────────────────────────

_CACHE: dict[str, Any] | None = None


def _load_secrets() -> dict[str, Any]:
    """Read `.streamlit/secrets.toml` once and cache.

    Returns an empty dict if the file is absent or unreadable — caller
    treats "no anchors" as the universal fallback state.

    Discipline: never include values in any exception that escapes this
    function. The bare error message + path is all the caller learns.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not SECRETS_PATH.exists():
        _CACHE = {}
        return _CACHE
    try:
        with SECRETS_PATH.open("rb") as f:
            _CACHE = tomllib.load(f)
    except Exception:
        # Swallow the specific error (it might include a value snippet from
        # the TOML parser's context). Caller knows it failed; that's enough.
        sys.stderr.write(
            f"private_anchors: failed to parse {SECRETS_PATH.name} (treating as empty). "
            f"Verify TOML syntax in `{SECRETS_PATH}`.\n"
        )
        _CACHE = {}
    return _CACHE


def _get_cell(section: str, tier: str, key: str) -> dict[str, Any] | None:
    """Drill into secrets[section][tier][key]; return None if any layer missing."""
    secrets = _load_secrets()
    return secrets.get(section, {}).get(tier, {}).get(key)


def load_llm_anchor(tier: str, model_key: str) -> LLMAnchor | None:
    """Return an LLMAnchor for `npu_llm_anchors.<tier>.<model_key>`, or
    None if absent / placeholder (tokps == 0).

    Args:
        tier:      e.g. "mid_int8", "high_int8", "high_fp"
        model_key: e.g. "qwen3_30b_a3b_moe", "qwen25_32b_dense", "qwen25_7b_dense"
    """
    cell = _get_cell("npu_llm_anchors", tier, model_key)
    if cell is None:
        return None
    tokps = float(cell.get("tokps", 0.0))
    if tokps <= 0:
        return None  # placeholder cell
    return LLMAnchor(
        tier=tier,
        model_key=model_key,
        tokps=tokps,
        peak_bw_gbps=float(cell.get("peak_bw_gbps", 134.4)),
        bw_share_frac=float(cell.get("bw_share_frac", 0.75)),
        bw_efficiency_frac=float(cell.get("bw_efficiency_frac", 0.70)),
        source=str(cell.get("source", "projected")),
    )


def load_cnn_anchor(tier: str, cnn_key: str) -> CNNAnchor | None:
    """Return a CNNAnchor for `cnn_anchors.<tier>.<cnn_key>`, or None if
    absent / placeholder (ms_per_inference == 0).

    Args:
        tier:    e.g. "mid_int8", "high_int8"
        cnn_key: e.g. "resnet50_w4", "yolov8n_w4", "yolov8n_w8"
    """
    cell = _get_cell("cnn_anchors", tier, cnn_key)
    if cell is None:
        return None
    ms = float(cell.get("ms_per_inference", 0.0))
    if ms <= 0:
        return None
    return CNNAnchor(
        tier=tier,
        cnn_key=cnn_key,
        ms_per_inference=ms,
        peak_bw_gbps=float(cell.get("peak_bw_gbps", 134.4)),
        bw_share_frac=float(cell.get("bw_share_frac", 0.75)),
        bw_efficiency_frac=float(cell.get("bw_efficiency_frac", 0.70)),
        source=str(cell.get("source", "projected")),
        input_resolution=str(cell.get("input_resolution", "")),
    )


def have_any_measured_anchors() -> bool:
    """True iff at least one cell in the secrets file has a measurement
    (non-zero tokps or ms_per_inference).

    Useful for deck/CLI gating: `if not have_any_measured_anchors(): skip
    private slide`. Does NOT reveal any values.
    """
    secrets = _load_secrets()
    for tier_dict in secrets.get("npu_llm_anchors", {}).values():
        if not isinstance(tier_dict, dict):
            continue
        for cell in tier_dict.values():
            if isinstance(cell, dict) and float(cell.get("tokps", 0.0)) > 0:
                return True
    for tier_dict in secrets.get("cnn_anchors", {}).values():
        if not isinstance(tier_dict, dict):
            continue
        for cell in tier_dict.values():
            if isinstance(cell, dict) and float(cell.get("ms_per_inference", 0.0)) > 0:
                return True
    return False


# Schema-reference constants for callers that want to iterate the full grid.
LLM_TIERS = ("mid_int8", "high_int8", "high_fp")
LLM_MODELS = ("qwen3_30b_a3b_moe", "qwen25_32b_dense", "qwen25_7b_dense")
CNN_TIERS = ("mid_int8", "high_int8")  # CNN measured INT-only on NPU High
CNN_KEYS = ("resnet50_w4", "yolov8n_w4", "yolov8n_w8")

LLM_MODEL_LABELS = {
    "qwen3_30b_a3b_moe":  "Qwen3-30B-A3B MoE (Q4_K_M)",
    "qwen25_32b_dense":   "Qwen 2.5 32B Instruct (Q4_K_M)",
    "qwen25_7b_dense":    "Qwen 2.5 7B Instruct (Q4_K_M)",
}
CNN_LABELS = {
    "resnet50_w4":  "ResNet-50 (4-bit weights, 224×224)",
    "yolov8n_w4":   "YOLOv8n (4-bit weights, 640×640)",
    "yolov8n_w8":   "YOLOv8n (8-bit weights, 640×640)",
}
TIER_LABELS = {
    "mid_int8":   "NPU Mid INT8 (200 eTOPS)",
    "high_int8":  "NPU High INT8 (400 eTOPS)",
    "high_fp":    "NPU High FP (200 eTOPS BF16/FP8)",
}
