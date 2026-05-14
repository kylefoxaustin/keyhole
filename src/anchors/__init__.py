"""Private NPU + CNN measured-silicon anchor loaders.

This subpackage provides a typed loader for `.streamlit/secrets.toml` —
the gitignored file that holds measured silicon performance numbers Kyle
populates locally. The loader's contract:

  - Schema-aware: returns typed dataclasses (`LLMAnchor`, `CNNAnchor`)
  - Graceful: returns None if the secrets file is absent, the cell is
    absent, or the headline value is zero (the placeholder state)
  - Discipline-preserving: never prints, logs, or surfaces values in
    exception messages — only KEY references

Public API:

    from src.anchors import load_llm_anchor, load_cnn_anchor

    anchor = load_llm_anchor("mid_int8", "qwen3_30b_a3b_moe")
    if anchor is not None:
        # anchor.tokps and anchor.bytes_per_token(...) available
        ...

See `.streamlit/secrets.toml.example` for the schema and
`personal-ai-framework/docs/private_anchor_secrets_spec.md` for the
cross-project spec.
"""

from .private_anchors import (
    LLMAnchor,
    CNNAnchor,
    load_llm_anchor,
    load_cnn_anchor,
    have_any_measured_anchors,
    BADGE_FOR_SOURCE,
    LLM_TIERS, LLM_MODELS, LLM_MODEL_LABELS,
    CNN_TIERS, CNN_KEYS, CNN_LABELS,
    TIER_LABELS,
)

__all__ = [
    "LLMAnchor", "CNNAnchor",
    "load_llm_anchor", "load_cnn_anchor",
    "have_any_measured_anchors",
    "BADGE_FOR_SOURCE",
    "LLM_TIERS", "LLM_MODELS", "LLM_MODEL_LABELS",
    "CNN_TIERS", "CNN_KEYS", "CNN_LABELS",
    "TIER_LABELS",
]
