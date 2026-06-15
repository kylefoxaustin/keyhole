#!/usr/bin/env python3
"""
vision_quant_uncap.py — isolate the VISION-TOWER quantization contribution to
prefill (TTFT) latency for Qwen3-VL-4B-Instruct on RTX 5090 (sm_120), vLLM 0.22.

Question: does quantizing the vision encoder (in addition to the language model)
"uncap" prefill? Three arms, LM quant held constant where applicable so the ONLY
variable across B->C is the vision tower:

  A. BF16 everything            (no quant; ceiling reference)
  B. FP8 dynamic, LM-ONLY       (visual.* excluded from quant, stays BF16)
  C. FP8 dynamic, FULL          (LM AND visual.* quantized — vLLM "fp8" default)

vLLM 0.22 fp8 path: `quantization="fp8"` builds a default Fp8Config() with
ignored_layers=[], and the Qwen3VL vision tower receives quant_config (qwen3_vl.py
~L1675), so the DEFAULT online-fp8 behaviour quantizes the vision tower too (= arm C).
For arm B we monkeypatch Fp8Config.get_quant_method to return UnquantizedLinearMethod
for any module whose prefix contains "visual." (exact-prefix is_layer_skipped is too
brittle for the per-layer vit naming, so we scope by substring on the well-defined
`visual.` namespace).

After load we INSPECT the live model: count fp8 vs unquantized linear methods under
model.visual.* and model.language_model.* to PROVE the scoping took effect, and to
confirm the vision GEMMs actually run in fp8 in arm C (not silently dequantized).

Privacy: synthetic 768px image only. No personal corpus, no commits, no sends.

One arm per invocation (clean VRAM). Writes a per-arm fragment; a final --merge
pass aggregates the three fragments into data/output/vision_quant_uncap.json.

  ~/.virtualenvs/vllm_fp4/bin/python scripts/vision_quant_uncap.py --arm A
  ~/.virtualenvs/vllm_fp4/bin/python scripts/vision_quant_uncap.py --arm C
  ~/.virtualenvs/vllm_fp4/bin/python scripts/vision_quant_uncap.py --arm B
  ~/.virtualenvs/vllm_fp4/bin/python scripts/vision_quant_uncap.py --merge
"""
import argparse, base64, gc, io, json, os, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "data", "output")
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
FINAL = os.path.join(OUTDIR, "vision_quant_uncap.json")

ARMS = {
    "A": dict(label="bf16_full", quant=None, exclude_vision=False,
              desc="BF16 everything (no quant) — ceiling"),
    "B": dict(label="fp8_lm_only", quant="fp8", exclude_vision=True,
              desc="FP8 dynamic, language-model only (visual.* kept BF16)"),
    "C": dict(label="fp8_full", quant="fp8", exclude_vision=False,
              desc="FP8 dynamic, FULL (LM + visual.* quantized)"),
}


def synthetic_page(longest=768):
    """A synthetic 'page' (white bg, drawn text/rects). Prefill latency is
    content-independent — this carries NO personal data."""
    from PIL import Image, ImageDraw
    W = longest
    H = int(longest * 1.3)  # portrait page; longest side stays = longest
    if H > longest:  # keep longest side == 768
        H = longest
        W = int(longest / 1.3)
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, W - 20, 70], outline=(0, 0, 0), width=3)
    for i in range(14):
        y = 100 + i * 40
        d.line([30, y, W - 30, y], fill=(40, 40, 40), width=2)
        d.text((35, y - 14), f"line {i:02d} the quick brown fox 0123456789", fill=(0, 0, 0))
    d.rectangle([40, H - 160, W - 40, H - 40], outline=(0, 0, 120), width=3)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def install_vision_exclusion():
    """Monkeypatch Fp8Config.get_quant_method so any module under `visual.`
    falls back to UnquantizedLinearMethod (LM-only fp8). Returns nothing; the
    patch records which prefixes it skipped on the config instance."""
    from vllm.model_executor.layers.quantization import fp8 as fp8mod
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    orig = fp8mod.Fp8Config.get_quant_method

    def patched(self, layer, prefix):
        if isinstance(layer, LinearBase) and "visual." in prefix:
            if not hasattr(self, "_skipped_vision"):
                self._skipped_vision = []
            self._skipped_vision.append(prefix)
            return UnquantizedLinearMethod()
        return orig(self, layer, prefix)

    fp8mod.Fp8Config.get_quant_method = patched


def classify(method):
    return type(method).__name__


def _inspect_worker(self):
    """Runs INSIDE the worker process (via collective_rpc, bound method form:
    receives the Worker as `self`). Walks the loaded model and buckets
    linear-layer quant_method classes by vision/language. Proves which GEMMs
    actually run in fp8."""
    from vllm.model_executor.layers.linear import LinearBase
    model = getattr(getattr(self, "model_runner", None), "model", None)
    buckets = {"visual": {}, "language": {}, "other": {}}
    examples = {"visual": {}, "language": {}}
    if model is None:
        return buckets, examples, "model-not-found"
    for name, mod in model.named_modules():
        if not isinstance(mod, LinearBase):
            continue
        qm = getattr(mod, "quant_method", None)
        if qm is None:
            continue
        cls = type(qm).__name__
        grp = ("visual" if "visual" in name
               else "language" if "language_model" in name else "other")
        buckets[grp][cls] = buckets[grp].get(cls, 0) + 1
        if grp in examples and cls not in examples[grp]:
            examples[grp][cls] = name
    return buckets, examples, "ok"


def inspect_model(llm):
    """Walk the loaded model; for every linear layer with a quant_method,
    bucket by (vision vs language). Tries direct attribute access first
    (works when the model lives in-process), then collective_rpc."""
    from vllm.model_executor.layers.linear import LinearBase

    def walk(model):
        buckets = {"visual": {}, "language": {}, "other": {}}
        examples = {"visual": {}, "language": {}}
        for name, mod in model.named_modules():
            if not isinstance(mod, LinearBase):
                continue
            qm = getattr(mod, "quant_method", None)
            if qm is None:
                continue
            cls = type(qm).__name__
            grp = ("visual" if "visual" in name
                   else "language" if "language_model" in name else "other")
            buckets[grp][cls] = buckets[grp].get(cls, 0) + 1
            if grp in examples and cls not in examples[grp]:
                examples[grp][cls] = name
        return buckets, examples

    # Try several known attribute paths (engine structure varies by version).
    paths = [
        lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
        lambda: llm.llm_engine.engine_core.engine_core.model_executor
                   .driver_worker.worker.model_runner.model,
    ]
    for p in paths:
        try:
            model = p()
            b, e = walk(model)
            if any(b.values()):
                return b, e
        except Exception:
            continue
    # Fallback: run inside the worker process via collective_rpc.
    try:
        out = llm.llm_engine.collective_rpc(_inspect_worker)
        b, e, status = out[0]
        b["_inspect_status"] = status
        return b, e
    except Exception as ex:
        return {"visual": {}, "language": {}, "other": {},
                "_inspect_error": repr(ex)}, {}


def run_arm(arm_key, n_prompts=10, resolution=768, max_model_len=8192):
    cfg = ARMS[arm_key]
    if cfg["exclude_vision"]:
        install_vision_exclusion()

    from vllm import LLM, SamplingParams
    import numpy as np

    kw = dict(model=MODEL, max_model_len=max_model_len,
              limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.85,
              enable_prefix_caching=False, enforce_eager=False,
              disable_log_stats=True)
    if cfg["quant"] is not None:
        kw["quantization"] = cfg["quant"]

    llm = LLM(**kw)

    buckets, examples = inspect_model(llm)

    img = synthetic_page(resolution)
    questions = [
        "What is on this page?", "How many lines of text are shown?",
        "Describe the layout.", "Is there a box at the top?",
        "Summarize the document.", "What language is this?",
        "List the visible elements.", "Is this a form?",
        "What is at the bottom?", "Count the rectangles.",
        "What text do you see?", "Describe the structure.",
    ][:n_prompts]
    msgs = [[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": img}},
        {"type": "text", "text": q + "\nAnswer concisely."}]}] for q in questions]

    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    llm.chat(msgs[:1], sp1)  # warmup (compile/caches)

    prefill_ms = []
    for m in msgs:
        t = time.perf_counter()
        llm.chat([m], sp1)
        prefill_ms.append(1000.0 * (time.perf_counter() - t))

    res = {
        "arm": arm_key, "label": cfg["label"], "desc": cfg["desc"],
        "model": MODEL, "quantization": cfg["quant"],
        "vision_excluded_from_quant": cfg["exclude_vision"],
        "resolution_px": resolution, "n_prompts": len(msgs), "single_stream": True,
        "ttft_ms_mean": round(float(np.mean(prefill_ms)), 2),
        "ttft_ms_median": round(float(np.median(prefill_ms)), 2),
        "ttft_ms_min": round(float(np.min(prefill_ms)), 2),
        "ttft_ms_all": [round(x, 2) for x in prefill_ms],
        "linear_method_counts": buckets,
        "linear_method_examples": examples,
    }
    frag = os.path.join(OUTDIR, f"vision_quant_uncap_{arm_key}.json")
    json.dump(res, open(frag, "w"), indent=2)
    print(f"\n[arm {arm_key}/{cfg['label']}] TTFT mean={res['ttft_ms_mean']}ms "
          f"median={res['ttft_ms_median']}ms")
    print(f"  vision linear methods : {buckets['visual']}")
    print(f"  language linear methods: {buckets['language']}")
    print(f"  -> {frag}")

    del llm
    gc.collect()


def merge():
    arms = {}
    for k in ("A", "B", "C"):
        frag = os.path.join(OUTDIR, f"vision_quant_uncap_{k}.json")
        if os.path.exists(frag):
            arms[k] = json.load(open(frag))
    out = {"experiment": "vision_tower_quant_uncap_prefill",
           "model": MODEL, "device": "RTX 5090 (sm_120)", "runtime": "vLLM 0.22",
           "metric": "single-stream prefill TTFT (ms), synthetic 768px page, max_tokens=1",
           "synthetic_image_only": True, "arms": arms}
    A = arms.get("A", {}).get("ttft_ms_mean")
    B = arms.get("B", {}).get("ttft_ms_mean")
    C = arms.get("C", {}).get("ttft_ms_mean")
    if B is not None and C is not None:
        out["vision_uncap_B_to_C_ms"] = round(B - C, 2)
        out["vision_uncap_B_to_C_pct"] = round(100.0 * (B - C) / B, 1)
    if A is not None and C is not None:
        out["bf16_to_fullfp8_gap_ms"] = round(A - C, 2)
    if A is not None and B is not None and C is not None and (A - C) != 0:
        out["vision_share_of_total_gap_pct"] = round(100.0 * (B - C) / (A - C), 1)
    json.dump(out, open(FINAL, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n-> {FINAL}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=768)
    a = ap.parse_args()
    if a.merge:
        merge()
    elif a.arm:
        run_arm(a.arm, n_prompts=a.n_prompts, resolution=a.resolution)
    else:
        ap.error("pass --arm {A,B,C} or --merge")
