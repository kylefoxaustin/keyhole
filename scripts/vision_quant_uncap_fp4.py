#!/usr/bin/env python3
"""
vision_quant_uncap_fp4.py — FP4 sibling of vision_quant_uncap.py. Isolate the
VISION-TOWER quantization contribution to prefill (TTFT) for Qwen3-VL-4B-Instruct
on RTX 5090 (sm_120), vLLM 0.22, with the LANGUAGE MODEL quantized to NVFP4.

Hypothesis: NVFP4 speeds the LM MORE than FP8, so the un-quantized BF16 vision
tower becomes a LARGER relative share of the remaining prefill — i.e. quantizing
the vision tower should matter MORE at FP4 than the ~8% / ~34%-of-gap seen at FP8.

  A. BF16 everything                      (no quant; ceiling reference)
  B. NVFP4 LM, vision tower BF16          (pre-quantized checkpoint, visual.* in `ignore`)
  C. NVFP4 LM + FP8 vision tower (MIXED)  (same ckpt; visual.* forced to dynamic FP8)

=== WHY THIS PATH (controlled-isolation method, clearly labeled) ===

Path 1 (dynamic NVFP4 mirroring the FP8 A/B/C, one base model, vision scope
toggled by monkeypatch) is NOT available in vLLM 0.22: ModelOptNvFp4LinearMethod
.create_weights() raises "dynamic quantization is not supported" unless
is_checkpoint_nvfp4_serialized=True. There is no online/dynamic NVFP4 weight path.
So FP4 REQUIRES a pre-quantized checkpoint (Path 2).

Arm B uses the pre-quantized NVFP4 checkpoint `nm-testing/Qwen3-VL-4B-Instruct-NVFP4`.
Its config.json quantization_config is compressed-tensors `nvfp4-pack-quantized`,
with EVERY `model.visual.*` linear listed in `ignore` -> the vision tower loads as
plain BF16 (UnquantizedLinearMethod), LM linears run NVFP4 (CompressedTensors W4A4
NVFP4, FlashInferCutlass GEMM on sm_120). That is exactly arm B's "FP4 LM, BF16
vision".

Arm C needs the vision tower ALSO low-precision. A checkpoint's pre-quantized
weights can't be excluded at load, and the inverse (forcing checkpoint-ignored
BF16 vision into *NVFP4*) is impossible: there is no NVFP4 weight data for those
layers and no dynamic NVFP4 path to make it at load time. The cleanest VERIFIABLE
low-precision vision we can apply on top of the NVFP4-LM checkpoint is DYNAMIC FP8
(online), which the FP8 run already used for its arm-C vision tower. So arm C is a
MIXED "FP4 LM + FP8 vision" arm: we monkeypatch CompressedTensorsConfig
.get_quant_method so that any `visual.` LinearBase that the checkpoint would leave
Unquantized instead gets an Fp8OnlineLinearMethod (the same online-FP8 method the
FP8 script's arm C used). LM layers are untouched -> they stay NVFP4.

CAVEAT (stated up front): arm C is "FP4 LM + FP8 vision", NOT pure-FP4 vision.
B and C share the SAME LM (identical NVFP4 checkpoint), so B->C isolates the
vision tower exactly — the only thing that changes is the vision linears going
BF16 -> FP8. The vision delta is therefore a LOWER BOUND on a pure-FP4-vision
uncap (FP4 vision would be at least as fast as FP8 vision), which is the
conservative direction for the hypothesis.

After load we INSPECT the live model (collective_rpc into the worker) and bucket
linear quant_method classes by visual/language to PROVE: (B) LM=NVFP4, vision=BF16;
(C) LM=NVFP4, vision=FP8 online. Confirms vision GEMMs in C run low-precision, not
silently dequantized.

Privacy: synthetic 768px image only. No personal corpus, no commits, no sends.
One arm per invocation (clean VRAM). --merge aggregates the three fragments.

  CUDA_HOME=/home/kyle/cuda-12.9 VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  ~/.virtualenvs/vllm_fp4/bin/python scripts/vision_quant_uncap_fp4.py --arm A
  ... --arm B ;  ... --arm C ;  ... --merge
"""
import argparse, base64, gc, io, json, os, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "data", "output")
BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
NVFP4_MODEL = "nm-testing/Qwen3-VL-4B-Instruct-NVFP4"
FINAL = os.path.join(OUTDIR, "vision_quant_uncap_fp4.json")

ARMS = {
    "A": dict(label="bf16_full", model=BASE_MODEL, quant=None, fp8_vision=False,
              desc="BF16 everything (no quant) — ceiling"),
    "B": dict(label="nvfp4_lm_only", model=NVFP4_MODEL, quant=None, fp8_vision=False,
              desc="NVFP4 language-model (pre-quant ckpt), vision tower BF16 (ckpt `ignore`)"),
    "C": dict(label="nvfp4_lm_fp8_vision", model=NVFP4_MODEL, quant=None, fp8_vision=True,
              desc="NVFP4 LM + FP8 vision tower (MIXED: visual.* forced to dynamic FP8)"),
}


def synthetic_page(longest=768):
    """A synthetic 'page' (white bg, drawn text/rects). Prefill latency is
    content-independent — this carries NO personal data."""
    from PIL import Image, ImageDraw
    W = longest
    H = int(longest * 1.3)
    if H > longest:
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


def install_fp8_vision_on_compressed_tensors():
    """Arm C only. Monkeypatch CompressedTensorsConfig.get_quant_method so that
    any `visual.` LinearBase the NVFP4 checkpoint leaves Unquantized (it's in the
    ckpt `ignore` list) instead receives a DYNAMIC FP8 online linear method — the
    same online-FP8 path the FP8 script used for its arm-C vision tower. LM layers
    keep their checkpoint NVFP4 scheme untouched.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as ct)
    from vllm.model_executor.layers.quantization import fp8 as fp8mod
    from vllm.model_executor.layers.linear import (
        LinearBase, UnquantizedLinearMethod)

    orig = ct.CompressedTensorsConfig.get_quant_method
    # one shared dynamic (online) fp8 config — activation_scheme=dynamic, no ckpt
    fp8_cfg = fp8mod.Fp8Config(is_checkpoint_fp8_serialized=False,
                               activation_scheme="dynamic")

    def patched(self, layer, prefix):
        m = orig(self, layer, prefix)
        if (isinstance(layer, LinearBase) and "visual." in prefix
                and isinstance(m, UnquantizedLinearMethod)):
            if not hasattr(self, "_fp8_vision_layers"):
                self._fp8_vision_layers = []
            self._fp8_vision_layers.append(prefix)
            om = fp8mod.Fp8OnlineLinearMethod(fp8_cfg)
            # mirror the dispatch in Fp8Config.get_quant_method
            try:
                from vllm.model_executor.layers.quantization.fp8 import (
                    get_marlin_input_dtype)
                om.marlin_input_dtype = get_marlin_input_dtype(prefix)
            except Exception:
                pass
            return om
        return m

    ct.CompressedTensorsConfig.get_quant_method = patched


def _inspect_worker(self):
    """Runs INSIDE the worker process via collective_rpc (bound-method form).
    Buckets linear-layer quant_method classes by vision/language."""
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
    """Direct attribute paths first, then collective_rpc into the worker."""
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
    if cfg["fp8_vision"]:
        install_fp8_vision_on_compressed_tensors()

    from vllm import LLM, SamplingParams
    import numpy as np

    kw = dict(model=cfg["model"], max_model_len=max_model_len,
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
    llm.chat(msgs[:1], sp1)  # warmup

    prefill_ms = []
    for m in msgs:
        t = time.perf_counter()
        llm.chat([m], sp1)
        prefill_ms.append(1000.0 * (time.perf_counter() - t))

    res = {
        "arm": arm_key, "label": cfg["label"], "desc": cfg["desc"],
        "model": cfg["model"], "quantization": cfg["quant"],
        "fp8_vision_overlay": cfg["fp8_vision"],
        "resolution_px": resolution, "n_prompts": len(msgs), "single_stream": True,
        "ttft_ms_mean": round(float(np.mean(prefill_ms)), 2),
        "ttft_ms_median": round(float(np.median(prefill_ms)), 2),
        "ttft_ms_min": round(float(np.min(prefill_ms)), 2),
        "ttft_ms_all": [round(x, 2) for x in prefill_ms],
        "linear_method_counts": buckets,
        "linear_method_examples": examples,
    }
    frag = os.path.join(OUTDIR, f"vision_quant_uncap_fp4_{arm_key}.json")
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
        frag = os.path.join(OUTDIR, f"vision_quant_uncap_fp4_{k}.json")
        if os.path.exists(frag):
            arms[k] = json.load(open(frag))
    out = {"experiment": "vision_tower_quant_uncap_prefill_FP4",
           "model_bf16": BASE_MODEL, "model_nvfp4": NVFP4_MODEL,
           "device": "RTX 5090 (sm_120)", "runtime": "vLLM 0.22",
           "metric": "single-stream prefill TTFT (ms), synthetic 768px page, max_tokens=1",
           "synthetic_image_only": True,
           "method_note": (
               "FP4 has no dynamic/online path in vLLM 0.22 (NVFP4 create_weights "
               "rejects dynamic quant), so a pre-quantized NVFP4 checkpoint is used "
               "for the LM. Arm C is MIXED 'FP4 LM + FP8 vision': the vision tower "
               "(ckpt-ignored, BF16) is forced to dynamic FP8 online — the same "
               "online-FP8 method the FP8 run used for its arm-C vision. B and C "
               "share the identical NVFP4 LM, so B->C isolates ONLY the vision tower. "
               "Vision delta is a LOWER BOUND on pure-FP4 vision."),
           "arms": arms}
    A = arms.get("A", {}).get("ttft_ms_mean")
    B = arms.get("B", {}).get("ttft_ms_mean")
    C = arms.get("C", {}).get("ttft_ms_mean")
    if B is not None and C is not None:
        out["vision_uncap_B_to_C_ms"] = round(B - C, 2)
        out["vision_uncap_B_to_C_pct"] = round(100.0 * (B - C) / B, 1)
    if A is not None and C is not None:
        out["bf16_to_C_gap_ms"] = round(A - C, 2)
    if A is not None and B is not None and C is not None and (A - C) != 0:
        out["vision_share_of_total_gap_pct"] = round(100.0 * (B - C) / (A - C), 1)
    # FP8 baseline for direct comparison (from vision_quant_uncap.json)
    out["fp8_reference"] = {
        "A_ms": 28.69, "B_ms": 24.83, "C_ms": 22.81,
        "vision_uncap_ms": 2.02, "vision_uncap_pct": 8.1,
        "vision_share_of_gap_pct": 34.4}
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
