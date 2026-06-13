#!/usr/bin/env python3
"""
skippy_precision_arm.py — the precision arm of the PixelRAG bake-off: how reader
NUMERIC PRECISION affects the visual-RAG reader at the retrieval knee. This is the
bridge to the precision deck — the reader's image-token PREFILL is exactly the
compute-bound regime where FP4 wins.

At the knee resolution, feed each query its GOLD page image (isolating the reader from
retrieval noise) and measure, per precision:
  - prefill latency (image-token prefill, compute-bound)
  - decode latency (per-token)
  - model VRAM footprint
  - answer agreement vs BF16 (normalized-match rate = quality preservation proxy)

BF16 and FP8 (transformers FineGrainedFP8Config, native sm_120) are MEASURED here.
FP4/NVFP4 of this VLM needs qutlass/vLLM (separate env) — so it is PROJECTED from the
deck's measured LLM numbers (decode 2.24x, prefill 3.59x vs BF16), since the reader's
image-prefill is the same compute-bound GEMM regime.

Output: data/output/skippy_precision_arm.json (aggregate only — no personal content).
Run:  python scripts/skippy_precision_arm.py --resolution 768 --sample 16
"""
import argparse, json, os, re, time
import numpy as np
import torch
from PIL import Image

SCRATCH = os.path.expanduser("~/skippy_pixelrag")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RID = "Qwen/Qwen3-VL-4B-Instruct"
# measured on the 5090 in the precision deck (same-base Qwen3-8B, vLLM): FP4 vs BF16
DECK_FP4 = {"decode_x": 2.24, "prefill_x": 3.59, "source": "precision_5090_vllm_fp4_vs_int.json"}
DECK_FP8 = {"decode_x": 1.55, "prefill_x": 1.71}


def norm(s):
    return re.sub(r"\s+", " ", s.lower().strip())[:200]


def resize(path, R):
    im = Image.open(path).convert("RGB"); im.thumbnail((R, R))
    p = f"/tmp/parm_{os.getpid()}.png"; im.save(p); return p


def run_precision(label, kwargs, samp, proc):
    from transformers import AutoModelForImageTextToText
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = AutoModelForImageTextToText.from_pretrained(RID, device_map="cuda", **kwargs)
    pre_ms, dec_ms, ans = [], [], []
    for img, q in samp:
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                {"type": "text", "text": q + "\nAnswer concisely."}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        n_in = inp["input_ids"].shape[1]
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize(); pre_ms.append(1000 * (time.time() - t))
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=32, do_sample=False)
        torch.cuda.synchronize(); dec_ms.append(1000 * (time.time() - t - pre_ms[-1] / 1000) / 31)
        ans.append(norm(proc.batch_decode(o[:, n_in:], skip_special_tokens=True)[0]))
    vram = torch.cuda.max_memory_allocated() / 1e9
    del model; torch.cuda.empty_cache()
    return {"prefill_ms": round(float(np.mean(pre_ms)), 1),
            "decode_ms_per_tok": round(float(np.mean(dec_ms)), 2),
            "vram_gb": round(vram, 2)}, ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--sample", type=int, default=16)
    args = ap.parse_args()

    man = {r["id"]: r for r in json.load(open(os.path.join(SCRATCH, "meta", "manifest.json")))["records"]}
    qd = json.load(open(os.path.join(SCRATCH, "meta", "queries.json")))["queries"]
    qd = [q for q in qd if q["gold_id"] in man][:args.sample]
    samp = [(resize(man[q["gold_id"]]["img"], args.resolution), q["query"]) for q in qd]

    from transformers import AutoProcessor, FineGrainedFP8Config
    proc = AutoProcessor.from_pretrained(RID)

    res = {"__meta__": {"reader": RID, "resolution_px": args.resolution, "sample": len(samp),
                        "host": "RTX 5090 sm_120", "note": "reader fed GOLD page (isolates reader from "
                        "retrieval); aggregate only, no personal content."}}

    bf16, ans_bf16 = run_precision("bf16", dict(dtype=torch.bfloat16), samp, proc)
    res["bf16"] = bf16
    try:
        fp8, ans_fp8 = run_precision("fp8", dict(quantization_config=FineGrainedFP8Config()), samp, proc)
        fp8["answer_agreement_vs_bf16"] = round(np.mean([a == b for a, b in zip(ans_fp8, ans_bf16)]), 3)
        fp8["prefill_speedup_vs_bf16"] = round(bf16["prefill_ms"] / fp8["prefill_ms"], 2)
        fp8["vram_reduction_vs_bf16"] = round(bf16["vram_gb"] / fp8["vram_gb"], 2)
        res["fp8_measured"] = fp8
    except Exception as e:
        res["fp8_measured"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}

    # FP4 projected from the deck's measured LLM numbers (compute-bound prefill regime)
    res["fp4_projected"] = {
        "from": DECK_FP4["source"],
        "prefill_ms_projected": round(bf16["prefill_ms"] / DECK_FP4["prefill_x"], 1),
        "decode_ms_per_tok_projected": round(bf16["decode_ms_per_tok"] / DECK_FP4["decode_x"], 2),
        "prefill_speedup_x": DECK_FP4["prefill_x"], "decode_speedup_x": DECK_FP4["decode_x"],
        "note": "projected: the reader's image-token prefill is the same compute-bound GEMM regime "
                "where the deck measured FP4 = 3.59x prefill / 2.24x decode vs BF16 on the 5090. "
                "Native NVFP4 VLM serving (qutlass/vLLM) is the follow-up to MEASURE it directly."}

    out = os.path.join(REPO, "data", "output", "skippy_precision_arm.json")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2)); print("wrote", out)


if __name__ == "__main__":
    main()
