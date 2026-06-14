#!/usr/bin/env python3
"""
skippy_precision_vllm.py — DIRECT measurement of the visual-RAG reader at BF16 / FP8 /
NVFP4 on vLLM (RTX 5090, sm_120). The transformers FineGrainedFP8 path hit a kernels
version wall; this measures the real thing on vLLM with pre-quantized checkpoints.

One precision per invocation (clean VRAM). At the knee resolution, each query's GOLD
page image + query goes to the reader; we read vLLM's per-request metrics for TTFT
(prefill, compute-bound — the image-token regime) and decode throughput, plus the
generated answer (for cross-precision agreement). Writes a per-precision fragment
data/output/skippy_precision_vllm_<label>.json (aggregate only — no personal content).

Run (canary FP8 first):
  CUDA_HOME=/home/kyle/cuda-12.9 ~/.virtualenvs/vllm_fp4/bin/python scripts/skippy_precision_vllm.py \
     --model Qwen/Qwen3-VL-4B-Instruct-FP8 --label fp8 --resolution 768 --sample 16
"""
import argparse, base64, io, json, os
import numpy as np
from PIL import Image

SCRATCH = os.path.expanduser("~/skippy_pixelrag")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_uri(path, R):
    im = Image.open(path).convert("RGB"); im.thumbnail((R, R))
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--sample", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=32)
    args = ap.parse_args()

    man = {r["id"]: r for r in json.load(open(os.path.join(SCRATCH, "meta", "manifest.json")))["records"]}
    qd = json.load(open(os.path.join(SCRATCH, "meta", "queries.json")))["queries"]
    qd = [q for q in qd if q["gold_id"] in man][:args.sample]

    import time
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=8192, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.85, enforce_eager=False, disable_log_stats=True,
              enable_prefix_caching=False)  # clean prefill timing (deck discipline)

    msgs = [[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri(man[q["gold_id"]]["img"], args.resolution)}},
        {"type": "text", "text": q["query"] + "\nAnswer concisely."}]}] for q in qd]

    DEC = 128
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    spN = SamplingParams(temperature=0.0, max_tokens=DEC + 1, ignore_eos=True)  # force full decode
    # warm up (compile/caches) so the first real request isn't penalised
    llm.chat(msgs[:1], sp1)

    prefill_ms, dec_per_tok, answers = [], [], []
    for m in msgs:
        t = time.perf_counter(); llm.chat([m], sp1); t1 = time.perf_counter() - t   # prefill + 1 tok
        t = time.perf_counter(); o = llm.chat([m], spN); tN = time.perf_counter() - t  # prefill + N+1
        n_out = len(o[0].outputs[0].token_ids)
        prefill_ms.append(1000 * t1)
        if n_out > 1:
            dec_per_tok.append(1000 * (tN - t1) / max(1, n_out - 1))
        answers.append(o[0].outputs[0].text.strip().lower()[:200])

    res = {"label": args.label, "model": args.model, "resolution_px": args.resolution,
           "sample": len(qd), "single_stream": True,
           "ttft_ms_mean": round(float(np.mean(prefill_ms)), 1),
           "decode_ms_per_tok_mean": round(float(np.mean(dec_per_tok)), 2) if dec_per_tok else None,
           "answers": answers}  # answers transient (gitignored file); only agreement RATE used downstream
    out = os.path.join(REPO, "data", "output", f"skippy_precision_vllm_{args.label}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"[{args.label}] TTFT={res['ttft_ms_mean']}ms  decode={res['decode_ms_per_tok_mean']}ms/tok  -> {out}")


if __name__ == "__main__":
    main()
