#!/usr/bin/env python3
"""
bench_precision_vllm.py — single-stream prefill/decode throughput for ONE model+quant
on the RTX 5090 via the vLLM offline LLM API. Companion to bench_precision_5090.sh
(llama.cpp); this is the MATURE-FP4-KERNEL runtime the llama.cpp finding flagged as
the missing comparison.

The honest question this answers: within ONE runtime (vLLM), holding the model fixed
(Qwen3-8B), does NVFP4 beat FP8 / BF16? That isolates the *precision* effect from the
*runtime* effect (llama.cpp's NVFP4 kernels are new; vLLM's are NVIDIA-optimized).

Measures, single-stream (batch=1, greedy, ignore_eos):
  - prefill regimes pp128..pp4096 : input_len / latency(input_len, out=1)
  - decode tg256                  : 256 / (lat(128,257) - lat(128,1))   [prefill subtracted]

Footprint (weight bytes) is taken from the checkpoint on disk by the orchestrator, NOT
from nvidia-smi: vLLM pre-reserves gpu_memory_utilization*VRAM for the KV pool, so
runtime VRAM is ~constant and not a footprint signal.

Usage:
  bench_precision_vllm.py --label bf16  --model Qwen/Qwen3-8B
  bench_precision_vllm.py --label fp8   --model Qwen/Qwen3-8B        --quantization fp8
  bench_precision_vllm.py --label nvfp4 --model nvidia/Qwen3-8B-NVFP4 --quantization modelopt_fp4
"""
import argparse, json, os, time, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "data/output/precision_5090_vllm_runs")

PREFILL_LENS = [128, 512, 1024, 2048, 4096]
DECODE_INPUT = 128
DECODE_OUT = 256
ITERS = 3


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    from vllm import LLM, SamplingParams

    print(f"[{args.label}] loading {args.model} (quant={args.quantization}, dtype={args.dtype}) ...",
          flush=True)
    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
        enable_prefix_caching=False,  # CRITICAL: else repeated prompts hit the cache and
                                      # "prefill" latency collapses to fixed overhead (invalid).
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()

    # Long token pool so each iteration can take a DISJOINT window — defends prefill timing
    # against any residual caching even with prefix caching off.
    base = tok.encode("The quick brown fox jumps over the lazy dog. " * 4000)
    need = max(PREFILL_LENS) + (ITERS + 2) * max(PREFILL_LENS)
    if len(base) < need:
        base = base * ((need // len(base)) + 1)

    def ids_of(n, offset=0):
        return base[offset:offset + n]

    def latency(input_len, output_len, iters=ITERS, unique=False):
        sp = SamplingParams(max_tokens=output_len, temperature=0.0, ignore_eos=True)
        llm.generate({"prompt_token_ids": ids_of(input_len, 0)}, sp, use_tqdm=False)  # warmup
        ts = []
        for k in range(iters):
            off = (k + 1) * input_len if unique else 0
            t0 = time.perf_counter()
            llm.generate({"prompt_token_ids": ids_of(input_len, off)}, sp, use_tqdm=False)
            ts.append(time.perf_counter() - t0)
        return median(ts)

    prefill_tok_s, prefill_lat = {}, {}
    for n in PREFILL_LENS:
        lat = latency(n, 1, unique=True)
        prefill_lat[f"pp{n}"] = round(lat, 5)
        prefill_tok_s[f"pp{n}"] = round(n / lat, 1)
        print(f"[{args.label}] pp{n}: {prefill_tok_s[f'pp{n}']} tok/s ({lat*1e3:.1f} ms)", flush=True)

    lat_dec = latency(DECODE_INPUT, DECODE_OUT + 1)
    lat_pre = latency(DECODE_INPUT, 1)
    dec_dt = max(lat_dec - lat_pre, 1e-6)
    decode_tok_s = round(DECODE_OUT / dec_dt, 1)
    print(f"[{args.label}] tg{DECODE_OUT}: {decode_tok_s} tok/s "
          f"({dec_dt*1e3:.1f} ms for {DECODE_OUT} tok)", flush=True)

    doc = {
        "label": args.label,
        "model": args.model,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "runtime": "vllm 0.22.0",
        "stream": "single (batch=1, greedy, ignore_eos)",
        "iters": ITERS,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": {f"tg{DECODE_OUT}": decode_tok_s},
        "raw_latency_s": {"prefill": prefill_lat,
                          f"decode_{DECODE_INPUT}_{DECODE_OUT+1}": round(lat_dec, 5),
                          f"prefill_{DECODE_INPUT}_1": round(lat_pre, 5)},
    }
    out = os.path.join(OUTDIR, f"{args.label}_vllm.json")
    with open(out, "w") as fp:
        json.dump(doc, fp, indent=2)
    print(f"[{args.label}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
