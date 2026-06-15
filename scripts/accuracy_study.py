#!/usr/bin/env python3
"""
accuracy_study.py — END-TO-END ANSWER CORRECTNESS for the keyhole visual-RAG
(PixelRAG) work, on a PUBLIC document-VQA benchmark with GOLD answers.

We already have retrieval accuracy + latency. This fills the missing piece:
given the CORRECT page, how often is the reader's ANSWER right (ANLS vs gold)?

Two axes:
  1. Precision  — Qwen3-VL-4B reader at BF16 / FP8 / NVFP4, IMAGE input.
                  (Turns the "NVFP4 agrees w/ BF16 only 62.5%" self-consistency
                   flag into a real correctness number vs ground truth.)
  2. Pipeline   — BF16 reader, IMAGE (pixel/visual-RAG) vs OCR TEXT (text-RAG),
                  holding the page correct. Isolates image-vs-text answering.

Benchmark: lmms-lab/DocVQA validation (gold `answers`, embedded page images).
  The vidore/*_test_subsampled sets have answer=None (verified) -> NOT used.
Metric: ANLS (Average Normalized Levenshtein Similarity, threshold 0.5) —
  the standard DocVQA metric, implemented locally (no LLM judge).

Pipeline (decoupled so the OCR CPU deps never touch the vLLM env):
  # 1. prep: download DocVQA-val, take a slice, dump page images + gold meta
  ~/.virtualenvs/vllm_fp4/bin/python scripts/accuracy_study.py prep --n 200
  # 2. OCR the page images (separate CPU venv; rapidocr-onnxruntime)
  ~/.virtualenvs/ocr_tmp/bin/python scripts/accuracy_study.py ocr
  # 3. reader runs (one precision+input per invocation -> clean VRAM)
  CUDA_HOME=/home/kyle/cuda-12.9 VLLM_USE_FLASHINFER_SAMPLER=0 \
    ~/.virtualenvs/vllm_fp4/bin/python scripts/accuracy_study.py reader \
      --model Qwen/Qwen3-VL-4B-Instruct --label bf16 --input image
  # 4. aggregate fragments -> data/output/accuracy_study.json
  ~/.virtualenvs/vllm_fp4/bin/python scripts/accuracy_study.py aggregate

Public data only. No personal corpus, no external APIs, no remote push.
"""
import argparse, base64, glob, io, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(REPO, "data", "frames", "accuracy_study")  # scratch (gitignored frames dir)
IMG_DIR = os.path.join(WORK, "images")
META_PATH = os.path.join(WORK, "meta.json")
OCR_PATH = os.path.join(WORK, "ocr.json")
FRAG_GLOB = os.path.join(WORK, "frag_*.json")
OUT_PATH = os.path.join(REPO, "data", "output", "accuracy_study.json")
DATASET = "lmms-lab/DocVQA"
SUBSET = "DocVQA"  # vs InfographicVQA
ANSWER_PROMPT = "Answer with just the answer, as briefly as possible."


# ---------------------------------------------------------------- ANLS metric
def _nl(a, b):
    """Normalized Levenshtein distance in [0,1] between lowercased/stripped strings."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


def anls_score(pred, golds, tau=0.5):
    """Standard DocVQA ANLS for one question: best (1 - NL) over gold answers,
    zeroed if below threshold tau. golds is a list of accepted answers."""
    if not golds:
        return None
    best = max(1.0 - _nl(pred, g) for g in golds)
    return best if best >= tau else 0.0


# ---------------------------------------------------------------- 1. prep
def cmd_prep(args):
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image
    os.makedirs(IMG_DIR, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files(DATASET, repo_type="dataset")
    shards = sorted(f for f in files
                    if f.startswith(f"{SUBSET}/validation-") and f.endswith(".parquet"))
    print(f"[prep] {DATASET}/{SUBSET} validation shards: {len(shards)}")
    import pyarrow.parquet as pq
    records, n = [], args.n
    for sh in shards:
        if len(records) >= n:
            break
        path = hf_hub_download(DATASET, sh, repo_type="dataset")
        t = pq.read_table(path).to_pylist()
        for row in t:
            if len(records) >= n:
                break
            qid = row["questionId"]
            img = row["image"]  # {'bytes':..., 'path':...}
            im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            fn = os.path.join(IMG_DIR, f"{qid}.png")
            im.save(fn, format="PNG")
            records.append({
                "questionId": qid,
                "question": row["question"],
                "answers": list(row["answers"]) if row["answers"] else [],
                "image": fn,
                "docId": row.get("docId"),
            })
    # require populated gold answers
    bad = [r for r in records if not r["answers"]]
    meta = {"dataset": f"{DATASET} ({SUBSET}) validation", "n_questions": len(records),
            "n_missing_gold": len(bad), "records": records}
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    json.dump(meta, open(META_PATH, "w"), indent=2)
    print(f"[prep] wrote {len(records)} questions (missing gold: {len(bad)}) -> {META_PATH}")
    print(f"[prep] images -> {IMG_DIR}")


# ---------------------------------------------------------------- 2. ocr
def cmd_ocr(args):
    """Run rapidocr-onnxruntime over each page image; save concatenated text.
    Run with the OCR venv: ~/.virtualenvs/ocr_tmp/bin/python."""
    from rapidocr_onnxruntime import RapidOCR
    meta = json.load(open(META_PATH))
    engine = RapidOCR()
    ocr = {}
    for i, r in enumerate(meta["records"]):
        res, _ = engine(r["image"])
        if res:
            # res: list of [box, text, score]; preserve reading order as returned
            text = " ".join(seg[1] for seg in res)
        else:
            text = ""
        ocr[str(r["questionId"])] = text
        if (i + 1) % 25 == 0:
            print(f"[ocr] {i+1}/{len(meta['records'])}")
    json.dump({"engine": "rapidocr-onnxruntime", "text": ocr}, open(OCR_PATH, "w"), indent=2)
    empty = sum(1 for v in ocr.values() if not v.strip())
    print(f"[ocr] done {len(ocr)} pages, empty={empty} -> {OCR_PATH}")


# ---------------------------------------------------------------- 3. reader
def _data_uri(path, R):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((R, R))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def cmd_reader(args):
    from vllm import LLM, SamplingParams
    meta = json.load(open(META_PATH))
    recs = meta["records"]
    ocr_text = None
    if args.input == "text":
        ocr = json.load(open(OCR_PATH))
        ocr_text = ocr["text"]

    llm = LLM(model=args.model, max_model_len=8192, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.85, enforce_eager=False, disable_log_stats=True,
              enable_prefix_caching=False)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    msgs = []
    for r in recs:
        if args.input == "image":
            content = [
                {"type": "image_url", "image_url": {"url": _data_uri(r["image"], args.resolution)}},
                {"type": "text", "text": f"{r['question']}\n{ANSWER_PROMPT}"},
            ]
        else:  # text
            page = ocr_text.get(str(r["questionId"]), "")
            content = [{"type": "text",
                        "text": f"Document text:\n{page}\n\nQuestion: {r['question']}\n{ANSWER_PROMPT}"}]
        msgs.append([{"role": "user", "content": content}])

    outs = llm.chat(msgs, sp)
    answers, scores = [], []
    n_empty = 0
    for r, o in zip(recs, outs):
        pred = o.outputs[0].text.strip()
        if not pred:
            n_empty += 1
        s = anls_score(pred, r["answers"])
        answers.append({"qid": r["questionId"], "pred": pred[:200], "gold": r["answers"], "anls": s})
        if s is not None:
            scores.append(s)

    anls = sum(scores) / len(scores) if scores else None
    frag = {"label": args.label, "model": args.model, "input": args.input,
            "resolution_px": args.resolution if args.input == "image" else None,
            "max_tokens": args.max_tokens, "n_scored": len(scores), "n_empty_pred": n_empty,
            "anls": round(anls, 4) if anls is not None else None, "answers": answers}
    frag_path = os.path.join(WORK, f"frag_{args.label}_{args.input}.json")
    json.dump(frag, open(frag_path, "w"), indent=2)
    print(f"[reader {args.label}/{args.input}] ANLS={frag['anls']} "
          f"n={len(scores)} empty_pred={n_empty} -> {frag_path}")


# ---------------------------------------------------------------- 4. aggregate
def cmd_aggregate(args):
    meta = json.load(open(META_PATH))
    ocr_meta = json.load(open(OCR_PATH)) if os.path.exists(OCR_PATH) else {}
    configs = []
    for fp in sorted(glob.glob(FRAG_GLOB)):
        f = json.load(open(fp))
        configs.append({"label": f["label"], "model": f["model"], "input": f["input"],
                        "resolution_px": f.get("resolution_px"), "anls": f["anls"],
                        "n_scored": f["n_scored"], "n_empty_pred": f["n_empty_pred"]})
    out = {"dataset": meta["dataset"], "metric": "ANLS (threshold 0.5)",
           "n_questions": meta["n_questions"],
           "ocr_engine": ocr_meta.get("engine"),
           "answer_prompt": ANSWER_PROMPT, "decoding": "temperature=0",
           "configs": configs}
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n[aggregate] -> {OUT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--n", type=int, default=200)
    sub.add_parser("ocr")
    r = sub.add_parser("reader")
    r.add_argument("--model", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--input", choices=["image", "text"], required=True)
    r.add_argument("--resolution", type=int, default=1024)
    r.add_argument("--max_tokens", type=int, default=32)
    sub.add_parser("aggregate")
    args = ap.parse_args()
    {"prep": cmd_prep, "ocr": cmd_ocr, "reader": cmd_reader, "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
