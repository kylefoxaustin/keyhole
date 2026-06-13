#!/usr/bin/env python3
"""
skippy_query_gen.py — generate retrieval ground-truth for the Skippy corpus, LOCALLY.

For a stride-sampled set of pages per class, the local Qwen3-VL-4B reader writes ONE
specific factual question answerable only from that page → {query, gold_id, class}.
This is the gold (query ↔ relevant page) map the bake-off retrieves against. Runs
entirely on-box — no Claude/OpenAI, no content leaves the machine. Reads/writes only
the scratch dir (~/skippy_pixelrag), never the repo.

Run:  python scripts/skippy_query_gen.py --per_class 80
"""
import argparse, json, os, time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

SCRATCH = os.path.expanduser("~/skippy_pixelrag")
PROMPT = ("You are reading one page of a technical document. Write exactly ONE specific, "
          "factual question that can be answered ONLY from the content on THIS page "
          "(reference a concrete detail: a value, name, parameter, table entry, or fact). "
          "Output only the question, nothing else.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=80)
    ap.add_argument("--max_new", type=int, default=48)
    args = ap.parse_args()

    man = json.load(open(os.path.join(SCRATCH, "meta", "manifest.json")))
    recs = man["records"]
    by_cls = {}
    for r in recs:
        by_cls.setdefault(r["class"], []).append(r)
    sampled = []
    for cls, rs in by_cls.items():
        stride = max(1, len(rs) // args.per_class)
        sampled += rs[::stride][:args.per_class]
    print(f"sampling {len(sampled)} pages across {len(by_cls)} classes")

    rid = "Qwen/Qwen3-VL-4B-Instruct"
    proc = AutoProcessor.from_pretrained(rid)
    model = AutoModelForImageTextToText.from_pretrained(rid, dtype="auto", device_map="cuda")

    out, t0 = [], time.time()
    for i, r in enumerate(sampled):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": r["img"]},
            {"type": "text", "text": PROMPT}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        n_in = inp["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=args.max_new, do_sample=False)
        q = proc.batch_decode(o[:, n_in:], skip_special_tokens=True)[0].strip().replace("\n", " ")
        out.append({"qid": f"q{i:04d}", "query": q, "gold_id": r["id"], "class": r["class"]})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(sampled)}  ({time.time()-t0:.0f}s)")

    dst = os.path.join(SCRATCH, "meta", "queries.json")
    json.dump({"n": len(out), "per_class": args.per_class, "queries": out}, open(dst, "w"), indent=2)
    print(f"wrote {len(out)} queries -> {dst} (scratch, never committed)  in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
