#!/usr/bin/env python3
"""
skippy_colqwen.py — PixelRAG's OWN embedder vs our 2B base.

PixelRAG embeds with Qwen3-VL-4B-Instruct + a screenshot-LoRA, last-token pooled +
L2-normalized to a SINGLE vector (confirmed in their embed code: pooling_type=LAST).
This replicates that embedder and compares its pixel retrieval to our Qwen3-VL-Embedding-2B
base on the identical Skippy corpus + queries — i.e. does PixelRAG's fine-tune actually
help on real docs, vs the simpler off-the-shelf 2B?

Aggregate metrics only. Run: python scripts/skippy_colqwen.py --resolution 768 --max 0
"""
import argparse, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRATCH = os.path.expanduser("~/skippy_pixelrag")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "Qwen/Qwen3-VL-4B-Instruct"
LORA = "Chrisyichuan/qwen3vl-4b-wiki-screenshot-multik-3x-lora"


def page_metrics(ranks):
    def rk(k): return sum(1 for r in ranks if r is not None and r < k) / len(ranks)
    nd = sum((1.0 / math.log2(r + 2)) for r in ranks if r is not None and r < 5) / len(ranks)
    return {"recall@1": round(rk(1), 4), "recall@5": round(rk(5), 4), "ndcg@5": round(nd, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=768); ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    man = json.load(open(f"{SCRATCH}/meta/manifest.json"))["records"]
    if args.max: man = man[:args.max]
    id2pos = {r["id"]: i for i, r in enumerate(man)}
    page_doc = [r["doc"] for r in man]
    qd = [q for q in json.load(open(f"{SCRATCH}/meta/queries.json"))["queries"] if q["gold_id"] in id2pos]
    queries = [q["query"] for q in qd]; gold = [id2pos[q["gold_id"]] for q in qd]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(BASE)
    base = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, LORA).eval()

    INSTR = "Represent the user's input."   # PixelRAG DEFAULT_INSTRUCTION

    @torch.no_grad()
    def embed(content):
        # PixelRAG recipe: "Instruct: ...\n" prefix + add_generation_prompt=True, last-token pool
        msg = [{"role": "user", "content": [{"type": "text", "text": f"Instruct: {INSTR}\n"}] + content}]
        inp = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        out = model(**inp, output_hidden_states=True)
        return F.normalize(out.hidden_states[-1][0, -1].float(), dim=-1).cpu().numpy()  # last-token pool

    import time; t0 = time.time()
    q_emb = np.stack([embed([{"type": "text", "text": q}]) for q in queries])
    p_emb = np.zeros((len(man), q_emb.shape[1]), dtype=np.float32)
    for i, r in enumerate(man):
        im = Image.open(r["img"]).convert("RGB"); im.thumbnail((args.resolution, args.resolution))
        tp = f"/tmp/cq_{os.getpid()}.png"; im.save(tp)
        p_emb[i] = embed([{"type": "image", "image": tp}])
        if (i + 1) % 200 == 0: print(f"  embedded {i+1}/{len(man)} ({time.time()-t0:.0f}s)")

    order = np.argsort(-(q_emb @ p_emb.T), axis=1)
    ranks = [int(np.where(order[i] == g)[0][0]) if g in order[i] else None for i, g in enumerate(gold)]
    def doc_r(k): return round(sum(1 for i, g in enumerate(gold) if any(page_doc[p] == page_doc[g] for p in order[i][:k])) / len(gold), 4)
    res = {"__meta__": {"embedder": f"{BASE} + screenshot-LoRA (last-token pooled)", "corpus": len(man),
                        "queries": len(queries), "compare_to": "Qwen3-VL-Embedding-2B base pixel_VL",
                        "note": "does PixelRAG's fine-tuned 4B embedder beat our 2B base? aggregate only."},
           "pixelrag_4b_lora": {"page": page_metrics(ranks), "doc_recall@1": doc_r(1), "doc_recall@5": doc_r(5)}}
    out = f"{REPO}/data/output/skippy_colqwen.json"
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res["pixelrag_4b_lora"], indent=2)); print("wrote", out)


if __name__ == "__main__":
    main()
