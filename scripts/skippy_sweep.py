#!/usr/bin/env python3
"""
skippy_sweep.py — PixelRAG bake-off + primary sweep over Skippy's own corpus.

Retrieval pool = all ingested Skippy pages; queries = the locally-generated gold map.
Only variable across arms: page as PIXELS (image, swept over resolution) vs TEXT
(native PDF text). Same Qwen3-VL-Embedding-2B embedder, same Qwen3-VL-4B reader.

Primary grid:
  resolution  — pixel page downscaled to long-side ∈ {512,768,1024,1536} → the token
                knee (image tokens fall with resolution; does retrieval/read hold?)
  doc-class   — datasheets / documents / writing (where pixel wins big vs where text is fine)
  top-k       — recall@{1,3,5,10} read off the full ranking
Output: data/output/skippy_pixelrag_sweep.json — AGGREGATE METRICS ONLY (scores, token
counts, class labels, resolutions). No document text, queries, or personal content.

Run:  python scripts/skippy_sweep.py --resolutions 512 768 1024 1536 --reader_sample 24
Privacy: reads scratch ~/skippy_pixelrag only; the repo never sees the source content.
"""
import argparse, json, math, os, time, tempfile
import numpy as np
import torch
from PIL import Image

SCRATCH = os.path.expanduser("~/skippy_pixelrag")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def metrics(ranks):
    def rk(k): return sum(1 for r in ranks if r is not None and r < k) / len(ranks)
    ndcg = sum((1.0 / math.log2(r + 2)) for r in ranks if r is not None and r < 5) / len(ranks)
    mrr = sum((1.0 / (r + 1)) if r is not None else 0.0 for r in ranks) / len(ranks)
    return {"recall@1": round(rk(1), 4), "recall@3": round(rk(3), 4),
            "recall@5": round(rk(5), 4), "recall@10": round(rk(10), 4),
            "ndcg@5": round(ndcg, 4), "mrr": round(mrr, 4)}


def rank_for(corpus_emb, query_emb, gold_pos):
    import faiss
    cn = corpus_emb / (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-8)
    qn = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
    idx = faiss.IndexFlatIP(cn.shape[1]); idx.add(cn.astype(np.float32))
    _, I = idx.search(qn.astype(np.float32), cn.shape[0])
    return [int(np.where(I[i] == g)[0][0]) if g in I[i] else None for i, g in enumerate(gold_pos)]


def per_class(ranks, q_class, classes):
    out = {"all": metrics(ranks)}
    for c in classes:
        sub = [r for r, qc in zip(ranks, q_class) if qc == c]
        if sub:
            out[c] = metrics(sub)
    return out


def resize_to(path, long_side, cache):
    """Downscale image so its long side == long_side; cache the temp path."""
    key = (path, long_side)
    if key in cache:
        return cache[key]
    im = Image.open(path).convert("RGB")
    im.thumbnail((long_side, long_side))
    p = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    im.save(p)
    cache[key] = p
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=int, nargs="+", default=[512, 768, 1024, 1536])
    ap.add_argument("--reader_sample", type=int, default=24)
    ap.add_argument("--max_corpus", type=int, default=0, help="0 = all pages")
    args = ap.parse_args()

    man = json.load(open(os.path.join(SCRATCH, "meta", "manifest.json")))["records"]
    if args.max_corpus:
        man = man[:args.max_corpus]
    id2pos = {r["id"]: i for i, r in enumerate(man)}
    classes = sorted({r["class"] for r in man})
    qd = json.load(open(os.path.join(SCRATCH, "meta", "queries.json")))["queries"]
    qd = [q for q in qd if q["gold_id"] in id2pos]
    queries = [q["query"] for q in qd]
    gold_pos = [id2pos[q["gold_id"]] for q in qd]
    q_class = [q["class"] for q in qd]
    print(f"corpus {len(man)} pages, {len(queries)} queries, classes {classes}")

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device="cuda")
    q_emb = np.asarray(emb.encode(queries, batch_size=16))

    res = {"__meta__": {"corpus_pages": len(man), "n_queries": len(queries),
                        "classes": classes, "resolutions": args.resolutions,
                        "embedder": "Qwen/Qwen3-VL-Embedding-2B", "reader": "Qwen/Qwen3-VL-4B-Instruct",
                        "host": "RTX 5090 sm_120", "source": "Skippy corpus (local; no content in repo)"},
           "text": {}, "pixel_by_resolution": {}}

    # ---- TEXT arm (resolution-independent) ----
    texts = [open(man[i]["txt"]).read()[:12000] for i in range(len(man))]
    t = time.time(); t_emb = np.asarray(emb.encode(texts, batch_size=16))
    res["text"] = per_class(rank_for(t_emb, q_emb, gold_pos), q_class, classes)
    res["text"]["_embed_s"] = round(time.time() - t, 1)

    # ---- PIXEL arm, swept over resolution ----
    cache = {}
    for R in args.resolutions:
        paths = [resize_to(man[i]["img"], R, cache) for i in range(len(man))]
        t = time.time(); p_emb = np.asarray(emb.encode([{"image": p} for p in paths], batch_size=8))
        m = per_class(rank_for(p_emb, q_emb, gold_pos), q_class, classes)
        m["_embed_s"] = round(time.time() - t, 1)
        res["pixel_by_resolution"][str(R)] = m
        print(f"  res {R}: pixel recall@1(all)={m['all']['recall@1']}  ({m['_embed_s']}s)")
    del emb; torch.cuda.empty_cache()

    # ---- reader INPUT-TOKEN / prefill: image (per res) vs text ----
    from transformers import AutoModelForImageTextToText, AutoProcessor
    rid = "Qwen/Qwen3-VL-4B-Instruct"
    proc = AutoProcessor.from_pretrained(rid)
    model = AutoModelForImageTextToText.from_pretrained(rid, dtype="auto", device_map="cuda")

    def measure(content):
        inp = proc.apply_chat_template([{"role": "user", "content": content}],
                                       tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        n = int(inp["input_ids"].shape[1])
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        return n, 1000 * (time.time() - t)

    samp = [man[gold_pos[i]] for i in range(min(args.reader_sample, len(gold_pos)))]
    q_txt = [{"type": "text", "text": "Answer from the document."}]
    reader = {"text": {"tokens": [], "prefill_ms": []}}
    for R in args.resolutions:
        reader[str(R)] = {"tokens": [], "prefill_ms": []}
    for rec in samp:
        nt, mt = measure([{"type": "text", "text": "Document:\n" + open(rec["txt"]).read()[:8000]}] + q_txt)
        reader["text"]["tokens"].append(nt); reader["text"]["prefill_ms"].append(mt)
        for R in args.resolutions:
            ni, mi = measure([{"type": "image", "image": resize_to(rec["img"], R, cache)}] + q_txt)
            reader[str(R)]["tokens"].append(ni); reader[str(R)]["prefill_ms"].append(mi)
    res["reader"] = {k: {"input_tokens_mean": round(float(np.mean(v["tokens"])), 1),
                         "prefill_ms_mean": round(float(np.mean(v["prefill_ms"])), 1)}
                     for k, v in reader.items()}
    res["reader"]["vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    out = os.path.join(REPO, "data", "output", "skippy_pixelrag_sweep.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\n" + json.dumps(res, indent=2))
    print("wrote", out, "(aggregate metrics only)")


if __name__ == "__main__":
    main()
