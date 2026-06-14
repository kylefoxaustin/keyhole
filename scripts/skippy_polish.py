#!/usr/bin/env python3
"""
skippy_polish.py — two rigor follow-ups on the main Skippy corpus:
  (1) DOC-LEVEL metric — success if the gold page's DOCUMENT is retrieved in top-k
      (resolves the transcript caveat: near-duplicate pages make exact-page R@1 unfair).
  (2) BGE CHUNK-CONTROL — give the dedicated text retriever a fair shot by chunking each
      page to <512 tokens, embedding all chunks, and scoring a page by its best chunk
      (max-pool). Tests whether BGE's earlier loss was just truncation.

Compares pixel-VL / text-VL / text-BGE(page) / text-BGE(chunked) at page- and doc-level.
Aggregate metrics only. Run: python scripts/skippy_polish.py --resolution 768
"""
import argparse, json, math, os, tempfile
import numpy as np

SCRATCH = os.path.expanduser("~/skippy_pixelrag")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def page_metrics(ranks):
    def rk(k): return sum(1 for r in ranks if r is not None and r < k) / len(ranks)
    nd = sum((1.0 / math.log2(r + 2)) for r in ranks if r is not None and r < 5) / len(ranks)
    return {"recall@1": round(rk(1), 4), "recall@5": round(rk(5), 4), "ndcg@5": round(nd, 4)}


def order_of(scores):
    return np.argsort(-scores, axis=1)


def page_ranks(order, gold):
    return [int(np.where(order[i] == g)[0][0]) if g in order[i] else None for i, g in enumerate(gold)]


def doc_recall(order, gold, page_doc, ks=(1, 5)):
    """success@k if any of top-k retrieved pages shares the gold page's doc."""
    out = {}
    for k in ks:
        hit = 0
        for i, g in enumerate(gold):
            gd = page_doc[g]
            if any(page_doc[p] == gd for p in order[i][:k]):
                hit += 1
        out[f"doc_recall@{k}"] = round(hit / len(gold), 4)
    return out


def sims(corpus, query, norm=True):
    if norm:
        corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    return query @ corpus.T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resolution", type=int, default=768)
    args = ap.parse_args()
    man = json.load(open(f"{SCRATCH}/meta/manifest.json"))["records"]
    id2pos = {r["id"]: i for i, r in enumerate(man)}
    page_doc = [r["doc"] for r in man]
    qd = [q for q in json.load(open(f"{SCRATCH}/meta/queries.json"))["queries"] if q["gold_id"] in id2pos]
    queries = [q["query"] for q in qd]; gold = [id2pos[q["gold_id"]] for q in qd]
    texts = [open(r["txt"]).read()[:12000] for r in man]

    from sentence_transformers import SentenceTransformer
    from PIL import Image
    import torch
    res = {"__meta__": {"corpus": len(man), "queries": len(queries),
                        "note": "page- and doc-level; BGE page vs chunked. aggregate only."}}

    def record(name, order):
        res[name] = {"page": page_metrics(page_ranks(order, gold)), **doc_recall(order, gold, page_doc)}

    # VL: pixel + text
    vl = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device="cuda")
    q_vl = np.asarray(vl.encode(queries, batch_size=16))
    d = tempfile.mkdtemp(); paths = []
    for i, r in enumerate(man):
        im = Image.open(r["img"]).convert("RGB"); im.thumbnail((args.resolution, args.resolution))
        p = f"{d}/{i}.png"; im.save(p); paths.append(p)
    record("pixel_VL", order_of(sims(np.asarray(vl.encode([{"image": p} for p in paths], batch_size=8)), q_vl)))
    record("text_VL", order_of(sims(np.asarray(vl.encode(texts, batch_size=16)), q_vl)))
    del vl; torch.cuda.empty_cache()

    # BGE page-level
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    qp = ["Represent this sentence for searching relevant passages: " + q for q in queries]
    q_bge = np.asarray(bge.encode(qp, batch_size=32, normalize_embeddings=True))
    record("text_BGE_page", order_of(sims(np.asarray(bge.encode(texts, batch_size=32, normalize_embeddings=True)), q_bge, norm=False)))

    # BGE chunked: split each page to <512-tok chunks, score page = best chunk
    chunks, owner = [], []
    for i, t in enumerate(texts):
        for c in range(0, max(1, len(t)), 1600):  # ~400 tokens
            chunks.append(t[c:c + 1600]); owner.append(i)
    c_emb = np.asarray(bge.encode(chunks, batch_size=64, normalize_embeddings=True))
    c_sim = q_bge @ c_emb.T  # (Q, n_chunks)
    page_score = np.full((len(queries), len(man)), -1e9)
    owner = np.asarray(owner)
    for pi in range(len(man)):
        cols = np.where(owner == pi)[0]
        page_score[:, pi] = c_sim[:, cols].max(axis=1)
    record("text_BGE_chunked", order_of(page_score))

    out = f"{REPO}/data/output/skippy_polish.json"
    json.dump(res, open(out, "w"), indent=2)
    for k in ["pixel_VL", "text_VL", "text_BGE_page", "text_BGE_chunked"]:
        print(f"  {k:18} page-R@5={res[k]['page']['recall@5']:.3f}  doc-R@1={res[k]['doc_recall@1']:.3f}  doc-R@5={res[k]['doc_recall@5']:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
