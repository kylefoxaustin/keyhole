#!/usr/bin/env python3
"""
skippy_text_control.py — the confound control. The text arm so far used the SAME VL
embedder (Qwen3-VL-Embedding-2B) on text, which may be its weaker pathway. This adds a
DEDICATED text retriever (BAAI/bge-base-en-v1.5) for the text arm, on the identical
query set, to separate two explanations of the pixel win:
  - if text-BGE >> text-VL and ~matches pixel-VL  -> the win was VL-embedder modality bias
  - if pixel-VL still wins over text-BGE          -> visual-RAG is genuinely better

Main Skippy corpus (persisted manifest + queries.json). Aggregate metrics only.
Run:  python scripts/skippy_text_control.py --resolution 768
"""
import argparse, json, math, os, tempfile
import numpy as np

SCRATCH = os.path.expanduser("~/skippy_pixelrag")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def metrics(ranks):
    def rk(k): return sum(1 for r in ranks if r is not None and r < k) / len(ranks)
    nd = sum((1.0 / math.log2(r + 2)) for r in ranks if r is not None and r < 5) / len(ranks)
    return {"recall@1": round(rk(1), 4), "recall@5": round(rk(5), 4), "ndcg@5": round(nd, 4)}


def ranks_of(corpus, query, gold):
    cn = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    order = np.argsort(-(qn @ cn.T), axis=1)
    return [int(np.where(order[i] == g)[0][0]) if g in order[i] else None for i, g in enumerate(gold)]


def per_class(ranks, qcls, classes):
    o = {"all": metrics(ranks)}
    for c in classes:
        sub = [r for r, qc in zip(ranks, qcls) if qc == c]
        if sub: o[c] = metrics(sub)
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resolution", type=int, default=768)
    args = ap.parse_args()
    man = json.load(open(f"{SCRATCH}/meta/manifest.json"))["records"]
    id2pos = {r["id"]: i for i, r in enumerate(man)}
    classes = sorted({r["class"] for r in man})
    qd = [q for q in json.load(open(f"{SCRATCH}/meta/queries.json"))["queries"] if q["gold_id"] in id2pos]
    queries = [q["query"] for q in qd]; gold = [id2pos[q["gold_id"]] for q in qd]; qcls = [q["class"] for q in qd]
    texts = [open(r["txt"]).read()[:12000] for r in man]

    from sentence_transformers import SentenceTransformer
    from PIL import Image
    res = {"__meta__": {"corpus": len(man), "queries": len(queries), "classes": classes,
                        "resolution_px": args.resolution,
                        "note": "confound control: pixel-VL vs text-VL vs text-BGE on identical queries."}}

    # VL embedder: pixel + text-VL
    vl = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device="cuda")
    q_vl = np.asarray(vl.encode(queries, batch_size=16))
    d = tempfile.mkdtemp(); paths = []
    for i, r in enumerate(man):
        im = Image.open(r["img"]).convert("RGB"); im.thumbnail((args.resolution, args.resolution))
        p = os.path.join(d, f"{i}.png"); im.save(p); paths.append(p)
    p_vl = np.asarray(vl.encode([{"image": p} for p in paths], batch_size=8))
    t_vl = np.asarray(vl.encode(texts, batch_size=16))
    res["pixel_VL"] = per_class(ranks_of(p_vl, q_vl, gold), qcls, classes)
    res["text_VL"] = per_class(ranks_of(t_vl, q_vl, gold), qcls, classes)
    del vl
    import torch; torch.cuda.empty_cache()

    # dedicated text retriever
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    qpref = ["Represent this sentence for searching relevant passages: " + q for q in queries]
    q_bge = np.asarray(bge.encode(qpref, batch_size=32, normalize_embeddings=True))
    t_bge = np.asarray(bge.encode(texts, batch_size=32, normalize_embeddings=True))
    res["text_BGE"] = per_class(ranks_of(t_bge, q_bge, gold), qcls, classes)

    out = f"{REPO}/data/output/skippy_text_control.json"
    json.dump(res, open(out, "w"), indent=2)
    for arm in ["pixel_VL", "text_VL", "text_BGE"]:
        m = res[arm]["all"]; print(f"  {arm:9} R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} nDCG@5={m['ndcg@5']:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
