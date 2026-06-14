#!/usr/bin/env python3
"""
skippy_fusion.py — pixel vs text vs HYBRID retrieval on the Skippy corpus.

At the knee resolution, embed corpus pages both ways (image + native text) with the
same Qwen3-VL-Embedding-2B, then compare three retrievers:
  pixel   — image similarity only
  text    — native-text similarity only
  hybrid_rrf      — reciprocal-rank fusion of the two rankings (parameter-free)
  hybrid_w0.5     — weighted sum of min-max-normalised similarities (alpha=0.5)
Answers: does combining the modalities beat the better single one? Output aggregate
metrics only (data/output/skippy_fusion.json, gitignored). No personal content.

Run:  python scripts/skippy_fusion.py --resolution 768
"""
import argparse, json, math, os, tempfile
import numpy as np

SCRATCH = os.path.expanduser("~/skippy_pixelrag")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def metrics(ranks):
    def rk(k): return sum(1 for r in ranks if r is not None and r < k) / len(ranks)
    ndcg = sum((1.0 / math.log2(r + 2)) for r in ranks if r is not None and r < 5) / len(ranks)
    mrr = sum((1.0 / (r + 1)) if r is not None else 0.0 for r in ranks) / len(ranks)
    return {"recall@1": round(rk(1), 4), "recall@5": round(rk(5), 4),
            "ndcg@5": round(ndcg, 4), "mrr": round(mrr, 4)}


def sims(corpus, query):
    cn = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    return qn @ cn.T  # (Q, C) similarity


def ranks_from_scores(scores, gold):
    order = np.argsort(-scores, axis=1)  # high score first
    out = []
    for i, g in enumerate(gold):
        pos = np.where(order[i] == g)[0]
        out.append(int(pos[0]) if len(pos) else None)
    return out


def minmax(s):
    lo = s.min(axis=1, keepdims=True); hi = s.max(axis=1, keepdims=True)
    return (s - lo) / (hi - lo + 1e-8)


def rrf(s_a, s_b, k0=60):
    ra = (-s_a).argsort(axis=1).argsort(axis=1)  # rank of each item (0=best)
    rb = (-s_b).argsort(axis=1).argsort(axis=1)
    return 1.0 / (k0 + ra) + 1.0 / (k0 + rb)


def resize_paths(man, R):
    from PIL import Image
    out = []
    d = tempfile.mkdtemp()
    for i, r in enumerate(man):
        im = Image.open(r["img"]).convert("RGB"); im.thumbnail((R, R))
        p = os.path.join(d, f"{i}.png"); im.save(p); out.append(p)
    return out


def per_class(ranks, qcls, classes):
    o = {"all": metrics(ranks)}
    for c in classes:
        sub = [r for r, qc in zip(ranks, qcls) if qc == c]
        if sub: o[c] = metrics(sub)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=768)
    args = ap.parse_args()

    man = json.load(open(f"{SCRATCH}/meta/manifest.json"))["records"]
    id2pos = {r["id"]: i for i, r in enumerate(man)}
    classes = sorted({r["class"] for r in man})
    qd = [q for q in json.load(open(f"{SCRATCH}/meta/queries.json"))["queries"] if q["gold_id"] in id2pos]
    queries = [q["query"] for q in qd]; gold = [id2pos[q["gold_id"]] for q in qd]
    qcls = [q["class"] for q in qd]

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device="cuda")
    q_emb = np.asarray(emb.encode(queries, batch_size=16))
    paths = resize_paths(man, args.resolution)
    p_emb = np.asarray(emb.encode([{"image": p} for p in paths], batch_size=8))
    t_emb = np.asarray(emb.encode([open(r["txt"]).read()[:12000] for r in man], batch_size=16))

    s_pix = sims(p_emb, q_emb); s_txt = sims(t_emb, q_emb)
    res = {"__meta__": {"resolution_px": args.resolution, "corpus": len(man), "queries": len(queries),
                        "classes": classes, "note": "pixel/text/hybrid retrieval; aggregate only."}}
    res["pixel"] = per_class(ranks_from_scores(s_pix, gold), qcls, classes)
    res["text"] = per_class(ranks_from_scores(s_txt, gold), qcls, classes)
    res["hybrid_rrf"] = per_class(ranks_from_scores(rrf(s_pix, s_txt), gold), qcls, classes)
    res["hybrid_w0.5"] = per_class(ranks_from_scores(0.5 * minmax(s_pix) + 0.5 * minmax(s_txt), gold), qcls, classes)

    out = f"{REPO}/data/output/skippy_fusion.json"
    json.dump(res, open(out, "w"), indent=2)
    for arm in ["pixel", "text", "hybrid_rrf", "hybrid_w0.5"]:
        m = res[arm]["all"]; print(f"  {arm:12} R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} nDCG@5={m['ndcg@5']:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
