#!/usr/bin/env python3
"""
pixelrag_bakeoff.py — lean, self-contained PixelRAG (visual-RAG) vs text-RAG bake-off
on the RTX 5090, in the keyhole measurement style.

The full PixelRAG paper repro is infeasible locally (~13TB page tiles + ~570GB FAISS
indexes + H100 serves + an OpenAI grader). This harness instead measures the SAME two
claims on a small ViDoRe retrieval slice, with the ONLY variable being whether a page
is represented as PIXELS (the page image) or TEXT (its OCR), through the SAME embedder
and the SAME reader:

  accuracy axis  — RETRIEVAL quality: does the right page come back? (recall@1/5, nDCG@5, MRR)
                   pixel-embed (Qwen3-VL-Embedding-2B on the image)  vs
                   text-embed  (same embedder on the page's OCR text)
  cost axis      — reader INPUT TOKENS for the retrieved page: fed as an image vs as OCR
                   text (this is where PixelRAG's "~10x fewer tokens" lives — optical
                   compression: a page-image is far fewer VLM tokens than its OCR text)
  latency axis   — 5090 wall-clock per stage: embed-corpus, query-embed, retrieve,
                   reader-prefill (image vs text path)

ViDoRe subsets are 1 query : 1 gold page; the corpus is all unique page images in the
slice. InfoVQA ships AWS-Textract OCR in the `ocr` column → a real text-parser baseline
with no OCR install.

Run:  python scripts/pixelrag_bakeoff.py --subset vidore/infovqa_test_subsampled --n 200
Env:  ~/.virtualenvs/pixelrag  (torch 2.11+cu128, transformers 5.12, sentence-transformers)
"""
import argparse, ast, json, math, os, time, tempfile
import numpy as np
import torch
from PIL import Image


def textract_to_text(ocr_field):
    """Pull plain text out of ViDoRe's Textract `ocr` field.

    The field is a STRING holding a python-repr list of JSON strings; each JSON
    object is a dict with separate PAGE / LINE / WORD arrays. Text lives in LINE
    blocks (fall back to WORD if a page has no lines)."""
    if not ocr_field:
        return ""
    try:
        lst = ast.literal_eval(ocr_field) if isinstance(ocr_field, str) else ocr_field
    except Exception:
        lst = [ocr_field]
    if isinstance(lst, (str, dict)):
        lst = [lst]
    lines = []
    for el in lst:
        try:
            obj = json.loads(el) if isinstance(el, str) else el
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        blocks = obj.get("LINE") or obj.get("WORD") or []
        for b in blocks:
            if isinstance(b, dict) and b.get("Text"):
                lines.append(b["Text"])
    return "\n".join(lines)


def ndcg_at_k(ranks, k=5):
    """ranks[i] = 0-based rank of the gold doc for query i (or None if not in top-k)."""
    s = 0.0
    for r in ranks:
        if r is not None and r < k:
            s += 1.0 / math.log2(r + 2)   # single relevant doc, rel=1
    return s / len(ranks)


def recall_at_k(ranks, k):
    return sum(1 for r in ranks if r is not None and r < k) / len(ranks)


def mrr(ranks):
    return sum((1.0 / (r + 1)) if r is not None else 0.0 for r in ranks) / len(ranks)


def faiss_rank(corpus_emb, query_emb, gold_idx):
    """L2-normalize, inner-product search; return 0-based rank of each query's gold doc."""
    import faiss
    cn = corpus_emb / (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-8)
    qn = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
    index = faiss.IndexFlatIP(cn.shape[1])
    index.add(cn.astype(np.float32))
    _, I = index.search(qn.astype(np.float32), cn.shape[0])  # full ranking
    ranks = []
    for i, gold in enumerate(gold_idx):
        pos = np.where(I[i] == gold)[0]
        ranks.append(int(pos[0]) if len(pos) else None)
    index_bytes = cn.nbytes
    return ranks, index_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="vidore/infovqa_test_subsampled")
    ap.add_argument("--n", type=int, default=200, help="slice size (queries = unique pages)")
    ap.add_argument("--reader_sample", type=int, default=40,
                    help="how many retrieved pages to push through the reader for the token/latency measurement")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(REPO, "data", "output",
                                   f"pixelrag_bakeoff_{args.subset.split('/')[-1]}.json")

    from datasets import load_dataset
    ds = load_dataset(args.subset, split="test").select(range(args.n))
    tmp = tempfile.mkdtemp(prefix="vidore_")
    img_paths, texts, queries = [], [], []
    for i, r in enumerate(ds):
        p = os.path.join(tmp, f"{i}.png")
        r["image"].convert("RGB").save(p)
        img_paths.append(p)
        texts.append(textract_to_text(r.get("ocr")))
        queries.append(r["query"])
    gold_idx = list(range(len(ds)))  # subsampled: query i ↔ page i
    have_text = sum(1 for t in texts if t.strip())
    print(f"slice: {len(ds)} pages/queries; pages with OCR text: {have_text}")

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device="cuda")

    res = {"__meta__": {
        "subset": args.subset, "n": len(ds), "pages_with_ocr": have_text,
        "embedder": "Qwen/Qwen3-VL-Embedding-2B", "reader": "Qwen/Qwen3-VL-4B-Instruct",
        "host": "RTX 5090 sm_120, torch 2.11+cu128",
        "note": "ViDoRe subsampled = 1 query:1 gold page; corpus = all pages in the slice. "
                "Only variable across arms is pixel-image vs OCR-text representation "
                "(same embedder, same reader)."}}

    # ---- query embeddings (shared) ----
    t = time.time(); q_emb = np.asarray(emb.encode(queries)); q_t = time.time() - t

    # ---- PIXEL arm: embed page images ----
    t = time.time(); p_emb = np.asarray(emb.encode([{"image": p} for p in img_paths]))
    pix_embed_t = time.time() - t
    pix_ranks, pix_bytes = faiss_rank(p_emb, q_emb, gold_idx)

    # ---- TEXT arm: embed OCR text ----
    t = time.time(); t_emb = np.asarray(emb.encode(texts)); txt_embed_t = time.time() - t
    txt_ranks, txt_bytes = faiss_rank(t_emb, q_emb, gold_idx)

    for arm, ranks, embt, nbytes in [("pixel", pix_ranks, pix_embed_t, pix_bytes),
                                     ("text", txt_ranks, txt_embed_t, txt_bytes)]:
        res[arm] = {
            "recall@1": round(recall_at_k(ranks, 1), 4),
            "recall@5": round(recall_at_k(ranks, 5), 4),
            "ndcg@5": round(ndcg_at_k(ranks, 5), 4),
            "mrr": round(mrr(ranks), 4),
            "corpus_embed_s": round(embt, 2),
            "corpus_embed_ms_per_doc": round(1000 * embt / len(ds), 1),
            "index_bytes": int(nbytes),
            "index_kb_per_doc": round(nbytes / len(ds) / 1024, 1),
        }
    res["__meta__"]["query_embed_s"] = round(q_t, 2)

    # ---- reader INPUT-TOKEN + prefill measurement (the 10x cost axis) ----
    del emb; torch.cuda.empty_cache()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    rid = "Qwen/Qwen3-VL-4B-Instruct"
    proc = AutoProcessor.from_pretrained(rid)
    model = AutoModelForImageTextToText.from_pretrained(rid, dtype="auto", device_map="cuda")

    def reader_measure(content, q):
        msgs = [{"role": "user", "content": content +
                 [{"type": "text", "text": q + "\nAnswer concisely."}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        n_in = int(inp["input_ids"].shape[1])
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=1, do_sample=False)  # prefill-dominated
        torch.cuda.synchronize()
        return n_in, time.time() - t

    sample = min(args.reader_sample, len(ds))
    pim_tok, ptxt_tok, pim_ms, ptxt_ms = [], [], [], []
    for i in range(sample):
        ni, mi = reader_measure([{"type": "image", "image": img_paths[i]}], queries[i])
        pim_tok.append(ni); pim_ms.append(1000 * mi)
        txt = texts[i][:8000] if texts[i].strip() else "(no OCR text)"
        nt, mt = reader_measure([{"type": "text", "text": "Document:\n" + txt}], queries[i])
        ptxt_tok.append(nt); ptxt_ms.append(1000 * mt)

    res["reader_tokens"] = {
        "sample": sample,
        "pixel_input_tokens_mean": round(float(np.mean(pim_tok)), 1),
        "text_input_tokens_mean": round(float(np.mean(ptxt_tok)), 1),
        "token_ratio_text_over_pixel": round(float(np.mean(ptxt_tok)) / float(np.mean(pim_tok)), 2),
        "pixel_prefill_ms_mean": round(float(np.mean(pim_ms)), 1),
        "text_prefill_ms_mean": round(float(np.mean(ptxt_ms)), 1),
        "reader_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }

    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
