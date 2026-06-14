#!/usr/bin/env python3
"""Faithful reproduction of PixelRAG's 4B + screenshot-LoRA single-vector embedder,
evaluated against the same Skippy corpus + queries used for the Qwen3-VL-Embedding-2B
baseline.

PRIVACY: Reads the personal corpus/queries from local scratch paths only. Writes ONLY
aggregate metric numbers to data/output/skippy_colqwen_native.json. No document text,
query text, file names, or titles are emitted anywhere. Nothing is sent off-machine.

Uses PixelRAG's OWN code:
  - embed.py:_init_direct_gpu  -> loads Qwen3VLForConditionalGeneration + LoRA adapter
                                  (BiQwen3 key remap + merge_and_unload + Conv3d patch)
  - embed.py:_embed_direct_gpu -> image (doc) embedding: model.model(**inputs),
                                  last-non-pad-token pooling, L2-normalize
  - serve/api.py:_encode_queries -> TEXT-only query path (reproduced here): different
                                  system instruction, processor(text=..., no images),
                                  same model.model + last-token pooling + L2-normalize.

Run with: ~/.virtualenvs/pixelrag/bin/python scripts/skippy_pixelrag_native.py [--limit N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

# --- PixelRAG's own embedding code ------------------------------------------
sys.path.insert(0, "/tmp/PixelRAG/embed/src")
from pixelrag_embed import embed as pre  # noqa: E402

MANIFEST = "/home/kyle/skippy_pixelrag/meta/manifest.json"
QUERIES = "/home/kyle/skippy_pixelrag/meta/queries.json"
OUT = "/home/kyle/Documents/GitHub/keyhole/data/output/skippy_colqwen_native.json"

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
ADAPTER = "Chrisyichuan/qwen3vl-4b-wiki-screenshot-multik-3x-lora"

# Instructions, matched to PixelRAG's two code paths.
DOC_INSTRUCTION = "Represent the user's input."  # embed.py DEFAULT_INSTRUCTION (image side)
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."  # serve/api.py


def load_image_768(path):
    """Match the 2B arm: resize so the long side is 768px (PIL thumbnail)."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((768, 768), Image.LANCZOS)
    return img


def embed_docs(engine, records, batch_size=8, instruction=DOC_INSTRUCTION):
    """Embed page images using PixelRAG's _embed_direct_gpu (image side)."""
    model, processor = engine
    # _embed_direct_gpu reads the module-level _INSTRUCTION; set it explicitly.
    pre.set_instruction(instruction)
    out = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        imgs = [load_image_768(r["img"]) for r in batch]
        embs = pre._embed_direct_gpu(engine, prompt=None, images=imgs)  # prompt unused
        out.extend(np.asarray(e, dtype=np.float32) for e in embs)
    return np.stack(out)


def embed_queries(engine, queries, batch_size=16, instruction=QUERY_INSTRUCTION):
    """Reproduce serve/api.py:_encode_queries faithfully (TEXT-only query path).

    Same model.model(**inputs) forward + last-non-pad-token pooling + L2-norm as the
    image side, but: (a) different system instruction, (b) text-only processor call
    (no images), (c) the user turn carries a text chunk instead of an image.
    """
    import torch

    model, processor = engine
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        messages_list = [
            [
                {"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": [{"type": "text", "text": q["query"]}]},
            ]
            for q in batch
        ]
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.model(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        last_token_indices = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[
            torch.arange(last_hidden.size(0), device=last_hidden.device),
            last_token_indices,
        ]
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        out.extend(pooled.cpu().float().numpy())
    return np.stack(out)


def evaluate(doc_vecs, query_vecs, records, queries):
    """Full-ranking cosine/IP retrieval (vectors are L2-normalized).

    page-level: rank of the gold page id.
    doc-level:  success@k if any top-k page shares the gold page's `doc`.
    """
    id_to_row = {r["id"]: i for i, r in enumerate(records)}
    row_doc = [r["doc"] for r in records]
    # IP == cosine since both sides L2-normalized.
    sims = query_vecs @ doc_vecs.T  # (Q, N)
    order = np.argsort(-sims, axis=1)  # full ranking, best first

    page_r1 = page_r5 = 0
    doc_r1 = doc_r5 = 0
    ndcg5_sum = 0.0
    n = 0
    for qi, q in enumerate(queries):
        gold_row = id_to_row.get(q["gold_id"])
        if gold_row is None:
            continue
        n += 1
        gold_doc = row_doc[gold_row]
        ranking = order[qi]
        # page rank position
        pos = int(np.where(ranking == gold_row)[0][0])
        if pos == 0:
            page_r1 += 1
        if pos < 5:
            page_r5 += 1
            ndcg5_sum += 1.0 / np.log2(pos + 2)  # single relevant page, IDCG=1
        # doc-level
        top5_docs = [row_doc[r] for r in ranking[:5]]
        if row_doc[ranking[0]] == gold_doc:
            doc_r1 += 1
        if gold_doc in top5_docs:
            doc_r5 += 1
    return {
        "n_queries": n,
        "page_recall@1": page_r1 / n,
        "page_recall@5": page_r5 / n,
        "nDCG@5": ndcg5_sum / n,
        "doc_recall@1": doc_r1 / n,
        "doc_recall@5": doc_r5 / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="sanity slice: first N pages + their queries")
    ap.add_argument("--doc-batch", type=int, default=8)
    ap.add_argument("--write", action="store_true", help="write aggregate JSON output")
    args = ap.parse_args()

    records = json.load(open(MANIFEST))["records"]
    queries = json.load(open(QUERIES))["queries"]

    if args.limit:
        records = records[: args.limit]
        keep = {r["id"] for r in records}
        queries = [q for q in queries if q["gold_id"] in keep]
        print(f"[sanity] {len(records)} pages, {len(queries)} queries with gold in slice", flush=True)

    # _init_direct_gpu joins adapter_path with a filename directly, so it needs a LOCAL
    # directory. Resolve the Hub repo id to its local snapshot dir first.
    from huggingface_hub import snapshot_download
    adapter_local = snapshot_download(ADAPTER)

    print(f"Loading model {MODEL} + LoRA {ADAPTER} via PixelRAG _init_direct_gpu ...", flush=True)
    t0 = time.time()
    engine = pre._init_direct_gpu(MODEL, gpu_id=0, adapter_path=adapter_local)
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    doc_vecs = embed_docs(engine, records, batch_size=args.doc_batch)
    print(f"embedded {len(doc_vecs)} docs in {time.time()-t0:.1f}s, dim={doc_vecs.shape[1]}", flush=True)

    # Primary (serve-faithful): docs use embed.py DOC_INSTRUCTION, queries use
    # serve/api.py QUERY_INSTRUCTION (asymmetric — exactly PixelRAG's index+serve path).
    t0 = time.time()
    qv_serve = embed_queries(engine, queries, instruction=QUERY_INSTRUCTION)
    print(f"embedded {len(qv_serve)} queries in {time.time()-t0:.1f}s", flush=True)
    metrics = evaluate(doc_vecs, qv_serve, records, queries)
    print("[serve-faithful asymmetric DOC/QRY]", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)

    # Sensitivity: symmetric instruction (same QUERY_INSTRUCTION on both sides),
    # which empirically retrieves best for this generative-VLM+reader-LoRA used as embedder.
    qv_sym = embed_queries(engine, queries, instruction=QUERY_INSTRUCTION)
    doc_vecs_sym = embed_docs(engine, records, batch_size=args.doc_batch, instruction=QUERY_INSTRUCTION)
    metrics_sym = evaluate(doc_vecs_sym, qv_sym, records, queries)
    print("[symmetric QRY/QRY (best-case sensitivity)]", flush=True)
    print(json.dumps(metrics_sym, indent=2), flush=True)

    if args.write:
        result = {
            "variant": "PixelRAG 4B + screenshot-LoRA (Qwen/Qwen3-VL-4B-Instruct + Chrisyichuan/qwen3vl-4b-wiki-screenshot-multik-3x-lora), single-vector retriever",
            "method": "PixelRAG embed.py _init_direct_gpu/_embed_direct_gpu (image=doc) + serve/api.py _encode_queries (text=query, reproduced); model.model(**inputs), last-non-pad-token pooling, L2-normalize; doc images thumbnail to 768px long side; cosine/IP full ranking",
            "n_pages": len(records),
            "embedding_dim": int(doc_vecs.shape[1]),
            "doc_instruction": DOC_INSTRUCTION,
            "query_instruction": QUERY_INSTRUCTION,
            "metrics_serve_faithful_asymmetric": metrics,
            "metrics_symmetric_query_instruction_best": metrics_sym,
            "baseline_2B_Qwen3VLEmbedding": {
                "page_recall@1": 0.300, "page_recall@5": 0.692, "nDCG@5": 0.511,
                "doc_recall@1": 0.725, "doc_recall@5": 0.825,
            },
            "notes": "This LoRA is PixelRAG's READER adapter (task_type=CAUSAL_LM, trained <image>xk + query -> answer), not an embedding/retriever adapter. PixelRAG's actual retriever is the Qwen3-VL-Embedding-2B model (= the baseline). LoRA helps over base 4B-Instruct but the generative VLM+reader-LoRA used as a single-vector embedder does not match the purpose-built 2B embedding model.",
        }
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(result, open(OUT, "w"), indent=2)
        print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
