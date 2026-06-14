#!/usr/bin/env python3
"""
skippy_emails.py — boundary-condition arm: PixelRAG on Skippy's EMAILS (low-layout text).

Datasheets are the pixel sweet spot (dense layout). Emails are the opposite — mostly
plain text, little visual structure — so this tests whether the pixel advantage is
layout-specific. Pipeline (all LOCAL, scratch only, aggregate-only output):
  sample .eml -> parse subject+body -> render the text to a page image (PixelRAG-style)
  -> local Qwen query-gen (gold map) -> embed pixel(image) vs text(raw) -> retrieval.

PRIVACY: reads scratch ~/skippy_pixelrag/_email_raw (extracted locally), writes images/
text/queries to scratch only; the repo gets code + aggregate metrics. No email content
leaves the box (query-gen is the local Qwen).

Run:  python scripts/skippy_emails.py --n 150 --queries 60
"""
import argparse, email, glob, json, math, os, random, re
from email import policy
import numpy as np
from PIL import Image, ImageDraw

SCRATCH = os.path.expanduser("~/skippy_pixelrag")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(SCRATCH, "_email_raw")


def body_text(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try: return part.get_content()
                except Exception: pass
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                try: return re.sub(r"<[^>]+>", " ", part.get_content())
                except Exception: pass
        return ""
    try:
        t = msg.get_content()
        return re.sub(r"<[^>]+>", " ", t) if msg.get_content_type() == "text/html" else t
    except Exception:
        return ""


def render_text(text, path, W=768, H=1000):
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    y = 12
    for raw in text.split("\n"):
        for i in range(0, max(1, len(raw)), 95):
            if y > H - 16: break
            d.text((12, y), raw[i:i + 95], fill="black"); y += 14
        if y > H - 16: break
    img.save(path)


def metrics(ranks):
    def rk(k): return sum(1 for r in ranks if r is not None and r < k) / len(ranks)
    nd = sum((1.0 / math.log2(r + 2)) for r in ranks if r is not None and r < 5) / len(ranks)
    return {"recall@1": round(rk(1), 4), "recall@5": round(rk(5), 4), "ndcg@5": round(nd, 4),
            "mrr": round(sum((1.0 / (r + 1)) if r is not None else 0 for r in ranks) / len(ranks), 4)}


def ranks_of(corpus, query, gold):
    cn = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    order = np.argsort(-(qn @ cn.T), axis=1)
    return [int(np.where(order[i] == g)[0][0]) if g in order[i] else None for i, g in enumerate(gold)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--min_chars", type=int, default=200)
    args = ap.parse_args()

    files = glob.glob(os.path.join(RAW, "**", "*.eml"), recursive=True)
    files = files[:: max(1, len(files) // (args.n * 3))]  # spread the sample
    pdir = os.path.join(SCRATCH, "email_pages"); tdir = os.path.join(SCRATCH, "email_text")
    os.makedirs(pdir, exist_ok=True); os.makedirs(tdir, exist_ok=True)
    recs = []
    for f in files:
        if len(recs) >= args.n: break
        try:
            msg = email.message_from_binary_file(open(f, "rb"), policy=policy.default)
        except Exception:
            continue
        subj = str(msg.get("subject", "") or "")
        body = body_text(msg) or ""
        text = (subj + "\n" + body).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) < args.min_chars:
            continue
        i = len(recs)
        ip = os.path.join(pdir, f"{i}.png"); tp = os.path.join(tdir, f"{i}.txt")
        render_text(text, ip); open(tp, "w").write(text[:12000])
        recs.append({"id": f"emails__{i}", "img": ip, "txt": tp})
    print(f"emails corpus: {len(recs)} (rendered text-images + raw text)")

    # local query-gen on a subset (reads the rendered email image -> question)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    rid = "Qwen/Qwen3-VL-4B-Instruct"
    proc = AutoProcessor.from_pretrained(rid)
    rdr = AutoModelForImageTextToText.from_pretrained(rid, dtype="auto", device_map="cuda")
    PROMPT = ("Write ONE specific question answerable only from this email (reference a concrete "
              "detail: a name, date, number, or fact). Output only the question.")
    stride = max(1, len(recs) // args.queries)
    qsel = recs[::stride][:args.queries]
    qmap = []
    for j, r in enumerate(qsel):
        m = [{"role": "user", "content": [{"type": "image", "image": r["img"]},
             {"type": "text", "text": PROMPT}]}]
        inp = proc.apply_chat_template(m, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = rdr.generate(**inp, max_new_tokens=48, do_sample=False)
        q = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().replace("\n", " ")
        qmap.append((q, recs.index(r)))
    del rdr; torch.cuda.empty_cache()

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device="cuda")
    queries = [q for q, _ in qmap]; gold = [g for _, g in qmap]
    q_emb = np.asarray(emb.encode(queries, batch_size=16))
    p_emb = np.asarray(emb.encode([{"image": r["img"]} for r in recs], batch_size=8))
    t_emb = np.asarray(emb.encode([open(r["txt"]).read() for r in recs], batch_size=16))

    res = {"__meta__": {"corpus": len(recs), "queries": len(queries), "content": "emails (low-layout text)",
                        "note": "boundary test: does pixel beat text when there's little layout? "
                                "aggregate only; no email content in repo."},
           "pixel": metrics(ranks_of(p_emb, q_emb, gold)),
           "text": metrics(ranks_of(t_emb, q_emb, gold))}
    out = f"{REPO}/data/output/skippy_emails.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"  pixel: {res['pixel']}\n  text:  {res['text']}\nwrote {out}")


if __name__ == "__main__":
    main()
