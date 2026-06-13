#!/usr/bin/env python3
"""
skippy_ingest.py — render Skippy's PDF knowledge to page images + native text for the
PixelRAG bake-off. PRIVACY: reads the personal-ai-framework knowledge dir and writes
ONLY to a scratch dir OUTSIDE any git repo (~/skippy_pixelrag). Nothing here is ever
committed; the keyhole repo gets this CODE plus aggregate metrics only.

Per PDF page it writes:  pages/<class>/<doc>_p<NNN>.png   (rendered image, ~144 dpi)
                          text/<class>/<doc>_p<NNN>.txt    (native PDF text layer)
and a manifest (meta/manifest.json) of page records {id, class, doc, page, img, txt,
n_chars} — paths + counts only, no content. The resolution sweep happens later via the
Qwen processor's pixel budget, so we render once at a good base resolution.

Run:  python scripts/skippy_ingest.py --classes datasheets documents writing --per_doc_cap 30
"""
import argparse, json, os, glob
import fitz  # PyMuPDF

PAI = os.path.expanduser("~/Documents/GitHub/personal-ai-framework/knowledge")
SCRATCH = os.path.expanduser("~/skippy_pixelrag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["datasheets", "documents", "writing"])
    ap.add_argument("--per_doc_cap", type=int, default=30, help="max pages rendered per PDF")
    ap.add_argument("--zoom", type=float, default=2.0, help="render zoom (2.0 ≈ 144 dpi)")
    ap.add_argument("--min_chars", type=int, default=40, help="skip near-empty pages")
    args = ap.parse_args()

    records = []
    for cls in args.classes:
        src = os.path.join(PAI, cls)
        pdfs = sorted(p for p in glob.glob(os.path.join(src, "**", "*"), recursive=True)
                      if p.lower().endswith(".pdf"))
        for pdir in ("pages", "text"):
            os.makedirs(os.path.join(SCRATCH, pdir, cls), exist_ok=True)
        n_pages_cls = 0
        for pdf in pdfs:
            doc_id = os.path.splitext(os.path.basename(pdf))[0].replace(" ", "_")[:60]
            try:
                d = fitz.open(pdf)
            except Exception as e:
                print(f"  skip {doc_id}: {type(e).__name__}")
                continue
            for pno in range(min(len(d), args.per_doc_cap)):
                page = d[pno]
                txt = page.get_text("text").strip()
                if len(txt) < args.min_chars:
                    continue  # likely a cover/figure-only page with no text layer
                pid = f"{cls}__{doc_id}__p{pno:03d}"
                img_p = os.path.join(SCRATCH, "pages", cls, f"{doc_id}_p{pno:03d}.png")
                txt_p = os.path.join(SCRATCH, "text", cls, f"{doc_id}_p{pno:03d}.txt")
                page.get_pixmap(matrix=fitz.Matrix(args.zoom, args.zoom)).save(img_p)
                open(txt_p, "w").write(txt)
                records.append({"id": pid, "class": cls, "doc": doc_id, "page": pno,
                                "img": img_p, "txt": txt_p, "n_chars": len(txt)})
                n_pages_cls += 1
            d.close()
        print(f"{cls}: {len(pdfs)} PDFs -> {n_pages_cls} text-bearing pages")

    os.makedirs(os.path.join(SCRATCH, "meta"), exist_ok=True)
    man = os.path.join(SCRATCH, "meta", "manifest.json")
    json.dump({"n_pages": len(records), "classes": args.classes, "records": records},
              open(man, "w"), indent=2)
    by_cls = {}
    for r in records:
        by_cls[r["class"]] = by_cls.get(r["class"], 0) + 1
    print(f"\nTOTAL {len(records)} pages  {by_cls}")
    print("manifest:", man, "(scratch, never committed)")


if __name__ == "__main__":
    main()
