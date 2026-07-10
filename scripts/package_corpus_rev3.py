#!/usr/bin/env python3
"""
package_corpus_rev3.py — fold qualcomm's two AI-Hub packaging patches into the corpus.

qualcomm had to patch the rev2 bundle locally to get it through Qualcomm AI Hub. This
bakes both fixes into the shipped bundle so the NEXT puller (or a fresh AI Hub run)
doesn't repeat them. Idempotent: re-running on an already-rev3 bundle is a no-op.

PATCH 1 — inline external weights (resnet50v1).
  resnet50v1.onnx keeps 101 initializers in a `resnet50v1.onnx.data` sidecar. qai_hub's
  upload didn't carry the sidecar -> "missing external weights". Inlining makes the graph
  self-contained (one file, no sidecar), which every consumer handles.

PATCH 2 — strip duplicate value_info (the three seg models).
  yolov8n-seg / yolo11s-seg / yoloe-26s-seg-pf list output0/output1 in BOTH graph.output
  AND graph.value_info. ONNX permits it; AI Hub's stricter validator rejects it. A tensor
  that is a graph output does not also need a value_info entry, so we drop the duplicates.

Both are lossless: same ops, same weights, same shapes — only the packaging changes.
Every output is re-verified to load and run in onnxruntime after patching.

Run: python3 scripts/package_corpus_rev3.py [--dir data/output/onnx_corpus_iq9]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

SEG_MODELS = ["yolov8n-seg", "yolo11s-seg", "yoloe-26s-seg-pf"]
EXTERNAL_WEIGHT_MODELS = ["resnet50v1"]


def inline_external_weights(path):
    """Rewrite a graph with external-data initializers as a single self-contained file."""
    model = onnx.load(str(path), load_external_data=True)
    onnx.save(model, str(path), save_as_external_data=False)
    n_ext = sum(1 for t in model.graph.initializer if t.data_location == 1)
    return n_ext  # 0 after save; reported pre-inline count is what mattered


def strip_duplicate_value_info(path):
    """Remove value_info entries whose name is already a graph input or output."""
    model = onnx.load(str(path), load_external_data=False)
    g = model.graph
    io_names = {o.name for o in g.output} | {i.name for i in g.input}
    dups = [vi.name for vi in g.value_info if vi.name in io_names]
    if not dups:
        return []
    kept = [vi for vi in g.value_info if vi.name not in io_names]
    del g.value_info[:]
    g.value_info.extend(kept)
    onnx.save(model, str(path))
    return dups


def verify_runs(path):
    """Load + one inference on CPU EP; return output shapes or raise."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {}
    for i in sess.get_inputs():
        shp = [d if isinstance(d, int) else 1 for d in i.shape]
        dt = np.float32 if "float" in i.type else np.int64
        if "label" in i.name:
            feed[i.name] = np.ones(shp, dtype=dt)
        elif "coord" in i.name:
            feed[i.name] = (np.ones(shp, dtype=dt) * 100)
        else:
            feed[i.name] = (np.random.randn(*shp).astype(dt) if dt == np.float32
                            else np.ones(shp, dtype=dt))
    outs = sess.run(None, feed)
    return [tuple(np.asarray(o).shape) for o in outs[:3]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/output/onnx_corpus_iq9")
    ap.add_argument("--backup", default=None, help="optional dir to copy the pre-patch bundle into")
    args = ap.parse_args()
    root = Path(args.dir)

    if args.backup:
        shutil.copytree(root, args.backup, dirs_exist_ok=True)
        print(f"backed up {root} -> {args.backup}\n")

    for model in EXTERNAL_WEIGHT_MODELS:
        p = root / f"{model}.onnx"
        pre = onnx.load(str(p), load_external_data=False).graph
        n_ext_before = sum(1 for t in pre.initializer if t.data_location == 1)
        if n_ext_before:
            inline_external_weights(p)
            sidecar = root / f"{model}.onnx.data"
            if sidecar.exists():
                sidecar.unlink()
            print(f"PATCH1 {model}: inlined {n_ext_before} external initializers, "
                  f"removed {sidecar.name}")
        else:
            print(f"PATCH1 {model}: already self-contained (no-op)")

    for model in SEG_MODELS:
        p = root / f"{model}.onnx"
        dups = strip_duplicate_value_info(p)
        print(f"PATCH2 {model}: stripped duplicate value_info {dups or '(none — no-op)'}")

    print("\nVERIFY (load + run, onnxruntime CPU EP):")
    ok = True
    for p in sorted(root.glob("*.onnx")):
        try:
            shapes = verify_runs(p)
            print(f"  ✅ {p.name:38s} -> {shapes}")
        except Exception as exc:                                    # noqa: BLE001
            ok = False
            print(f"  ❌ {p.name:38s} {type(exc).__name__}: {str(exc)[:120]}")

    if not ok:
        raise SystemExit("verification failed — NOT bumping MANIFEST")

    # Bump the manifest to rev3.
    man_path = root / "MANIFEST.json"
    man = json.loads(man_path.read_text())
    man["__meta__"]["revision"] = 3
    man["__meta__"]["revision_3_changes"] = [
        "PATCH1 (qualcomm AI Hub): inlined resnet50v1's external weights — rev2 kept them "
        "in resnet50v1.onnx.data, which qai_hub's uploader dropped ('missing external "
        "weights'). The bundle is now self-contained; there is NO .data sidecar and none "
        "is needed.",
        "PATCH2 (qualcomm AI Hub): stripped output0/output1 from graph.value_info on "
        "yolov8n-seg / yolo11s-seg / yoloe-26s-seg-pf — they were listed in both value_info "
        "and graph.output, which AI Hub's ONNX validator rejects. Ops/weights/shapes "
        "unchanged.",
    ]
    # rev2 note referenced the sidecar as required; correct it.
    for m in man.get("models", []):
        if m.get("name") == "resnet50" and "aux" in m:
            m["aux"] = "self-contained as of rev3 (weights inlined; no .data sidecar)"
    man_path.write_text(json.dumps(man, indent=2))
    print(f"\nMANIFEST bumped to rev3. Bundle self-contained, AI-Hub-clean.")


if __name__ == "__main__":
    main()
