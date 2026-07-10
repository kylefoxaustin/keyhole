#!/usr/bin/env python3
"""
quantize_corpus_qdq.py — produce QDQ (explicitly-quantized) INT8 ONNX for the corpus.

WHY THIS EXISTS. `trtexec --int8` with no calibration cache uses TensorRT's *implicit*
quantization: it assigns arbitrary dynamic ranges, then mostly declines to select INT8
kernels and instead inserts reformat layers at every precision boundary. Measured on
this corpus: yolov8n-seg got 41% MORE layers and ran SLOWER than FP16; efficientsam's
encoder ran 2x slower; clip_vit's latency was identical to FP16 (nothing ran in INT8);
three models did not build at all. Those numbers describe TensorRT's fallback machinery,
not INT8 silicon — and diffing them against a calibrated QNN/HTP result would produce a
confident, wrong cross-platform claim.

Explicit quantization fixes both halves: Q/DQ nodes carry real calibrated scales, so
TensorRT selects genuine INT8 kernels (latency becomes representative) AND the numerics
mean something (accuracy becomes quotable).

TensorRT requires SYMMETRIC quantization (zero-point 0) for both weights and
activations; asymmetric Q/DQ silently degrades to slower kernels.

Calibration uses real frames from the bake-off clip, not random noise: activation
ranges from noise are meaningless, and the whole point of calibration is the ranges.

Run: ~/.virtualenvs/keyhole/bin/python scripts/quantize_corpus_qdq.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "output" / "onnx_corpus_iq9"
DST = REPO / "data" / "output" / "onnx_corpus_qdq"
FRAMES = sorted((REPO / "data" / "frames" / "EW_clip_720p").glob("*.png"))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

# Small frame count -> synthesise extra calibration samples with deterministic crops,
# so activation ranges see more than 14 framings of the same scene.
CROPS = [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.7, 0.7),
         (0.3, 0.3, 1.0, 1.0), (0.15, 0.15, 0.85, 0.85)]


def load_batch(size, mean=None, std=None, scale=255.0):
    """[N,3,H,W] float32 calibration tensors from the bake-off frames."""
    out = []
    for path in FRAMES:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        for l, t, r, b in CROPS:
            crop = img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
            arr = np.asarray(crop.resize((size, size), Image.BILINEAR), np.float32) / scale
            if mean is not None:
                arr = (arr - mean) / std
            out.append(arr.transpose(2, 0, 1)[None])
    return out


class Reader(CalibrationDataReader):
    def __init__(self, feeds):
        self.it = iter(feeds)

    def get_next(self):
        return next(self.it, None)


def encoder_embeddings(encoder_onnx, n=16):
    """The decoder's input is the encoder's output. Calibrating it on random tensors
    would fabricate the very activation ranges we are trying to measure, so run the
    real encoder over real frames and calibrate on what it actually emits."""
    sess = ort.InferenceSession(str(encoder_onnx), providers=["CPUExecutionProvider"])
    feeds = []
    for img in load_batch(1024, scale=255.0)[:n]:
        emb = sess.run(None, {"batched_images": img})[0]
        feeds.append({
            "image_embeddings": emb,
            "batched_point_coords": np.array([[[[512.0, 512.0]]]], np.float32),
            "batched_point_labels": np.array([[[1.0]]], np.float32),
        })
    return feeds


SPECS = {
    "resnet50v1":        lambda: [{"input": x} for x in load_batch(224, IMAGENET_MEAN, IMAGENET_STD)],
    "clip_vit_b32_visual": lambda: [{"image": x} for x in load_batch(224, CLIP_MEAN, CLIP_STD)],
    "yolov8n-seg":       lambda: [{"images": x} for x in load_batch(640)],
    "yolo11s-seg":       lambda: [{"images": x} for x in load_batch(640)],
    "yoloe-26s-seg-pf":  lambda: [{"images": x} for x in load_batch(640)],
    "efficient_sam_vitt_encoder": lambda: [{"batched_images": x} for x in load_batch(1024)],
    "efficient_sam_vitt_decoder": lambda: encoder_embeddings(SRC / "efficient_sam_vitt_encoder.onnx"),
}


def main():
    if not FRAMES:
        sys.exit(f"no calibration frames under {FRAMES}")
    DST.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or list(SPECS)
    print(f"{len(FRAMES)} frames x {len(CROPS)} crops = "
          f"{len(FRAMES)*len(CROPS)} calibration samples\n")

    for model in only:
        src = SRC / f"{model}.onnx"
        prepped = DST / f"{model}.prep.onnx"
        out = DST / f"{model}.onnx"
        print(f"=== {model}", flush=True)
        try:
            # skip_symbolic_shape: these graphs are already fully static (freeze_onnx_static.py),
            # so ORT's symbolic pass has nothing to solve — and it crashes on clip_vit
            # (AssertionError) and yoloe (opset-20 node it can't dispatch) trying.
            quant_pre_process(str(src), str(prepped), skip_symbolic_shape=True)
            feeds = SPECS[model]()
            quantize_static(
                str(prepped), str(out),
                calibration_data_reader=Reader(feeds),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={
                    # TensorRT needs zero-point 0 on both sides or it leaves the fast path.
                    "ActivationSymmetric": True,
                    "WeightSymmetric": True,
                    # ORT otherwise quantizes bias to INT32 and wraps it in a
                    # DequantizeLinear. TensorRT rejects DQ on INT32 ("only activation
                    # datatypes allowed as input to this layer") and the parse dies.
                    # TRT folds bias into the INT8 conv itself, so leave it alone.
                    "QuantizeBias": False,
                },
            )
            mb = out.stat().st_size / 1e6
            print(f"    -> {out.name}  {mb:.1f} MB  ({len(feeds)} calib samples)\n", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f"    !! FAILED: {type(exc).__name__}: {str(exc)[:220]}\n", flush=True)
        finally:
            prepped.unlink(missing_ok=True)
            for stray in DST.glob("*.prep.onnx_data"):
                stray.unlink(missing_ok=True)

    # External-data sidecars must travel with their graph.
    for extra in SRC.glob("*.onnx.data"):
        if not (DST / extra.name).exists():
            shutil.copy(extra, DST / extra.name)


if __name__ == "__main__":
    main()
