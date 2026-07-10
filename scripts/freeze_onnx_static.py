#!/usr/bin/env python3
"""
freeze_onnx_static.py — make the vision-corpus ONNX fully static (batch=1, fixed HW).

Three of the seven corpus exports carry dynamic dims, and one carries a *shape
tensor* (`orig_im_size`, INT64) whose VALUES drive Gather ops. Benchmark harnesses
(trtexec, QNN converters) fill unbound inputs with random data — for a shape tensor
that means random int64 extents, which is either garbage output or an OOM.

This rewrites the affected models so the graph matches what MANIFEST.json already
claims: fixed batch=1, fixed spatial size, no free shape tensors. Deployment would
freeze these anyway; doing it in the shipped ONNX means every platform in the
three-way sweep converts the *same* graph.

  clip_vit_b32_visual        image[batch,3,224,224]        -> [1,3,224,224]
  efficient_sam_vitt_encoder batched_images[b,3,h,w]       -> [1,3,1024,1024]
  efficient_sam_vitt_decoder image_embeddings[b,256,64,64] -> [1,256,64,64]
                             batched_point_coords[1,1,n,2] -> [1,1,1,2]
                             batched_point_labels[1,1,n]   -> [1,1,1]
                             orig_im_size (shape tensor)   -> frozen const [1024,1024]

Run: python3 scripts/freeze_onnx_static.py data/output/onnx_corpus_iq9
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

# EfficientSAM ViT-T is trained at 1024x1024; that is the deployment resolution.
SAM_HW = 1024

# model stem -> {input name: concrete dims}  (None entry => freeze to constant, see FREEZE_CONST)
PIN_DIMS = {
    "clip_vit_b32_visual": {"image": [1, 3, 224, 224]},
    "efficient_sam_vitt_encoder": {"batched_images": [1, 3, SAM_HW, SAM_HW]},
    "efficient_sam_vitt_decoder": {
        "image_embeddings": [1, 256, 64, 64],
        "batched_point_coords": [1, 1, 1, 2],
        "batched_point_labels": [1, 1, 1],
    },
}

# model stem -> {input name: (np array value)}  — promoted from graph input to initializer
FREEZE_CONST = {
    "efficient_sam_vitt_decoder": {
        "orig_im_size": np.array([SAM_HW, SAM_HW], dtype=np.int64),
    },
}


def pin_input_dims(model, name, dims):
    for inp in model.graph.input:
        if inp.name != name:
            continue
        shape = inp.type.tensor_type.shape
        if len(shape.dim) != len(dims):
            raise ValueError(f"{name}: rank {len(shape.dim)} != {len(dims)}")
        for d, v in zip(shape.dim, dims):
            d.ClearField("dim_param")
            d.dim_value = v
        return True
    raise KeyError(f"input {name} not found")


def freeze_to_initializer(model, name, arr):
    """Move a graph input into the initializer set so its VALUE is baked in."""
    keep = [i for i in model.graph.input if i.name != name]
    if len(keep) == len(model.graph.input):
        raise KeyError(f"input {name} not found")
    del model.graph.input[:]
    model.graph.input.extend(keep)
    model.graph.initializer.append(numpy_helper.from_array(arr, name=name))


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/output/onnx_corpus_iq9")
    for stem in sorted(set(PIN_DIMS) | set(FREEZE_CONST)):
        path = root / f"{stem}.onnx"
        model = onnx.load(str(path))

        for name, dims in PIN_DIMS.get(stem, {}).items():
            pin_input_dims(model, name, dims)
        for name, arr in FREEZE_CONST.get(stem, {}).items():
            freeze_to_initializer(model, name, arr)

        # Shape inference re-derives every downstream dim from the pinned inputs;
        # without it the value_info still advertises the old symbolic extents.
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        onnx.checker.check_model(model, full_check=False)
        onnx.save(model, str(path))

        remaining = [
            (i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim])
            for i in model.graph.input
        ]
        print(f"{stem}: inputs now {remaining}")


if __name__ == "__main__":
    main()
