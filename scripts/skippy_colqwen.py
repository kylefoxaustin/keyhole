#!/usr/bin/env python3
"""
skippy_colqwen.py — FAITHFUL head-to-head: PixelRAG's 4B+screenshot-LoRA embedder variant
vs the Qwen3-VL-Embedding-2B default (which our study already used = PixelRAG's default).

Reproduces PixelRAG's exact direct_gpu embedding recipe (embed/src/pixelrag_embed/embed.py):
  - load Qwen3-VL-4B-Instruct + the screenshot-LoRA, with the BiQwen3->ConditionalGeneration
    adapter KEY REMAP + set_peft_model_state_dict + merge_and_unload  (the step the naive
    PeftModel.from_pretrained skips -> LoRA silently no-ops, which broke the first attempt)
  - prompt = system(instruction)/user(image), add_generation_prompt=True
  - forward through model.model -> last_hidden_state, last-non-pad-token pool, L2 normalize
  - fp32 Conv3d patch-embed workaround if cuDNN < 9.15

Compares page/doc retrieval to the 2B base on the identical Skippy corpus + queries @768px.
Aggregate only. Run: python scripts/skippy_colqwen.py --resolution 768

OUTCOME (2026-06-14): this hand-rolled forward did NOT reach fidelity — even with the
LoRA loaded (keys already matched, remap a no-op) and the exact prompt/pool recipe, the
single-vector embeddings come out near-random (page R@5 0.05 / doc R@5 0.28 vs the 2B's
0.69 / 0.83). That is a REPLICATION-FIDELITY failure, NOT evidence the variant is bad —
extracting embeddings from the merged generative model via model.model + last-token pool
clearly differs from PixelRAG's serving path in some way we didn't capture. The clean way
to test the 4B+LoRA variant is to run PixelRAG's own `pixelrag embed` CLI (vLLM/sglang/
direct_gpu serve), a separate task. KEY POINT: PixelRAG's DEFAULT & recommended embedder
is Qwen3-VL-Embedding-2B (embed.py default) — the exact model our study already uses — so
our results faithfully represent PixelRAG; the 4B+LoRA head-to-head is not needed for the
conclusion. Kept here as a documented dead-end, not a finding.
"""
import argparse, json, math, os, re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRATCH = os.path.expanduser("~/skippy_pixelrag")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "Qwen/Qwen3-VL-4B-Instruct"
LORA = "Chrisyichuan/qwen3vl-4b-wiki-screenshot-multik-3x-lora"
INSTR = "Represent the user's input."


def page_metrics(ranks):
    def rk(k): return sum(1 for r in ranks if r is not None and r < k) / len(ranks)
    nd = sum((1.0 / math.log2(r + 2)) for r in ranks if r is not None and r < 5) / len(ranks)
    return {"recall@1": round(rk(1), 4), "recall@5": round(rk(5), 4), "ndcg@5": round(nd, 4)}


def load_lora_merged():
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as MC
    except Exception:
        from transformers import AutoModelForImageTextToText as MC
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    model = MC.from_pretrained(BASE, trust_remote_code=True, torch_dtype=torch.bfloat16)
    apath = snapshot_download(LORA)
    raw = load_file(os.path.join(apath, "adapter_model.safetensors"))
    remap = {re.sub(r"^(base_model\.model\.)(language_model\.|visual\.)", r"\1model.\2", k): v
             for k, v in raw.items()}
    n = sum(1 for a, b in zip(raw, remap) if a != b)
    model = PeftModel.from_pretrained(model, apath)
    set_peft_model_state_dict(model, remap)
    model = model.merge_and_unload().cuda().eval()
    print(f"LoRA merged ({n}/{len(raw)} keys remapped)")
    if torch.backends.cudnn.version() < 91500:
        pe = model.model.visual.patch_embed
        def fp32_pe(hs, _pe=pe):
            conv = _pe.proj
            x = hs.view(-1, _pe.in_channels, _pe.temporal_patch_size, _pe.patch_size, _pe.patch_size)
            ow, ob = conv.weight.data, conv.bias.data
            conv.weight.data, conv.bias.data = ow.float(), ob.float()
            out = conv(x.float()).view(-1, _pe.embed_dim)
            conv.weight.data, conv.bias.data = ow, ob
            return out.to(torch.bfloat16)
        pe.forward = fp32_pe
        print("applied fp32 Conv3d patch-embed workaround")
    return model, proc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resolution", type=int, default=768)
    args = ap.parse_args()
    man = json.load(open(f"{SCRATCH}/meta/manifest.json"))["records"]
    id2pos = {r["id"]: i for i, r in enumerate(man)}
    page_doc = [r["doc"] for r in man]
    qd = [q for q in json.load(open(f"{SCRATCH}/meta/queries.json"))["queries"] if q["gold_id"] in id2pos]
    queries = [q["query"] for q in qd]; gold = [id2pos[q["gold_id"]] for q in qd]

    model, proc = load_lora_merged()

    @torch.no_grad()
    def embed(content):
        msg = [{"role": "system", "content": [{"type": "text", "text": INSTR}]},
               {"role": "user", "content": content}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        imgs = [c["image"] for c in content if c.get("type") == "image"]
        inp = proc(text=[text], images=imgs or None, return_tensors="pt", padding=True).to("cuda")
        out = model.model(**inp)
        h = out.last_hidden_state                       # (1, seq, D), post-RMSNorm
        idx = inp["attention_mask"].sum(1) - 1
        pooled = h[torch.arange(h.size(0), device=h.device), idx]
        return F.normalize(pooled.float(), dim=-1)[0].cpu().numpy()

    import time; t0 = time.time()
    q_emb = np.stack([embed([{"type": "text", "text": q}]) for q in queries])
    p_emb = np.zeros((len(man), q_emb.shape[1]), dtype=np.float32)
    for i, r in enumerate(man):
        im = Image.open(r["img"]).convert("RGB"); im.thumbnail((args.resolution, args.resolution))
        p_emb[i] = embed([{"type": "image", "image": im}])
        if (i + 1) % 200 == 0: print(f"  embedded {i+1}/{len(man)} ({time.time()-t0:.0f}s)")

    order = np.argsort(-(q_emb @ p_emb.T), axis=1)
    ranks = [int(np.where(order[i] == g)[0][0]) if g in order[i] else None for i, g in enumerate(gold)]
    def doc_r(k): return round(sum(1 for i, g in enumerate(gold) if any(page_doc[p] == page_doc[g] for p in order[i][:k])) / len(gold), 4)
    res = {"__meta__": {"embedder": f"{BASE} + screenshot-LoRA (faithful: remap+merge, system/user, model.model)",
                        "corpus": len(man), "queries": len(queries),
                        "compare_to": "Qwen3-VL-Embedding-2B base (PixelRAG default) pixel_VL R@5 0.692",
                        "note": "does PixelRAG's 4B+LoRA variant beat the 2B default? aggregate only."},
           "pixelrag_4b_lora": {"page": page_metrics(ranks), "doc_recall@1": doc_r(1), "doc_recall@5": doc_r(5)}}
    out = f"{REPO}/data/output/skippy_colqwen.json"
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res["pixelrag_4b_lora"], indent=2)); print("wrote", out)


if __name__ == "__main__":
    main()
