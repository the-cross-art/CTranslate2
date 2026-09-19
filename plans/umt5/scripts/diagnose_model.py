#!/usr/bin/env python
"""Dump everything the UMT5 conversion plan needs about a checkpoint.

Usage:
    .venv-umt5/bin/python plans/umt5/scripts/diagnose_model.py <model-id-or-local-path>

Produces evidence E-03 (config), E-05 (per-layer bias layout), E-06 (vocabulary),
plus the N-02 tie_word_embeddings check. Read-only; downloads nothing it can avoid.
"""
import argparse, json, os, sys, itertools

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--skip-weights", action="store_true",
                    help="config + tokenizer only (no multi-GB download)")
    a = ap.parse_args()
    mid = a.model

    import torch, transformers
    print("=" * 72)
    print("ENV  python %s | torch %s | transformers %s"
          % (sys.version.split()[0], torch.__version__, transformers.__version__))
    print("MODEL", mid)

    # ---- raw config.json (what is actually on disk/hub) -------------------
    print("=" * 72); print("[1] RAW config.json")
    raw = None
    try:
        if os.path.isdir(mid):
            p = os.path.join(mid, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(mid, "config.json")
        raw = json.load(open(p))
        print("    path:", p)
        for k in ("model_type", "architectures", "tie_word_embeddings", "vocab_size",
                  "num_layers", "num_decoder_layers", "d_model", "num_heads", "d_kv", "d_ff",
                  "feed_forward_proj", "dense_act_fn", "is_gated_act",
                  "relative_attention_num_buckets", "relative_attention_max_distance",
                  "decoder_start_token_id", "pad_token_id", "eos_token_id",
                  "layer_norm_epsilon", "transformers_version", "tokenizer_class"):
            print(f"    {k:33s} {raw.get(k, '<absent>')!r}")
        if "model_type" not in raw:
            print("    >>> N-01: no 'model_type' key -> AutoConfig will FAIL (see F-03)")
        extra = set(raw) - {
            "model_type","architectures","tie_word_embeddings","vocab_size","num_layers",
            "num_decoder_layers","d_model","num_heads","d_kv","d_ff","feed_forward_proj",
            "dense_act_fn","is_gated_act","relative_attention_num_buckets",
            "relative_attention_max_distance","decoder_start_token_id","pad_token_id",
            "eos_token_id","layer_norm_epsilon","transformers_version","tokenizer_class",
            "dropout_rate","initializer_factor","is_encoder_decoder","use_cache",
            "torch_dtype","dtype","classifier_dropout","max_new_tokens"}
        if extra:
            print("    unexpected/extra keys:", sorted(extra))
    except Exception as e:
        print("    FAILED:", type(e).__name__, str(e)[:200])

    # ---- how the converter will dispatch ---------------------------------
    print("=" * 72); print("[2] DISPATCH (what CTranslate2 does at transformers.py:107)")
    cfg = None
    try:
        cfg = transformers.AutoConfig.from_pretrained(mid)
        print("    AutoConfig OK ->", type(cfg).__name__)
        print("    registry key CTranslate2 will look up:", type(cfg).__name__)
    except Exception as e:
        print("    AutoConfig FAILS:", type(e).__name__, str(e)[:160])
        print("    >>> this is N-01, it fires BEFORE the loader registry")
        try:
            cfg = transformers.UMT5Config.from_pretrained(mid)
            print("    UMT5Config.from_pretrained OK (workaround: add model_type to config.json)")
        except Exception as e2:
            print("    UMT5Config also failed:", type(e2).__name__, str(e2)[:160])

    # ---- N-02 tie_word_embeddings ----------------------------------------
    if cfg is not None and raw is not None:
        print("=" * 72); print("[3] N-02 tie_word_embeddings")
        print("    config.json     :", raw.get("tie_word_embeddings", "<absent>"))
        print("    loaded config   :", getattr(cfg, "tie_word_embeddings", "<absent>"))
        if raw.get("tie_word_embeddings") is False and getattr(cfg, "tie_word_embeddings", None) is True:
            print("    >>> FORCED True by transformers -> CT2 will set scale_outputs =",
                  getattr(cfg, "d_model", "?"), "** -0.5   (see F-04 / D-11)")

    # ---- tokenizer / vocabulary ------------------------------------------
    print("=" * 72); print("[4] VOCABULARY (K-12)")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(mid, use_fast=False)
        v = tok.get_vocab()
        vs = getattr(cfg, "vocab_size", None) if cfg is not None else None
        n_extra = sum(1 for k in v if k.startswith("<extra_id_"))
        print("    tokenizer        :", type(tok).__name__)
        print("    config.vocab_size:", vs)
        print("    len(get_vocab()) :", len(v))
        if vs is not None:
            print("    gap              :", vs - len(v))
        print("    <extra_id_*> already present:", n_extra)
        print("    pad/eos/unk      :", tok.pad_token, tok.eos_token, tok.unk_token)
        if n_extra and vs is not None and vs - len(v) > 0:
            print("    >>> K-12: inherited loader would append", vs - len(v),
                  "DUPLICATE <extra_id_*> tokens")
    except Exception as e:
        print("    FAILED:", type(e).__name__, str(e)[:200])

    if a.skip_weights:
        print("=" * 72); print("[5] WEIGHTS: skipped (--skip-weights)"); return

    # ---- weights: bias layout + lm_head ----------------------------------
    print("=" * 72); print("[5] WEIGHTS (E-05: per-layer bias layout)")
    sd = None
    try:
        from huggingface_hub import hf_hub_download
        for fn in ("model.safetensors", "pytorch_model.bin"):
            path = os.path.join(mid, fn) if os.path.isdir(mid) else None
            if path and os.path.exists(path):
                pass
            elif os.path.isdir(mid):
                continue
            else:
                try: path = hf_hub_download(mid, fn)
                except Exception: continue
            if fn.endswith(".safetensors"):
                from safetensors.torch import load_file
                sd = load_file(path)
            else:
                sd = torch.load(path, map_location="cpu", weights_only=True)
            print("    loaded:", path); break
        if sd is None:
            print("    no single-file checkpoint found (sharded?); "
                  "load the model with from_pretrained and inspect state_dict() manually")
            return
    except Exception as e:
        print("    FAILED:", type(e).__name__, str(e)[:200]); return

    for stack in ("encoder", "decoder"):
        ks = [k for k in sd if k.startswith(stack) and "relative_attention_bias" in k
              and "SelfAttention" in k]
        ks.sort(key=lambda k: int(k.split(".block.")[1].split(".")[0]))
        print(f"    {stack}: {len(ks)} self-attention bias tables")
        if not ks:
            continue
        print(f"      shape: {tuple(sd[ks[0]].shape)}")
        distinct = all(not torch.equal(sd[x], sd[y])
                       for x, y in itertools.combinations(ks, 2))
        print(f"      all pairwise distinct: {distinct}")
        if len(ks) == 1:
            print("      >>> T5-STYLE (shared bias). Phase 2 (C++) NOT needed for this model.")
        elif distinct:
            print("      >>> UMT5-STYLE (per-layer bias). Phase 2 (C++) REQUIRED.")
        else:
            print("      >>> MIXED: some tables identical. See error-catalog X-05.")
    xattn = [k for k in sd if "EncDecAttention.relative_attention_bias" in k]
    if xattn:
        print("    note: cross-attention bias tensors present but unused by HF:", len(xattn))

    print("=" * 72); print("[6] lm_head / embedding tying")
    lm = next((k for k in sd if k.endswith("lm_head.weight")), None)
    sh = next((k for k in sd if k.endswith("shared.weight")), None)
    if lm and sh:
        eq = bool(torch.equal(sd[lm], sd[sh]))
        print(f"    {lm} {tuple(sd[lm].shape)}")
        print(f"    {sh} {tuple(sd[sh].shape)}")
        print("    values equal (actually tied):", eq)
        if not eq:
            print("    max abs diff:", float((sd[lm].float() - sd[sh].float()).abs().max()))
            print("    >>> checkpoint is UNTIED. Cross-check against [3].")
    else:
        print("    lm_head.weight present:", bool(lm), "| shared.weight present:", bool(sh))
    print("=" * 72); print("done")

if __name__ == "__main__":
    main()
