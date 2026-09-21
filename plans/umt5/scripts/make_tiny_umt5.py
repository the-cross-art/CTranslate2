#!/usr/bin/env python
"""Create a tiny randomly-initialised UMT5 for fast iteration.

Real google/umt5-small is 1.2 GB, mostly because its vocabulary is 256384 x 512.
This builds a structurally identical but tiny UMT5ForConditionalGeneration: same
architecture (per-layer relative attention bias, gated-gelu FFN, untied lm_head),
a few MB, converts in seconds.

Use it to iterate on converter/runtime changes. Use the REAL model to judge accuracy:
random weights tell you nothing about output quality.

Usage:  make_tiny_umt5.py [out_dir]   (default /tmp/umt5-tiny)
"""
import json, os, shutil, sys
import torch
from transformers import UMT5Config, UMT5ForConditionalGeneration

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/umt5-tiny"
    torch.manual_seed(0)

    # vocab_size must match the tokenizer we borrow below (t5-small's, 32100 tokens
    # padded to 32128) or the converter rightly refuses: the embedding matrix and the
    # vocabulary have to line up.
    cfg = UMT5Config(
        vocab_size=32128, d_model=64, d_kv=8, d_ff=128,
        num_layers=3, num_decoder_layers=3, num_heads=4,
        relative_attention_num_buckets=32, relative_attention_max_distance=128,
        feed_forward_proj="gated-gelu", decoder_start_token_id=0,
        pad_token_id=0, eos_token_id=1,
    )
    model = UMT5ForConditionalGeneration(cfg).eval()

    # Random init can leave the per-layer bias tables suspiciously similar; make each
    # layer's table unmistakably distinct so the shared-vs-per-layer detection is exercised.
    with torch.no_grad():
        for stack in (model.encoder, model.decoder):
            for i, block in enumerate(stack.block):
                w = block.layer[0].SelfAttention.relative_attention_bias.weight
                w.normal_(mean=float(i + 1), std=0.5)

    shutil.rmtree(out, ignore_errors=True)
    model.save_pretrained(out, safe_serialization=False)

    # save_pretrained writes model_type correctly; make sure (google's own repos omit it)
    p = os.path.join(out, "config.json")
    c = json.load(open(p)); c["model_type"] = "umt5"
    json.dump(c, open(p, "w"), indent=2)

    # The converter needs a tokenizer. Borrow t5-small's (32100 tokens); cfg.vocab_size
    # above is set to 32128 to match it after padding.
    from huggingface_hub import hf_hub_download
    src = "t5-small"
    for f in ("spiece.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        try:
            shutil.copy(hf_hub_download(src, f), os.path.join(out, f))
        except Exception as e:
            print(f"  note: could not fetch {f} ({type(e).__name__})")

    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    n_enc = len(model.encoder.block); n_dec = len(model.decoder.block)
    print(f"created {out}  ({size/1e6:.1f} MB)")
    print(f"  {n_enc} encoder + {n_dec} decoder layers, d_model={cfg.d_model}, "
          f"vocab={cfg.vocab_size}")
    print(f"  per-layer relative attention bias: yes (shape "
          f"{tuple(model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight.shape)})")
    print("\n  NOTE: random weights. Structure is real; output quality is meaningless.")

if __name__ == "__main__":
    main()
