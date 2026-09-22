# Issue draft — ready to paste

**Repo:** https://github.com/OpenNMT/CTranslate2/issues/new
**Title:** `T5 vocabularies contain duplicate <extra_id_*> tokens with shifted ids`
**Labels to request:** `bug`

> File this **second**, after B-01, and only once you've reproduced it yourself.
> B-02 is smaller and cleaner, but its fix changes output for every T5 model — so the
> maintainer answer matters more than the code.

---

## Description

`T5Loader.get_vocabulary` pads the vocabulary to `config.vocab_size` by appending
`<extra_id_i>` names that the tokenizer **already provides**. The result is a
correctly-sized vocabulary containing duplicate token names, with sentinel ids shifted
relative to the tokenizer.

## To reproduce

```bash
ct2-transformers-converter --model t5-small --output_dir /tmp/ct2-t5
```

```python
import json
from transformers import AutoTokenizer, AutoConfig
cfg = AutoConfig.from_pretrained("t5-small")
hf  = AutoTokenizer.from_pretrained("t5-small", use_fast=False).get_vocab()
ct2 = json.load(open("/tmp/ct2-t5/shared_vocabulary.json"))

print("config.vocab_size :", cfg.vocab_size)          # 32128
print("tokenizer vocab   :", len(hf))                 # 32100
print("gap               :", cfg.vocab_size - len(hf))# 28
print("<extra_id_*> in tokenizer:", sum(1 for k in hf if k.startswith("<extra_id_")))  # 100
print("duplicates in CT2 vocab  :", len(ct2) - len(set(ct2)))                          # 28

idx = {t: i for i, t in enumerate(ct2)}
for t in ("<extra_id_0>", "<extra_id_5>"):
    print(f"{t}: HF id {hf[t]} -> CT2 index {idx[t]}")
```

```
<extra_id_0>: HF id 32099 -> CT2 index 32100
<extra_id_5>: HF id 32094 -> CT2 index 32105
```

## Cause

`python/ctranslate2/converters/transformers.py:1257-1264`:

```python
def get_vocabulary(self, model, tokenizer):
    tokens = super().get_vocabulary(model, tokenizer)
    extra_ids = model.config.vocab_size - len(tokens)
    for i in range(extra_ids):
        tokens.append("<extra_id_%d>" % i)
    return tokens
```

The assumption is that the gap consists of *missing* sentinels. For `t5-small` it doesn't:
all 100 `<extra_id_*>` are already present, and the 28-token gap is padding to a multiple of
64. The appended names therefore collide with real tokens. The list length is right, so
nothing raises.

Across the family:

| model | vocab_size | tokenizer | gap | sentinels present | duplicates |
|---|---|---|---|---|---|
| `t5-small` | 32128 | 32100 | 28 | 100 | **28** |
| `google/mt5-small` | 250112 | 250100 | 12 | 0 (spelled `▁<extra_id_N>`) | 0 |
| `google/umt5-small` | 256384 | 256300 | 84 | 300 | **84** |

mT5 escapes only because its sentinels carry a SentencePiece `▁` prefix, so the appended
bare names don't collide.

## Impact

Latent for translation — which is presumably why it hasn't surfaced — since ordinary decoding
never emits a sentinel. It matters for span-infilling, the objective T5 was pretrained on and
the reason the sentinels exist.

## Possible fix

Only append sentinels the tokenizer doesn't already provide; give any remaining padding
non-colliding names, and assert the final length:

```python
def get_vocabulary(self, model, tokenizer):
    tokens = ModelLoader.get_vocabulary(self, model, tokenizer)
    known = set(tokens)
    for i in range(model.config.vocab_size - len(tokens)):
        name = "<extra_id_%d>" % i
        tokens.append(name if name not in known else "<unused_%d>" % i)
    if len(tokens) != model.config.vocab_size:
        raise ValueError(...)
    return tokens
```

## Question before I open a PR

This changes the vocabulary written for **every** T5 model, so `shared_vocabulary.json` stops
matching previously converted models. Is that acceptable, or would you want it behind a flag?
That's the main decision here — happy to implement whichever you prefer.

## Environment

```
CTranslate2 4.8.2 (master), transformers 5.17.0, Python 3.12, macOS arm64
```

---

## AI-use disclosure — REQUIRED

Same as B-01. `CONTRIBUTING.md` lines 31-37. Write it in your own words, and only after you
have reproduced the numbers yourself.
