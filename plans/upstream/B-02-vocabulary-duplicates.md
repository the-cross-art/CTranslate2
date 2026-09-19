# B-02 · T5 vocabularies contain duplicate `<extra_id_*>` tokens at misaligned ids

**Severity:** medium — latent for translation, wrong for span-infilling.
**Scope:** Python converter only (~10 lines).
**Status:** reproduced on this machine. **Not fixed for T5/mT5.** No upstream issue exists.
(The same bug class is already fixed for UMT5 on `umt5/phase1-converter`.)

---

## 1. Symptom

`t5-small`, stock CTranslate2 `master`:

```
config.vocab_size                 : 32128
len(tokenizer vocab)              : 32100
gap CTranslate2 must pad          :    28
<extra_id_*> already in tokenizer :   100

CTranslate2 vocabulary: 32128 entries, 28 DUPLICATES

  <extra_id_0>:  HuggingFace id 32099  ->  CTranslate2 index 32100
  <extra_id_5>:  HuggingFace id 32094  ->  CTranslate2 index 32105
```

Reproduce: `plans/onboarding/02-HANDS-ON.md`, Step 4.

---

## 2. Root cause

`python/ctranslate2/converters/transformers.py:1257-1264`:

```python
def get_vocabulary(self, model, tokenizer):
    tokens = super().get_vocabulary(model, tokenizer)
    extra_ids = model.config.vocab_size - len(tokens)
    for i in range(extra_ids):
        tokens.append("<extra_id_%d>" % i)
    return tokens
```

The assumption is that the gap between `config.vocab_size` and the tokenizer's vocabulary
consists of *missing* sentinel tokens. For `t5-small` that's false: all 100 `<extra_id_*>`
are **already present**, and the 28-token gap is **padding** (T5 pads its vocabulary to a
multiple of 64/128 for GPU efficiency).

So the padding slots get names that collide with real tokens. The list is the right *length*,
so nothing raises — but `<extra_id_0>` now appears twice, at 32099 and 32100.

Any token→id lookup in CTranslate2 resolves the collision one way; the tokenizer resolves it
the other. Ids drift by one.

Measured across the family:

| model | vocab_size | tokenizer | gap | sentinels present | duplicates |
|---|---|---|---|---|---|
| `t5-small` | 32128 | 32100 | 28 | 100 | **28** |
| `google/mt5-small` | 250112 | 250100 | 12 | 0 (they're spelled `▁<extra_id_N>`) | 0 |
| `google/umt5-small` | 256384 | 256300 | 84 | 300 | **84** → fixed by `UMT5Loader` |

mT5 escapes only by accident: its sentinels carry a SentencePiece `▁` prefix, so the appended
bare names don't collide.

---

## 3. Why it went unnoticed

Ordinary translation never emits a sentinel, so the mis-named entries sit at the end of the
list doing nothing. It bites when T5 is used for **span infilling** — the fill-in-the-blank
objective T5 was actually pretrained on, and the reason the sentinels exist.

---

## 4. Proposed fix — NOT yet implemented for T5/mT5

Pad with names that cannot collide, and assert the result:

```python
def get_vocabulary(self, model, tokenizer):
    tokens = ModelLoader.get_vocabulary(self, model, tokenizer)

    # Only append sentinels that the tokenizer does not already provide; any remaining
    # gap is vocabulary padding and gets placeholder names.
    known = set(tokens)
    for i in range(model.config.vocab_size - len(tokens)):
        name = "<extra_id_%d>" % i
        tokens.append(name if name not in known else "<unused_%d>" % i)

    if len(tokens) != model.config.vocab_size:
        raise ValueError(...)
    return tokens
```

The exact shape used for UMT5 is on `umt5/phase1-converter`
(`transformers.py`, `UMT5Loader.get_vocabulary`) — it pads unconditionally with
`<unused_%d>` because UMT5's tokenizer always ships all 300 sentinels.

### ⚠️ The blocking question: this changes existing models

Fixing this changes the vocabulary written for **every T5 model**, so `model.bin` /
`shared_vocabulary.json` stop being byte-identical to previous releases. Consequences to
think through *before* writing code:

- Is that acceptable to maintainers, or does it need a compatibility flag?
- Does anything downstream depend on the current (wrong) names?
- Does the repo's own T5 test need updating?

This is the one real design decision here. **Ask in the issue; don't decide it alone.**

---

## 5. Evidence block for the issue

```
CTranslate2 4.8.2 (master), transformers 5.17.0

$ ct2-transformers-converter --model t5-small --output_dir /tmp/ct2-t5

config.vocab_size                 : 32128
len(tokenizer.get_vocab())        : 32100
gap                               :    28
<extra_id_*> already in tokenizer :   100
CTranslate2 vocabulary            : 32128 entries, 28 DUPLICATES

  <extra_id_0>: HF id 32099  ->  CT2 index 32100
  <extra_id_5>: HF id 32094  ->  CT2 index 32105

Cause: T5Loader.get_vocabulary (converters/transformers.py:1257-1264) pads the vocabulary
to config.vocab_size by appending <extra_id_i> names that the tokenizer already provides.
The gap is padding to a multiple of 64, not missing sentinels. The list length is correct
so no error is raised, but sentinel ids shift by one relative to the tokenizer.

Latent for translation; incorrect for span-infilling, T5's own pretraining objective.
google/umt5-small shows the same pattern with an 84-token gap.
```
