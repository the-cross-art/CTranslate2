# B-01 · mT5 decoder output is wrongly rescaled under transformers 5.x

**Severity:** high — silently destroys mT5 output quality.
**Scope:** Python converter only (~10 lines).
**Status:** reproduced on this machine. **Not fixed.** No upstream issue exists.

---

## 1. Symptom

`google/mt5-small`, stock CTranslate2 `master`, no local modifications:

```
HF  log-probs: [-23.28 -38.27 -65.91 -20.92 -34.24 -72.02 -81.49 -23.53]
CT2 log-probs: [-12.03 -11.93 -12.86 -12.01 -12.21 -13.28 -13.70 -11.15]
max |diff| = 67.79   mean |diff| = 32.56
```

`log(1/250112) = -12.43`. CTranslate2's output sits right at that value for every token —
a **near-uniform distribution**, i.e. the model expressing no preference at all. Hugging
Face's distribution is sharply peaked.

Control, identical script and machine: **`t5-small` → max |diff| = 1.07e-06.**

Reproduce: `plans/onboarding/02-HANDS-ON.md`, Step 5.

---

## 2. Root cause

`transformers` v5 gave T5 a dedicated config flag for the decoder-output rescaling, replacing
the old `tie_word_embeddings` proxy. The three models in the family now disagree:

| model file | condition guarding `* (d_model ** -0.5)` |
|---|---|
| `modeling_t5.py:1044` | `if self.config.scale_decoder_outputs:` |
| `modeling_mt5.py` | **no rescaling anywhere** — `MT5ForConditionalGeneration` defines its own `forward` |
| `modeling_umt5.py:1052` | `if self.config.tie_word_embeddings:` (still the old proxy) |

Config values:

| model | `tie_word_embeddings` | `scale_decoder_outputs` | HF rescales? |
|---|---|---|---|
| `t5-small` | True | **True** | yes |
| `google/mt5-small` | True | **absent** | **no** |
| `google/umt5-small` | True (forced) | absent | yes |

CTranslate2 still keys off the old proxy — `python/ctranslate2/converters/transformers.py:1252`:

```python
if model.config.tie_word_embeddings:
    spec.decoder.scale_outputs = model.config.d_model**-0.5
```

For mT5 that is now simply wrong: `tie_word_embeddings` is `True`, so CTranslate2 multiplies
the decoder output by `1/sqrt(512) = 0.0442` before the vocabulary projection, while Hugging
Face multiplies by nothing. Scaling the hidden state by 0.044 flattens the logits, and softmax
turns a flat logit vector into a uniform distribution.

At runtime the scale is applied at `src/layers/transformer.cc:880-881`. **The C++ is fine —
the converter is feeding it a value it shouldn't.**

---

## 3. Why it went unnoticed

- Multiplying all logits by a positive constant **does not change `argmax`**, so greedy
  decoding still emits plausible text. Only scores, sampling, and length-penalised beam
  ranking are corrupted.
- The repo's mT5 test (`python/tests/test_transformers.py:106`,
  `ml6team/mt5-small-german-query-generation`) asserts **output tokens**, not scores — so it
  passes while the distribution underneath is wrong.

---

## 4. Proposed fix — NOT yet implemented

Read the new flag when present, fall back to the old behaviour when it isn't:

```python
# T5Loader.get_model_spec, replacing transformers.py:1252-1253
scale_decoder_outputs = getattr(model.config, "scale_decoder_outputs", None)
if scale_decoder_outputs is None:
    # transformers < 5 had no dedicated flag and used tie_word_embeddings as a proxy
    scale_decoder_outputs = model.config.tie_word_embeddings
if scale_decoder_outputs:
    spec.decoder.scale_outputs = model.config.d_model**-0.5
```

⚠️ **This is wrong for mT5 as written** and must not be shipped unexamined: mT5's config has
no `scale_decoder_outputs`, so the fallback fires and preserves exactly the bug. The fix has
to key off *behaviour*, not just the flag. Options:

- **(a)** Override `get_model_spec` in `MT5Loader` to never set `scale_outputs`, matching
  `modeling_mt5.py`. Narrow, obviously correct for transformers 5.x.
- **(b)** Branch on the installed `transformers` version.
- **(c)** Ask upstream whether mT5's missing rescale is intentional or itself a
  `transformers` bug. **Do this first** — if it's a `transformers` regression, the right fix
  is there, not here.

**(c) then (a)** is the sensible order.

### Open questions to settle before coding

1. Does the dropped rescale in `modeling_mt5.py` match mT5's original T5X training? If not,
   `transformers` is the buggy layer.
2. Which `transformers` versions must the converter support? `python/tests/requirements.txt`
   pins `5.9.0.*`; we measured on `5.17.0`. **Verify the behaviour on 5.9.0 before filing** —
   CI runs that version.
3. Does UMT5's use of the old proxy (`modeling_umt5.py:1052`) also need reporting upstream?
   Related but separate — see `plans/umt5/07-VERIFIED-FINDINGS.md` F-04.

---

## 5. Evidence block for the issue

```
CTranslate2 4.8.2 (master), transformers 5.17.0, torch 2.14.0, CPU float32

$ ct2-transformers-converter --model google/mt5-small --output_dir /tmp/ct2-mt5
Forced target "La maison est merveilleuse.", teacher-forced per-token log-probs:

  HF  : [-23.28 -38.27 -65.91 -20.92 -34.24 -72.02 -81.49 -23.53]
  CT2 : [-12.03 -11.93 -12.86 -12.01 -12.21 -13.28 -13.70 -11.15]
  max |diff| = 67.79      (log(1/250112) = -12.43 => CT2 is near-uniform)

Same script, t5-small: max |diff| = 1.07e-06

Cause: transformers 5 moved the decoder-output rescale to a `scale_decoder_outputs`
config flag. modeling_t5.py:1044 uses it; modeling_mt5.py dropped the rescale entirely;
converters/transformers.py:1252 still keys off tie_word_embeddings, so CTranslate2
multiplies mT5's decoder output by d_model**-0.5 when HF does not.
```

Harness: `plans/umt5/scripts/parity.py` (uses a fixed natural-language forced target —
pretrained-only T5 models emit `<extra_id_*>`, whose spellings differ between tokenizers and
would confound the measurement).
