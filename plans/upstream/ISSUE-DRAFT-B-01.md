# Issue draft — ready to paste

**Repo:** https://github.com/OpenNMT/CTranslate2/issues/new
**Title:** `mT5 decoder output is incorrectly rescaled with transformers 5.x`
**Labels to request:** `bug`

> Paste everything below the line. Re-run the repro on your machine first and use *your*
> numbers — do not paste ours unverified.

---

## Description

With `transformers` 5.x, CTranslate2 applies the T5 decoder-output rescaling
(`d_model ** -0.5`) to mT5 models, but Hugging Face does not. The converted model produces a
near-uniform output distribution.

This affects models converted today with the `transformers` version pinned in
`python/tests/requirements.txt` (`transformers==5.9.0.*`).

## To reproduce

```bash
ct2-transformers-converter --model google/mt5-small --output_dir /tmp/ct2-mt5
```

Score a fixed target against Hugging Face (teacher-forced per-token log-probs, CPU float32,
source `"Translate English to French: The house is wonderful."`, target
`"La maison est merveilleuse."`):

```
HF  log-probs: [-23.28 -38.27 -65.91 -20.92 -34.24 -72.02 -81.49 -23.53]
CT2 log-probs: [-12.03 -11.93 -12.86 -12.01 -12.21 -13.28 -13.70 -11.15]
max |diff| = 67.79    mean |diff| = 32.56
```

`log(1 / 250112) = -12.43`. CTranslate2 sits at that value for every token — a near-uniform
distribution. Hugging Face's is sharply peaked.

**Control, same script and machine, `t5-small`: `max |diff| = 1.07e-06`.**

## Cause

`transformers` 5.x moved the decoder-output rescaling to a dedicated `scale_decoder_outputs`
config flag, and mT5 no longer rescales at all:

| modeling file | guard for `sequence_output * (self.model_dim ** -0.5)` |
|---|---|
| `t5/modeling_t5.py` | `if self.config.scale_decoder_outputs:` |
| `mt5/modeling_mt5.py` | **no rescale anywhere** — `MT5ForConditionalGeneration` defines its own `forward` |
| `umt5/modeling_umt5.py` | `if self.config.tie_word_embeddings:` |

Verified identical on **transformers v5.9.0 and v5.17.0** (0 rescale sites in
`modeling_mt5.py` in both; `scale_decoder_outputs` is the guard in `modeling_t5.py` in both).

Config values:

| model | `tie_word_embeddings` | `scale_decoder_outputs` | HF rescales? | CT2 rescales? |
|---|---|---|---|---|
| `t5-small` | True | `True` | yes | yes ✅ |
| `google/mt5-small` | True | **absent** | **no** | **yes** ❌ |

`T5Loader.get_model_spec` still keys off the pre-5.x proxy —
`python/ctranslate2/converters/transformers.py:1252`:

```python
if model.config.tie_word_embeddings:
    spec.decoder.scale_outputs = model.config.d_model**-0.5
```

`MT5Loader` inherits it, so CTranslate2 multiplies mT5's decoder output by
`1/sqrt(512) = 0.0442` while Hugging Face multiplies by nothing. The runtime side
(`src/layers/transformer.cc:880-881`) is correct — it applies the value the converter gave it.

## Why this wasn't caught

Scaling all logits by a positive constant does not change `argmax`, so greedy decoding still
emits plausible text. Only scores, sampling, and length-penalised beam ranking are affected.
The existing mT5 test (`python/tests/test_transformers.py:106`) asserts output tokens, not
scores, so it passes.

## Questions before I open a PR

1. Is mT5's missing rescale intended in `transformers`, or a regression there? If the latter,
   the fix may belong upstream rather than here.
2. Preferred shape — override `get_model_spec` in `MT5Loader` to never set `scale_outputs`,
   or read `scale_decoder_outputs` with a `tie_word_embeddings` fallback for
   `transformers` < 5? (A naive fallback keeps the bug for mT5, since mT5's config has no
   `scale_decoder_outputs` — so it has to key off the model family, not just the flag.)
3. Which `transformers` versions must the converter still support?

Happy to open a PR once you confirm the direction.

## Environment

```
CTranslate2 4.8.2 (master)
transformers 5.17.0 (also verified against v5.9.0 source, the version CI pins)
torch 2.14.0, Python 3.12, macOS arm64, CPU float32
```

---

## AI-use disclosure — REQUIRED, read before posting

`CONTRIBUTING.md` (lines 31-37) requires it:

> *"Contributors must explicitly disclose how AI was used and remain fully responsible for
> correctness, performance, and design. Submissions that appear generated without deep
> understanding will be declined."*

Add a line in your own words, honestly. For example:

> *Disclosure: I investigated this with AI assistance (Claude). I reproduced every number
> above on my own machine and verified the `transformers` source claims myself before
> filing.*

**Only write that if it's true — so run the repro yourself first.**

Be ready to answer, in your own words: why scaling logits doesn't change `argmax`, why
`t5-small` passes while mT5 fails, and where the value is consumed at runtime.
