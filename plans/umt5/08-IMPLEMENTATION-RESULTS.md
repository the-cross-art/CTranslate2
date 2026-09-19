# Implementation results (2026-09-20)

Built, implemented, and measured. Branches: `umt5/phase0-baseline` →
`umt5/phase1-converter` (479667f3) → `umt5/phase2-runtime` (709c2711).

**Environment:** macOS arm64, Apple clang 21, cmake 4.4.3 (venv), Accelerate + Ruy,
`OPENMP_RUNTIME=NONE`, transformers 5.17.0, torch 2.14.0, CTranslate2 built from this tree
(`import ctranslate2` resolves to `python/ctranslate2/__init__.py` — K-07 avoided).

---

## 1. What works

| Check | Result |
|---|---|
| `ct2-transformers-converter` on UMT5 | **exit 0** |
| Bias tables on disk | **8 encoder + 8 decoder**, 0 cross-attention |
| Bias values vs HF checkpoint | **exact** (`np.array_equal`), all 16 pairwise distinct |
| Aliasing: UMT5 | **0** biases aliased away |
| Aliasing: t5-small | **10** aliases (layers 1-5 → layer 0), 2 real tables |
| Vocabulary | 256384 entries, **0 duplicates**, 300 `<extra_id_*>`, 84 `<unused_*>` |
| Runtime detection (E-12) | umt5 `shared_position_bias=0` (both stacks); t5-small `=1` (both) |
| C++ test suite | **197 ran, 196 passed, 1 skipped** — identical to baseline |
| t5-small `model.bin` | **byte-identical** sha256 before/after the converter change |
| t5-small parity | **max \|diff\| 1.07e-06** — PASS |
| Build warnings | 7, all pre-existing (nlohmann json); **0 `-Wreorder`** |

### The C++ change is provably load-bearing (falsification test, X-01/E-14)

Same model, same harness, only the runtime differs:

| Branch | Greedy 20 tokens vs HF | log-prob max diff |
|---|---|---|
| `umt5/phase1-converter` (converter only) | **diverges at step 1** | 0.996 |
| `umt5/phase2-runtime` (+ C++ fix) | **exact match, all 20** | 0.605 |

The harness demonstrably fails before the fix and passes P4 after it. Without this pair,
"it works" would be unfalsifiable.

---

## 2. What does NOT work — an unresolved residual

**P2 (per-token forced-target log-probs) still fails for UMT5:** max |diff| **0.605**,
mean 0.206, against a harness noise floor of **1.07e-06** measured on t5-small.

**This is not caused by either change in this project.** Evidence:

- A **1-layer** UMT5 (`shared_position_bias=1`, i.e. the *original* code path, where
  per-layer vs shared is meaningless) still fails at **0.022**.
- Zeroing every bias table still fails at **0.025** ⇒ **not bias-related at all**.
- Error is **flat in sequence length** (0.148 at src_len 2, 0.219 at src_len 17)
  ⇒ structural per-token, not accumulating through attention.

### Hypotheses tested and **ruled out**

| Hypothesis | Test | Result |
|---|---|---|
| GELU formula mismatch | CT2 `gelu_tanh_func` vs HF `NewGELUActivation` | **identical** formula, same constants |
| Gated-GELU activation | rebuilt model as `gated-relu` | error unchanged (0.0222 vs 0.0220) → **not GELU** |
| `head_dim` (UMT5 has 6×64=384 ≠ d_model 512; CT2 defaults `_d_model/_num_heads` = 512/6 = **85**) | injected explicit `head_dim=64` | **no change at all** → CT2 derives the head split from tensor shapes in this path |
| Logit scaling (`scale_outputs`) | fitted best scalar α on HF logits | best α = 0.90 at the grid edge, error 0.241 vs 0.243 → **no scalar explains it** |
| Position-bias sharing | 1-layer model | fails on the unchanged path → **not it** |

Still untested: the untied `lm_head` path, and isolating encoder vs decoder (blocked because
`ctranslate2.Encoder` rejects a seq2seq spec — an encoder-only conversion would be needed).

**Practical impact:** greedy/beam output matches HF exactly, so translation is usable. Scores,
softmax sharpness, sampling and length-penalised beam ranking would be affected. **Do not
claim numerical parity for UMT5 until this is closed.**

---

## 3. Two PRE-EXISTING CTranslate2 bugs found along the way

Neither is caused by this work; both exist on `master`. Both are worth reporting upstream.

### B-01 · mT5 is badly broken with transformers 5.x — 67 nats off

`google/mt5-small`, unmodified shipped path (`shared_position_bias=1`, converter untouched):

```
HF  log-probs: [-23.28 -38.27 -65.91 -20.92 -34.24 -72.02 -81.49 -23.53]
CT2 log-probs: [-12.03 -11.93 -12.86 -12.01 -12.21 -13.28 -13.70 -11.15]
max |diff| = 67.79
```

CT2's output is near-uniform (≈ log 1/250112 = -12.4); HF's is sharply peaked.

**Root cause:** transformers 5.x introduced a dedicated `scale_decoder_outputs` config field.

| model | `tie_word_embeddings` | `scale_decoder_outputs` | HF rescales? | CT2 rescales? |
|---|---|---|---|---|
| t5-small | True | **True** (`modeling_t5.py:1044`) | yes | yes → ✅ match |
| **mt5-small** | True | **absent**; `MT5ForConditionalGeneration` defines its own `forward` with **no rescaling** | **no** | **yes** → ❌ 67 nats |
| umt5 | True (forced) | absent; uses `tie_word_embeddings` (`modeling_umt5.py:1052`) | yes | yes → match |

`transformers.py:1252` keys off `tie_word_embeddings`, the pre-5.x proxy. For mT5 that is now
simply wrong: CT2 multiplies the decoder output by `d_model**-0.5` when HF does not.

### B-02 · T5 vocabularies contain duplicate `<extra_id_*>` with misaligned ids

`t5-small`, shipped path:

```
config.vocab_size 32128 | tokenizer 32100 | gap 28 | <extra_id_*> already present: 100
CT2 vocabulary: 32128 entries, 28 DUPLICATES
<extra_id_0>: HF id 32099  but  CT2 index 32100
<extra_id_5>: HF id 32094  but  CT2 index 32105
```

`T5Loader.get_vocabulary` (`transformers.py:1257-1264`) pads with `<extra_id_i>` names that
already exist. Latent for ordinary translation (which is why it has gone unnoticed), wrong for
any sentinel/span-infilling use — T5's own pretraining objective. This is the same bug class
as K-12, which `UMT5Loader` now fixes for UMT5 only.

---

## 4. Also confirmed

- **N-01** reproduced: stock `google/umt5-*` `config.json` has no `model_type`, so
  `AutoConfig` raises before the registry. Worked around locally by adding the key; the
  fallback (D-12/W3) is **not implemented**.
- **N-02** reproduced: `UMT5Config.__post_init__` forces `tie_word_embeddings=True`. HF warns
  and refuses to tie (weights are safe), but both HF and CT2 then apply `d_model**-0.5`.
  Because they agree, parity cannot catch it. The resulting distribution is near-uniform
  (step-0 logit std **0.219**), which is why `umt5-small` generates degenerate repetition for
  this prompt in **both** HF and CT2.

## 5. Harness note

`plans/umt5/scripts/parity.py` uses a **fixed natural-language forced target**
("La maison est merveilleuse."), not HF's greedy output. Pretrained-only T5-family models emit
`<extra_id_*>` for this prompt, and sentinel spellings differ per tokenizer (`▁<extra_id_0>`
in mT5 vs `<extra_id_0>` in CT2's padding), which confounded the first runs with a vocabulary
artifact. `ctranslate2.Encoder` cannot load a seq2seq spec, so P2 scores through both stacks
via `Translator.score_batch` rather than comparing encoder states directly.

## 6. Not done

- W3 / D-12 (`model_type` fallback) — stock Google repos still need a patched `config.json`.
- Phase 3: unit test, docs, CHANGELOG (deferred with upstreaming; local fork for now).
- The P2 residual root cause.
- int8/float16, CUDA, tensor-parallel, flash attention: untested.
