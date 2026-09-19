# UMT5 support in CTranslate2 — master plan

> Base commit: `d44d2d06` (v4.8.2), branch `master`.
> Audience: an implementing agent (Sonnet) + the repo owner.
> Read this file first, then execute phases in order. Each phase has its own file.

---

## 1. The question: is it even possible?

**Yes.** It is possible, and the work is bounded. But it is **not** a one-line converter
registration. Anyone who tells you "just add `@register_loader("UMT5Config")` on the T5
loader" is wrong, and the failure mode of that shortcut is the worst kind: **the conversion
succeeds, the model loads, it generates fluent-looking text, and it is numerically wrong.**

The work splits into two independent pieces:

| Piece | Language | Size | Risk |
|---|---|---|---|
| A. Converter: recognise `UMT5Config`, map HF weights → CT2 spec | Python | ~80 lines | Low |
| B. Runtime: stop sharing the relative-attention position bias across layers | C++ | ~25 lines | Medium |
| C. Config dispatch: stock `google/umt5-*` repos have no `model_type` key | Python | ~15 lines | Low |

All three are required. B is the part that makes it correct; C is the part that makes it
work on the checkpoints Google actually published. C was **not** in the first draft of this
plan — it was found by measurement (`07-VERIFIED-FINDINGS.md` F-03).

---

## 2. What the user actually hit

`python/ctranslate2/converters/transformers.py:106-120`:

```python
config_name = config.__class__.__name__          # -> "UMT5Config"
loader = _MODEL_LOADERS.get(config_name)         # -> None
if loader is None:
    raise ValueError(
        "No conversion is registered for the model configuration %s "
        "(supported configurations are: %s)" % (config_name, ...))
```

The registry (`_MODEL_LOADERS`) is keyed on the **exact HF config class name**. It contains
`T5Config` (line 1231) and `MT5Config` (line 1356). It does **not** contain `UMT5Config`.
That is the literal error. It is a missing-entry error, not a "this is impossible" error.

---

## 3. Why T5's loader cannot simply be reused — the real blocker

### 3.1 The architectural difference

T5 / mT5 and UMT5 are the same shape (encoder-decoder, pre-norm, RMSNorm, gated-GELU FFN,
relative attention bias, no absolute position encodings). They differ in **exactly one
thing that matters to CTranslate2**:

- **T5 / mT5**: only the *first* self-attention layer owns a
  `relative_attention_bias` embedding table. Every other layer reuses layer 0's bias.
- **UMT5**: **every** self-attention layer owns its **own** `relative_attention_bias`
  table. (This is the whole point of the UMT5/UniMax checkpoint family.)

> ✅ **VERIFY-1 — DONE, CONFIRMED.** See `07-VERIFIED-FINDINGS.md` F-01. Measured against
> `transformers` 5.17.0 source *and* the real `google/umt5-small` weights: every
> `UMT5LayerSelfAttention` builds `UMT5Attention(..., has_relative_attention_bias=True)`,
> cross-attention gets `False`, and the checkpoint carries 8 encoder + 8 decoder bias tables,
> all pairwise distinct. Negative control: `t5-small` and `mt5-small` have exactly one each.
> **The premise holds. Phase 2 is required, not optional.**

### 3.2 How the current converter would silently corrupt UMT5

`transformers.py:1281-1305`, `T5Loader.set_stack`:

```python
for i, (layer_spec, block) in enumerate(zip(spec.layer, module.block)):
    self.set_self_attention(layer_spec.self_attention, block.layer[0])   # (a) copies THIS layer's bias

    if i > 0:
        # Reuse relative attention bias from the first layer.
        first_self_attention = spec.layer[0].self_attention
        layer_spec.self_attention.relative_attention_bias = (               # (b) then OVERWRITES it
            first_self_attention.relative_attention_bias)
        layer_spec.self_attention.relative_attention_max_distance = (
            first_self_attention.relative_attention_max_distance)
```

For T5 that `if i > 0` block is a no-op fixup (layers 1..N have no bias of their own).
For UMT5, step (a) reads the correct per-layer table and step (b) **throws it away** and
substitutes layer 0's. Result: layers 1..N-1 get the wrong positional prior. No exception,
no warning. This is the #1 silent-corruption trap of this project.

### 3.3 How the C++ runtime would silently corrupt UMT5 even with a perfect converter

`src/layers/transformer.cc:460-466` (encoder):

```cpp
StorageView position_bias(output.dtype(), output.device());

for (size_t l = 0; l < _layers.size(); ++l) {
  (*_layers[l])(input, lengths_mask.get(), output, padder.get(), &position_bias);
  ...
}
```

and `src/layers/transformer.cc:731` + `:807` (decoder) do the same: **one** `position_bias`
buffer is created per forward pass and threaded through every layer.

`src/layers/attention.cc:231-246`:

```cpp
if (relative_attention_bias) {
  StorageView local_position_bias(output.dtype(), output.device());
  if (!position_bias)
    position_bias = &local_position_bias;          // <-- the escape hatch we will use

  if (position_bias->empty()) {                    // <-- only layer 0 ever fills it
    *position_bias = compute_relative_bias(*relative_attention_bias, ...);
  }
  ... add position_bias to the attention scores ...
}
```

Layer 0 fills the buffer from *its* table. Layers 1..N find it non-empty and reuse it.
So even if the converter writes 12 distinct bias tables into the model file, the runtime
reads only the first one.

**This is the actual blocker, and it is why converter-only patches cannot work.**

### 3.4 The good news

Look again at `attention.cc:233-235`. If the caller passes `position_bias == nullptr`,
each attention layer falls back to a **local** buffer and computes the bias from **its own**
table. The per-layer path already exists and is already correct — it is simply never taken,
because the encoder/decoder always pass a non-null shared buffer.

So Phase 2 is: **decide per model whether to pass `&position_bias` or `nullptr`.**
That is ~25 lines of C++, no new ops, no new kernels, no serialization format change.

Also confirmed working in our favour:
- `MultiHeadAttention` already reads its bias per-layer:
  `attention.cc:305` `model.get_variable_if_exists(scope + "/relative_attention_bias")`,
  scope = `encoder/layer_<i>/self_attention`. Per-layer tables are *already* loadable.
- `MultiHeadAttentionSpec` already declares `relative_attention_bias` per layer
  (`python/ctranslate2/specs/attention_spec.py:60-62`). No spec-shape change needed.
- Per-layer `relative_attention_max_distance` is already an attribute
  (`attention.cc:330-332`). No change needed.

---

## 4. How we detect "share vs per-layer" at runtime (key design decision)

`python/ctranslate2/specs/model_spec.py:169-189` `_alias_variables()` finds byte-identical
tensors at serialization time and stores the duplicates as **aliases**. `src/models/model.cc:279-283`
resolves an alias by inserting **the same `shared_ptr<StorageView>`** under the second name,
and `Model::copy_to` (`model.cc:812-829`) deliberately preserves that sharing.

Consequence: for a **T5** model, `get_variable_if_exists("encoder/layer_0/.../relative_attention_bias")`
and `...layer_5/...` return the **same raw pointer**. For a **UMT5** model they return
**different pointers**.

→ We can detect the mode with a pointer comparison at construction time. No new model
attribute, no `binary_version` bump, no `spec_revision` bump, and **every existing T5/mT5/
FLAN-T5 model keeps byte-identical behaviour**.

**Recommended: Option R (Runtime pointer detection).**
A fallback **Option F (explicit spec flag)** is specified in `03-PHASE-2-runtime.md §2.6`
in case review rejects R. Do **not** implement both. Decision is logged in `06-DECISIONS.md`.

Trade-off table:

| | Option R — pointer detection | Option F — explicit flag |
|---|---|---|
| C++ diff | ~25 lines | ~15 lines |
| Python diff | 0 lines | ~10 lines (spec + converter) |
| Old runtime + new UMT5 model | silently shares (wrong) | silently shares (wrong) — same |
| Old runtime + new T5 model | unaffected | unaffected |
| Needs `spec_revision` bump | no | no (attribute default = false) |
| Fails if two UMT5 layers happen to have byte-identical bias tables | degrades to sharing for those layers — **harmless**, they are identical | n/a |
| Depends on alias machinery staying as-is | yes | no |

The "byte-identical tables" edge case is worth stating precisely: if two UMT5 layers had
identical tables, aliasing would collapse them, pointer comparison would still find *other*
layers differing, the model would take the per-layer path, and the two identical layers
would each compute the same (correct) bias. **There is no wrong-output path.** The only
pathological case is *all* layers identical, which is numerically indistinguishable from T5.

---

## 5. Branch strategy

One branch per phase, each independently reviewable, each merged into `umt5/integration`
only after its own gate passes.

```
master (d44d2d06)
 └─ umt5/phase0-baseline        (no src/ change; harness + captured HF reference tensors)
     └─ umt5/phase1-converter   (python/ only; UMT5Loader)
         └─ umt5/phase2-runtime (src/ + include/ only; per-layer position bias)
             └─ umt5/phase3-validation (tests + docs + CHANGELOG)
                 └─ umt5/integration  -> PR to master
```

Rules for the implementing agent:
- **Never** mix a Python change and a C++ change in one commit.
- Each phase file ends with a **GATE** section. Do not open the next phase file until the
  gate passes and you have pasted the gate evidence into `06-DECISIONS.md`.
- If a gate fails, stop and report. Do not improvise a workaround that changes the plan's
  shape; log it in `05-ERROR-CATALOG.md` under the matching error ID and ask.

---

## 6. Scope — what we WILL do

- ✅ `UMT5Loader` registered for `UMT5Config`, covering `UMT5ForConditionalGeneration`.
- ✅ Correct per-layer relative-attention-bias conversion for encoder **and** decoder.
- ✅ C++ runtime support for per-layer position bias, opt-in by detection, zero behaviour
  change for existing models.
- ✅ Vocabulary / special-token handling for the UMT5 sentencepiece tokenizer.
- ✅ A numerical parity harness (`plans/umt5/scripts/`) comparing CT2 vs HF logits, with an
  explicit tolerance, run on **CPU float32** first.
- ✅ Unit test in `python/tests/test_transformers.py` following the existing pattern.
- ✅ Docs: `docs/guides/transformers.md` supported-model list + a `## UMT5` section.
- ✅ CHANGELOG entry.

## 7. Scope — what we will NOT do

- ❌ Not touching T5Gemma / T5Gemma2 loaders. They are separate architectures with their own
  norm placement; they are out of scope and must not regress.
- ❌ No `binary_version` bump (`model_spec.py:25` = 6, `model.h:20` = 6). Not needed.
- ❌ No `TransformerSpec.revision` bump (`transformer_spec.py`, currently 7). Bumping it
  would make **every newly converted model**, including plain T5, unloadable on older
  CTranslate2 builds. Not worth it. See `06-DECISIONS.md` D-03.
- ❌ No flash-attention support for relative attention bias. Flash attention has no
  position-bias path; we will assert it stays disabled for these models
  (see `05-ERROR-CATALOG.md` U-07), not implement it.
- ❌ No new fused kernels, no perf optimisation of `compute_relative_bias`. Per-layer bias
  computation is O(num_layers × L_q × L_k × heads) instead of O(1 × ...). For encoder
  lengths ≤ 512 this is small relative to the FFN. If benchmarks say otherwise, that is a
  **follow-up**, not this project. See `06-DECISIONS.md` D-05.
- ❌ No UMT5 `UMT5EncoderModel`-only (encoder-only) support in phase 1-3. It is a natural
  follow-up; explicitly deferred so the parity harness has one target.
- ❌ No quantized (int8) parity guarantee in the gates. int8 is smoke-tested only; parity is
  asserted on float32.
- ❌ No GPU/CUDA-specific work. Development and gates are CPU. A CUDA smoke test is listed
  as optional in Phase 3 for whoever has the hardware.

---

## 7b. Findings that post-date this document

`07-VERIFIED-FINDINGS.md` records what was actually measured on 2026-09-20 and **supersedes
anything here that contradicts it**. The headlines:

- **F-01** the per-layer-bias premise is proven on real weights — Phase 2 stands.
- **F-03 (new blocker)** `AutoConfig.from_pretrained` raises `Unrecognized model` on stock
  `google/umt5-*` because their `config.json` has no `model_type` key. This fires at
  `transformers.py:107`, *before* the loader registry is consulted, so Phase 1's registry
  entry alone does not fix stock Google repos.
- **F-04 (new, silent)** transformers 5.x hard-forces `UMT5Config.tie_word_embeddings = True`,
  overriding the checkpoint's `false`. CT2 and HF read the same wrong flag so parity stays
  green, but conversion output becomes transformers-version-dependent.
- **F-05** the vocabulary-padding bug (K-12) is real: 84 duplicate `<extra_id_*>` tokens.
  The fix is mandatory, not conditional.
- **K-10 and K-11 cannot fire** — both conditional branches deleted from Phase 1.

## 8. Reading order

1. `00-OVERVIEW.md` ← you are here
2. `01-PHASE-0-baseline.md` — environment, reproduce the error, capture HF ground truth
3. `02-PHASE-1-converter.md` — the Python loader
4. `03-PHASE-2-runtime.md` — the C++ change
5. `04-PHASE-3-validation.md` — parity, tests, docs
6. `05-ERROR-CATALOG.md` — every error we expect, plus probes for the ones we don't
7. `06-DECISIONS.md` — decision log + evidence log, filled in as you go
8. `07-VERIFIED-FINDINGS.md` — **what was actually measured.** Wins any conflict with 00-05.
