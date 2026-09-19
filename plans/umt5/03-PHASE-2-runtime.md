# Phase 2 — The runtime: per-layer relative position bias

**Branch:** `umt5/phase2-runtime` (from `umt5/phase1-converter`)
**Touches:** `src/layers/transformer.cc`, `include/ctranslate2/layers/transformer.h`.
**Must NOT touch:** `src/layers/attention.cc` (the per-layer path already exists there),
any Python file, any spec, any serialization constant.
**Goal:** a UMT5 model uses each layer's own bias; every existing model is bit-identical.

This is the phase that requires actual care. Read all of §2.1 before editing.

---

## 2.1 Exactly what is wrong today

### Encoder — `src/layers/transformer.cc:460-466`

```cpp
StorageView position_bias(output.dtype(), output.device());

for (size_t l = 0; l < _layers.size(); ++l) {
  (*_layers[l])(input, lengths_mask.get(), output, padder.get(), &position_bias);
  if (l + 1 < _layers.size())
    input = std::move(output);
}
```

### Decoder — `src/layers/transformer.cc:731` and the call at `:807`

```cpp
StorageView position_bias(dtype, device);       // line 731, once per forward pass
...
      (*_layers[l])(*layer_in_chunk, ..., &position_bias, offset);   // line 807
```

### The consumer — `src/layers/attention.cc:231-246`

```cpp
if (relative_attention_bias) {
  StorageView local_position_bias(output.dtype(), output.device());

  if (!position_bias)
    position_bias = &local_position_bias;      // (A) per-layer path — already correct

  if (position_bias->empty()) {                // (B) only true for the first layer
    const dim_t query_length = queries.dim(2);
    const dim_t key_length   = keys.dim(2);
    *position_bias = compute_relative_bias(*relative_attention_bias,
                                           query_length, key_length,
                                           maximum_relative_position,
                                           is_decoder,
                                           with_cache ? key_length - 1 : 0);
  }
  ...
}
```

Path (A) already does the right thing. It is unreachable because the encoder/decoder always
pass a non-null pointer. **The fix is to pass `nullptr` when the layers have distinct bias
tables.** No new computation is introduced; an existing branch is enabled.

---

## 2.2 How we decide (Option R, per `00-OVERVIEW.md` §4)

Evidence chain, all verified in the tree:

- `python/ctranslate2/specs/model_spec.py:169-189` — `_alias_variables()` replaces a duplicate
  tensor's value with the *name* of the earlier one. `SKIP_CREATING_ALIAS`
  (`model_spec.py:38`) contains only the two rotary factors, so `relative_attention_bias`
  **is** eligible for aliasing.
- `src/models/model.cc:278-283` — `register_variable_alias` inserts the alias name pointing at
  **the same `std::shared_ptr<StorageView>`**.
- `src/models/model.cc:812-829` — `Model::copy_to` explicitly preserves pointer sharing
  (`seen_variables` map keyed on `const StorageView*`), so the property survives a device copy.
- `src/layers/attention.cc:305` — each `MultiHeadAttention` resolves its bias via
  `model.get_variable_if_exists(scope + "/relative_attention_bias")`, scope
  `encoder/layer_<i>/self_attention` (built in `transformer.cc:59-67` from
  `build_layers_list`'s `prefix + "_" + i`, `include/ctranslate2/layers/common.h:26-44`).

⇒ For T5 all layers return the **same** pointer. For UMT5 they return **different** pointers.

Phase 1's **G1.4 / evidence E-08** is the empirical confirmation of this for both families.
**If E-08 did not confirm it, stop and switch to Option F (§2.6).**

---

## 2.3 The change — step by step

### Step 2.3.1 — add the helper (anonymous-namespace-local, in `src/layers/transformer.cc`)

Place it near the other file-local helpers (e.g. just above `TransformerEncoder::TransformerEncoder`
at line 404):

```cpp
    // T5-style models compute the relative attention bias once in the first layer and
    // share it with all following layers. UMT5-style models give every layer its own
    // bias table. When the converter emits identical tables they are serialized as
    // aliases pointing to the same StorageView, so identity of the resolved pointers
    // tells the two cases apart without any change to the model format.
    static bool has_shared_position_bias(const models::Model& model,
                                         const std::string& scope,
                                         const size_t num_layers) {
      const auto bias_name = [&scope](size_t i) {
        return scope + "/layer_" + std::to_string(i) + "/self_attention/relative_attention_bias";
      };

      const StorageView* first = model.get_variable_if_exists(bias_name(0));
      if (!first)
        return true;  // no relative attention bias at all: the shared buffer stays unused

      for (size_t i = 1; i < num_layers; ++i) {
        if (model.get_variable_if_exists(bias_name(i)) != first)
          return false;
      }
      return true;
    }
```

Behaviour notes the implementer must understand:
- `num_layers < 2` → loop body never runs → returns `true` → shared buffer → unchanged
  behaviour for 1-layer models. Correct (with one layer there is nothing to share *with*).
- No bias at all (every non-T5 model: BART, NLLB, Whisper, wav2vec2, GPT-*, …) → early
  `return true` → **identical code path to today.** This is the property that makes the
  change safe.
- A layer missing its bias while layer 0 has one → pointer differs (it is `nullptr`) →
  returns `false` → per-layer path. That layer then adds no bias, which is what its weights
  say. No crash, no silent substitution.

### Step 2.3.2 — `TransformerEncoder`

`include/ctranslate2/layers/transformer.h`, in the `TransformerEncoder` private section
(after `_layers`, around line 164) add:

```cpp
      const bool _shared_position_bias;
```

**Placement matters.** C++ initialises members in *declaration* order and the helper needs
`_layers` to be built (for `_layers.size()`). Declare `_shared_position_bias` **after**
`_layers`, and initialise it after `_layers` in the ctor init list. Compile with
`-Wreorder` (it is on by default with `-Wall`) and treat any reorder warning as a failure
→ **U-02**.

`src/layers/transformer.cc`, in the ctor (currently ends line 423), append to the init list
**after** the `_layers(...)` entry and **before** `_position_encoder(...)` is fine as long as
declaration order in the header matches. Safest: declare and initialise it **last**, after
`_tensor_parallel`:

```cpp
      , _tensor_parallel(model.tensor_parallel())
      , _shared_position_bias(has_shared_position_bias(model, scope, _layers.size()))
    {
    }
```

Then change the forward pass at `transformer.cc:460-466`:

```cpp
      StorageView position_bias(output.dtype(), output.device());
      StorageView* position_bias_ptr = _shared_position_bias ? &position_bias : nullptr;

      for (size_t l = 0; l < _layers.size(); ++l) {
        (*_layers[l])(input, lengths_mask.get(), output, padder.get(), position_bias_ptr);
        if (l + 1 < _layers.size())
          input = std::move(output);
      }
```

### Step 2.3.3 — `TransformerDecoder`

Same shape. Header: add `const bool _shared_position_bias;` to `TransformerDecoder`'s private
section, declared **after** `_layers`. Ctor (`transformer.cc:487-...`): initialise with
`has_shared_position_bias(model, scope, _layers.size())`, respecting declaration order.

> The `TransformerDecoder` ctor has a body (it sets alignment heads, `transformer.cc:518+`).
> You may alternatively assign in the body if the init-list ordering gets awkward — then the
> member must not be `const`. Prefer the `const` + correct-order version; only drop `const`
> if ordering genuinely cannot be satisfied, and say so in `06-DECISIONS.md`.

Then at `transformer.cc:731`:

```cpp
      StorageView position_bias(dtype, device);
      StorageView* position_bias_ptr = _shared_position_bias ? &position_bias : nullptr;
```

and at the call site, `transformer.cc:807`, replace `&position_bias` with `position_bias_ptr`.

**Grep afterwards** to be sure you got every call site:

```bash
grep -n "position_bias" src/layers/transformer.cc
```
Expected remaining `&position_bias` occurrences: **zero** in `TransformerEncoder::operator()`
and `TransformerDecoder::operator()`. The parameter forwarding at lines 108, 145, 247, 321,
345 passes an already-`StorageView*` argument through — **leave those alone.**

---

## 2.4 Things that must be re-checked in this phase, not assumed

### 2.4.a The decoder's chunked (sliding-window) loop

`transformer.cc:733-746` splits `layer_in` into `layer_ins` chunks when `_sliding_window` is
set, and the layer loop at `:757` runs **inside** a chunk loop. The single `position_bias`
declared at `:731` is therefore reused across chunks of **different lengths**. That is
already latently wrong for any model combining sliding-window with relative attention bias —
but no such model exists today (`_sliding_window` comes from `decoder/sliding_window`,
used by T5Gemma2/Mistral-style models, none of which use relative attention bias).

**Action:** do **not** fix this. Do add a one-line comment noting it, and record it in
`06-DECISIONS.md` as **D-06 (known, out of scope, pre-existing)**. On the per-layer path
(`position_bias_ptr == nullptr`) the problem disappears by construction.

### 2.4.b Incremental decoding offset

`attention.cc:244` passes `with_cache ? key_length - 1 : 0` as `query_offset`. The decoder
creates `position_bias` fresh on **every** call (line 731 is inside `operator()`), so it is
per-step, not cached across steps. Per-layer buffers inherit the same lifetime. No change.
**Verify empirically** in Phase 3 by checking step-2+ logits, not just step 1 — see
**U-06**.

### 2.4.c Tensor parallel

`attention.cc:247-255` slices `position_bias` per MPI rank into a *temporary*
(`position_bias_tmp`) and never writes back into `*position_bias`. So slicing is per-layer
already and works identically on the local path. No change. Untestable on this machine
(`WITH_TENSOR_PARALLEL=OFF`) — record as **untested** in `06-DECISIONS.md`, do not claim it works.

### 2.4.d Flash attention

`TransformerEncoderLayer` picks `FlashMultiHeadAttention` when `use_flash_attention`
(`transformer.cc:59-67`). `FlashMultiHeadAttention` takes a `position_bias` parameter
(`include/ctranslate2/layers/flash_attention.h:31`) — confirm whether it *uses* it.
`model.cc:198` already forces flash attention off unless the compute type is float16/bfloat16,
and `WITH_FLASH_ATTN` defaults OFF (`CMakeLists.txt:25`), so our builds never hit it.

**Action:** read `src/layers/flash_attention.cc` and answer: does it silently ignore
relative attention bias? If yes, that is a **pre-existing T5-with-flash-attention bug**, not
ours. Record as evidence **E-11** and as **D-07 (out of scope)**. Do not fix it here. Do not
claim UMT5 works with flash attention.

### 2.4.e `num_layers` vs the helper's argument

`_layers.size()` is the authoritative count (built by probing `layer_0`, `layer_1`, …
until one is missing — `common.h:31-42`). Use it, not a config attribute.

---

## 2.5 Rebuild and re-verify the toolchain

```bash
cmake --build build-umt5 -j"$(sysctl -n hw.ncpu)"
python -c "import ctranslate2, inspect; print(ctranslate2.__file__)"
```

> Re-run Phase 0 checkpoint **0.3c**. If `ctranslate2.__file__` points at site-packages
> rather than this tree, your C++ change is not loaded and every result below is fiction.
> This is **U-05** and it is the most common way this phase wastes a day.

Positive control that the rebuild is live: temporarily add
`spdlog::warn("shared_position_bias={}", _shared_position_bias);` to the encoder ctor, load a
model, confirm you see `false` for UMT5 and `true` for t5-small, then remove it. **Record the
two log lines as evidence E-12** — this is the direct proof that detection works, and it is
worth more than any downstream output comparison.

---

## 2.6 Option F — fallback if Option R is rejected in review

Only if `00-OVERVIEW.md` §4's pointer-identity argument is rejected (e.g. a reviewer objects
to depending on the alias machinery). **Do not implement alongside Option R.**

1. `python/ctranslate2/specs/transformer_spec.py`: add
   `per_layer_position_bias: bool = False` to `TransformerEncoderSpec.__init__`,
   `TransformerDecoderSpec.__init__` and `TransformerSpec.from_config`; when true set
   `self.per_layer_position_bias = True` on the stack spec (a bare `True` serializes as an
   int8 attribute, same as `pre_norm`).
2. `python/ctranslate2/converters/transformers.py`: `UMT5Loader.get_model_spec` passes
   `per_layer_position_bias=True`.
3. `src/layers/transformer.cc`: replace the helper call with
   `!model.get_flag_with_default(scope + "/per_layer_position_bias", false)`.

Cost: touches the spec (so Phase 1 and Phase 2 stop being cleanly separable), and old models
get the default `false` → shared → unchanged. It does **not** require a `spec_revision` bump
because `get_flag_with_default` tolerates the attribute's absence.

Risk unique to F: an old CTranslate2 binary loading a new UMT5 model ignores the flag and is
silently wrong — but Option R has exactly the same exposure, so this is not a differentiator.

---

## 2.7 GATE 2 — all must hold before opening Phase 3

- [ ] G2.1 Clean build, **zero new warnings**. Diff `cmake --build` output against the Phase 0
      build log. Any `-Wreorder` warning = fail (**U-02**).
- [ ] G2.2 `./build-umt5/tests/ctranslate2_test ./tests/data` — result **identical** to E-01.
      Not "similar". Identical.
- [ ] G2.3 Evidence **E-12**: detection logs `false` for UMT5, `true` for t5-small,
      `true` for mt5-small, and `true` (early-return branch) for a non-T5 model such as
      `facebook/bart-base` or `Helsinki-NLP/opus-mt-en-de`.
- [ ] G2.4 t5-small and mt5-small translation output is **character-identical** to the same
      run on `master`. Capture both, `diff` them. → Evidence **E-13**.
- [ ] G2.5 UMT5 output **differs** from the Phase-1 "pre-fix output" (E-10). If it is the
      same, the change is not live → **U-05**.
- [ ] G2.6 `git diff master --stat` shows changes confined to `src/layers/transformer.cc`
      and `include/ctranslate2/layers/transformer.h` (Option R), plus the Phase-1 Python file
      inherited from the parent branch.
- [ ] G2.7 `D-06` and `D-07` recorded in `06-DECISIONS.md` as knowingly-out-of-scope.

---

## 2.8 Commit

```
Support per-layer relative attention bias in Transformer stacks

T5 shares one relative attention bias table across all layers, so the
encoder and decoder compute the position bias once and thread a single
buffer through every layer. UMT5 gives each layer its own table, which
that buffer silently discards.

Detect the two cases by comparing the resolved variable pointers: the
converter serializes identical tables as aliases, which the model loader
resolves to a single shared StorageView. When the tables differ, pass a
null buffer so each attention layer computes the bias from its own table
via the existing local path in MultiHeadAttention.

Models without a relative attention bias, and models with fewer than two
layers, take the previous code path unchanged.
```
