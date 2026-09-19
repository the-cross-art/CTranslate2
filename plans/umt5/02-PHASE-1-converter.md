# Phase 1 — The converter: `UMT5Loader`

**Branch:** `umt5/phase1-converter` (from `umt5/phase0-baseline`)
**Touches:** `python/ctranslate2/converters/transformers.py` **only.**
**Must NOT touch:** anything under `src/`, `include/`, `python/ctranslate2/specs/`.

> **This file was rewritten after measurement.** `07-VERIFIED-FINDINGS.md` is the source of
> truth. Two conditional branches from the first draft are gone (they cannot fire), one
> conditional became mandatory, and two new required pieces of work appeared.

**Goal:** `ct2-transformers-converter --model google/umt5-small --output_dir ...` succeeds and
produces **8 distinct** encoder bias tables and **8 distinct** decoder bias tables. At the end
of Phase 1 the model will still *run wrong* — that is Phase 2's job. Phase 1's gate is about
**what is on disk**, not output quality.

---

## 1.0 Read these before writing anything

| File:line | Why |
|---|---|
| `transformers.py:106-120` | `_load` — where **both** failure modes live (F-03 at :107, registry at :113) |
| `transformers.py:1231-1353` | `T5Loader` in full — you are subclassing it |
| `transformers.py:1281-1310` | `set_stack` — contains the bias-sharing hack you must override |
| `transformers.py:1257-1264` | `get_vocabulary` — the `<extra_id_*>` padding bug (F-05) |
| `transformers.py:1252-1253` | `tie_word_embeddings` → `scale_outputs` (F-04) |
| `transformers.py:1356-1360` | `MT5Loader` — the 4-line precedent. **UMT5 is NOT this.** |
| `python/ctranslate2/specs/attention_spec.py:60-62` | the two per-layer fields we fill |
| `python/ctranslate2/specs/model_spec.py:169-189` | `_alias_variables` — why identical tables collapse |

---

## 1.1 Three pieces of work

| | What | Why | Status |
|---|---|---|---|
| **W1** | `UMT5Loader` with an overridden `set_stack` | keeps each layer's own bias table | required |
| **W2** | Overridden `get_vocabulary` | the inherited one appends 84 duplicate `<extra_id_*>` tokens | required (F-05) |
| **W3** | `AutoConfig` dispatch fallback | stock `google/umt5-*` `config.json` has no `model_type` | required for stock repos (F-03) |

W1 and W2 go in the new loader. W3 touches shared converter code — keep it in its **own
commit** so it can be reviewed, reverted, or upstreamed independently. It is genuinely useful
on its own (it fixes every legacy HF repo with a missing `model_type`, not just UMT5).

---

## 1.2 W1 — the loader

Insert **immediately after** `MT5Loader` (currently ends at `transformers.py:1360`), so the
three T5-family loaders sit together.

```python
@register_loader("UMT5Config")
class UMT5Loader(T5Loader):
    """Loader for UMT5 (https://arxiv.org/abs/2304.09151).

    UMT5 is structurally identical to mT5 except that *every* self-attention layer owns
    its own relative attention bias table, whereas T5/mT5 compute the bias once in the
    first layer and share it. We therefore reuse T5Loader and only drop the bias-sharing
    step in set_stack.

    Note: models converted by this loader require a CTranslate2 runtime that supports
    per-layer position bias (see src/layers/transformer.cc). Older runtimes will silently
    use the first layer's bias for all layers.
    """

    @property
    def architecture_name(self):
        return "UMT5ForConditionalGeneration"

    def set_stack(self, spec, module, is_decoder=False):
        self.set_layer_norm(spec.layer_norm, module.final_layer_norm)
        self.set_embeddings(
            (
                spec.embeddings[0]
                if isinstance(spec.embeddings, list)
                else spec.embeddings
            ),
            module.embed_tokens,
        )

        spec.scale_embeddings = False

        for layer_spec, block in zip(spec.layer, module.block):
            # Unlike T5, every layer carries its own relative attention bias, so there
            # is no first-layer bias to copy over here.
            self.set_self_attention(layer_spec.self_attention, block.layer[0])

            if layer_spec.self_attention.relative_attention_bias is None:
                raise ValueError(
                    "Expected every UMT5 self-attention layer to define a relative "
                    "attention bias, but one layer did not. The model may not be a "
                    "standard UMT5 checkpoint."
                )

            if is_decoder:
                self.set_cross_attention(layer_spec.attention, block.layer[1])

            self.set_ffn(layer_spec.ffn, block.layer[-1])
```

### Why the explicit `raise`

`set_attention` (`transformers.py:1345`) only writes the bias when
`attention.has_relative_attention_bias` is truthy. That attribute **does** exist today
(F-02), but if a future `transformers` renames or drops it, the field stays `None` and you
would be debugging a runtime shape error instead of a converter error. Fail loudly, here.

### The three-line diff you must NOT write

```python
@register_loader("UMT5Config")
class UMT5Loader(T5Loader):
    @property
    def architecture_name(self):
        return "UMT5ForConditionalGeneration"
```

This mirrors `MT5Loader` and **will appear to work**: it converts without error, loads without
error, and generates plausible text. It is wrong — `T5Loader.set_stack`'s `if i > 0` block
discards 14 of the 16 real bias tables. If you find yourself writing this, re-read
`00-OVERVIEW.md` §3.2 and `07-VERIFIED-FINDINGS.md` F-01.

---

## 1.3 W2 — the vocabulary fix (mandatory; measured, not hypothetical)

**Measured on `google/umt5-small`** (`07-VERIFIED-FINDINGS.md` F-05):

```
config.vocab_size            : 256384
len(tokenizer)               : 256300      (T5Tokenizer; fast and slow agree)
gap                          :     84
<extra_id_*> already in vocab:    300
```

`T5Loader.get_vocabulary` (`transformers.py:1257-1264`) closes an 84-token gap by appending
`<extra_id_0>` … `<extra_id_83>`. But **all 300 sentinels are already in the tokenizer.** The
result is a correctly-sized 256384-entry vocabulary — so **nothing raises** — containing 84
duplicated token strings whose names collide with the real sentinels at their true ids.

Latent for plain translation. A real bug for span-infilling, which is UMT5's own pretraining
objective. Add to `UMT5Loader`:

```python
    def get_vocabulary(self, model, tokenizer):
        # Skip T5Loader.get_vocabulary: UMT5 tokenizers already contain the <extra_id_*>
        # sentinels, and the remaining gap is padding to a multiple of 128.
        tokens = ModelLoader.get_vocabulary(self, model, tokenizer)

        for i in range(model.config.vocab_size - len(tokens)):
            tokens.append("<unused_%d>" % i)

        if len(tokens) != model.config.vocab_size:
            raise ValueError(
                "Vocabulary size mismatch: got %d tokens but the model expects %d"
                % (len(tokens), model.config.vocab_size)
            )

        return tokens
```

Two things to confirm before committing:
1. `ModelLoader.get_vocabulary` is the right super-call. **Chaining to `super()` would call
   `T5Loader.get_vocabulary` and re-introduce the exact bug you are removing.** Read
   `ModelLoader.get_vocabulary` and make sure calling it unbound like this is correct in this
   codebase; if it isn't, restructure rather than chain.
2. `<unused_%d>` does not already appear in the vocabulary. Check before settling on the name.

---

## 1.4 W3 — the `AutoConfig` dispatch fallback (separate commit)

**Measured** (`07-VERIFIED-FINDINGS.md` F-03): `google/umt5-small`'s published `config.json`
has **no `model_type` key**, so `transformers.py:107` raises

```
ValueError: Unrecognized model in google/umt5-small.
Should have a `model_type` key in its config.json.
```

before the registry at line 113 is ever reached. The class defines `model_type = "umt5"`
(`configuration_umt5.py:34`), so the explicit path works:

```
AutoConfig.from_pretrained("google/umt5-small")  -> ValueError
UMT5Config.from_pretrained("google/umt5-small")  -> OK
```

and `config.json` **does** carry `"architectures": ["UMT5ForConditionalGeneration"]`.

**Decision required — record as D-12 in `06-DECISIONS.md` before coding:**

- **F-03a — docs only.** Tell users to add `"model_type": "umt5"`. Zero code; stock repos
  stay broken out of the box.
- **F-03b — architecture fallback.** Catch the `ValueError` in `_load`, read
  `config.json`'s `architectures[0]`, map it to a loader, and proceed. Generic: helps every
  pre-4.31 HF repo with a missing `model_type`.
- **F-03c — hybrid (recommended).** F-03b, but only on failure, and with an error message
  naming both the architecture found and the manual fix if no loader matches.

Sketch for F-03c (adapt to the real code; do not paste blind):

```python
            try:
                config = transformers.AutoConfig.from_pretrained(
                    self._model_name_or_path, trust_remote_code=self._trust_remote_code
                )
            except ValueError:
                # Some older Hub repositories omit "model_type" from config.json, which
                # AutoConfig requires. Fall back to the declared architecture.
                config = self._config_from_architecture()
```

where `_config_from_architecture` reads `config.json` (via `huggingface_hub.hf_hub_download`
for a repo id, or a local path), looks up `architectures[0]` in
`transformers.models.auto.modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES` — or
simply `getattr(transformers, arch).config_class` — and calls
`that_config_class.from_pretrained(...)`. If the architecture is unknown, re-raise the
original `ValueError` **with** the architecture name and the `"model_type"` fix appended.

Constraints:
- Must **not** change behaviour when `AutoConfig` succeeds. Every existing model takes the
  identical path.
- Must not swallow unrelated `ValueError`s — match on the message or re-raise if the fallback
  finds nothing.
- Keep it in its own commit.

---

## 1.5 F-04 — the `tie_word_embeddings` trap: decide, then document

**No code change is proposed here — but the decision must be conscious.**

`configuration_umt5.py:75-76` unconditionally discards the checkpoint's flag:

```python
kwargs.pop("tie_word_embeddings", None)
self.tie_word_embeddings = True   # force it for T5 family
```

`google/umt5-small` is genuinely untied (`lm_head.weight` vs `shared.weight`, max abs diff
116.65), and HF logs a warning and refuses to tie. But both HF (`modeling_umt5.py:1052-1055`)
and CT2 (`transformers.py:1252-1253`) read the lying flag and apply `d_model**-0.5` to the
decoder output. **They agree, so parity will be green and will not flag this.**

**D-11 — recommended: match HF.** Inherit the behaviour, because "match HF" is the converter
contract and diverging would make the parity harness meaningless. But:

- Add a comment in `UMT5Loader` pointing at `configuration_umt5.py:75-76` so the next person
  understands why `scale_outputs` is set on an untied model.
- State in the docs that conversion output is **transformers-version-dependent** for UMT5, and
  pin a version in the conversion instructions.
- Record the transformers version used for the conversion alongside the model.

Do **not** "fix" it by reading `config.json` directly unless you first check what HF 5.9.0
(the version repo CI pins, F-06) does — if 5.9.0 behaves differently, you have a
reproducibility problem to solve before a correctness one.

---

## 1.6 Things that are already correct — leave them alone

All verified in `07-VERIFIED-FINDINGS.md` F-02:

- **Cross-attention has no bias.** `UMT5LayerCrossAttention` builds its attention with
  `has_relative_attention_bias=False`; the guard at `transformers.py:1345` writes nothing.
  Correct. (Note: the `t5-small` *checkpoint* contains a dead
  `decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight` that HF never
  loads — don't be confused by it. `google/umt5-small` has no such stray tensor.)
- **`queries_scale = 1.0`** (`transformers.py:1331`) — confirmed by an explicit upstream
  comment in `UMT5Attention.__init__`: *"UMT5 folds the relative position bias into the
  attention scores and does not scale the query/key dot product."* → `self.scaling = 1.0`.
- **`relative_attention_max_distance`** written per-layer (`transformers.py:1346-1348`), read
  per-layer by C++ (`attention.cc:330-332`). Nothing to do.
- **`spec.scale_embeddings = False`** — kept.
- **Module names** (`.SelfAttention`, `.EncDecAttention`, `.DenseReluDense.wi_0/.wi_1/.wo`,
  `.layer_norm`, `.final_layer_norm`, `.q/.k/.v/.o`, `.block`, `.layer`, `.embed_tokens`,
  `.lm_head`) are identical to T5 → inheriting `set_ffn`, `set_attention`,
  `set_self_attention`, `set_cross_attention`, `set_layer_norm`, `set_config` is safe.

### Deleted from the first draft — these cannot fire

- ~~K-10 `num_decoder_layers is None`~~ — `UMT5Config.__post_init__` defaults it to
  `num_layers`, and `google/umt5-small` sets `8` explicitly. **No `get_model_spec` override
  needed.** Inherit it.
- ~~K-11 unsupported `dense_act_fn`~~ — it is `"gelu_new"`, present in
  `_SUPPORTED_ACTIVATIONS` (`transformers.py:33`) → `Activation.GELUTanh`. `is_gated_act` is
  `True`. Inherit.

---

## 1.7 Inspection script

`plans/umt5/scripts/inspect_ct2_model.py` — loads a converted model and prints, for every
variable matching `*/relative_attention_bias`: name, shape, and a fingerprint
(`float(arr.sum())`, `arr[0,0]`, `arr[-1,-1]`); plus which names became **aliases**.

Easiest route — convert in-process and read the spec before serialization:

```python
from ctranslate2.converters.transformers import TransformersConverter
conv = TransformersConverter("google/umt5-small")
spec = conv._load()
for name, value in spec.variables(ordered=True):
    if name.endswith("relative_attention_bias"):
        ...
```

Then call `_alias_variables()` on a **copy** and re-list to see which collapse. Compare
fingerprints against the `enc_bias_*.npy` / `dec_bias_*.npy` from Phase 0.

> `_load()` is private and its return shape may differ. If it fights you, fall back to
> converting to a temp dir and reading `model.bin` with a struct reader mirroring
> `model_spec.py:390-420`. Budget 30 minutes; G1.3/G1.4 can be met via `spec.variables()` alone.

---

## 1.8 GATE 1 — all must hold before opening Phase 2

- [ ] G1.1 `ct2-transformers-converter --model google/umt5-small --output_dir /tmp/ct2-umt5`
      exits 0. **If W3 was deferred (F-03a), this requires a patched `config.json`** — record
      exactly which route you took.
- [ ] G1.2 **8** encoder bias entries and **8** decoder bias entries; none for cross-attention.
- [ ] G1.3 Every bias fingerprint distinct, each matching the Phase-0 `.npy` **exactly**
      (`np.array_equal`, not `allclose`). → Evidence **E-07**.
- [ ] G1.4 After `_alias_variables()`, **no** `relative_attention_bias` is aliased away; and on
      a converted `t5-small`, layers 1..N-1 **are** aliased to layer 0. → Evidence **E-08**.
      This is the discriminator Phase 2 depends on — see U-09/U-10.
- [ ] G1.5 Vocabulary is exactly 256384 entries, **contains no duplicates**
      (`len(set(tokens)) == len(tokens)`), and matches the embedding rows for both encoder and
      decoder. → Evidence **E-06b**. The duplicate check is the W2 regression test.
- [ ] G1.6 `t5-small`, `google/mt5-small`, `jordimas/t5gemma-s-s-ul2` still convert, producing
      **byte-identical** `model.bin` to a conversion on `master` (`shasum -a 256` before and
      after). → Evidence **E-09**.
- [ ] G1.7 `black --check`, `isort --check`, `flake8` clean.
- [ ] G1.8 UMT5 translation is **expected to be wrong** here. Run it and record the output as
      **E-10** ("pre-fix"). Phase 2 must change it; if it doesn't, see U-05.
- [ ] G1.9 D-11 (tie_word_embeddings) and D-12 (F-03 route) recorded in `06-DECISIONS.md`.

---

## 1.9 Commits

**Commit 1 — the loader:**
```
Add UMT5 converter for Hugging Face Transformers

UMT5 differs from mT5 in that every self-attention layer owns its own
relative attention bias table instead of sharing the first layer's.
T5Loader.set_stack explicitly overwrites layers 1..N with layer 0's bias,
so UMT5Loader overrides set_stack to keep each layer's own table.

UMT5 tokenizers already contain the <extra_id_*> sentinels, so the
vocabulary is padded with placeholder tokens instead of appending
duplicate sentinel names.

Converted models require runtime support for per-layer position bias
(added separately); older runtimes will silently reuse the first layer's
bias for every layer.
```

**Commit 2 — the dispatch fallback (only if W3 is in scope):**
```
Fall back to the declared architecture when config.json has no model_type

Some Hub repositories published before transformers 4.31 omit the
"model_type" key that AutoConfig requires, including google/umt5-*.
Conversion failed with "Unrecognized model" before the loader registry
was consulted. When AutoConfig fails, dispatch on config.json's
"architectures" entry instead; behaviour is unchanged when it succeeds.
```
