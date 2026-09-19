# Verified findings — measured, not assumed

Everything below was **run** on 2026-09-20 against real code and real weights.
This file supersedes any assumption in `00`–`05` that contradicts it.

**Environment (evidence E-00):**

```
python        3.12.11
torch         2.14.0
transformers  5.17.0          <-- see F-06, repo CI pins 5.9.0.*
platform      macOS arm64 (Apple Silicon)
venv          .venv-umt5 (uv 0.9.26)
```

CTranslate2 itself is **still not built** — everything here is from the HF side, which
needed no build. The C++ claims in `03-PHASE-2-runtime.md` remain **unverified at runtime**.

---

## F-01 ✅ CONFIRMED — every UMT5 layer owns its own relative attention bias

This was VERIFY-1, the load-bearing premise of the whole plan. It holds, twice over.

**From source** (`transformers/models/umt5/modeling_umt5.py`):

```python
class UMT5LayerSelfAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        self.SelfAttention = UMT5Attention(config, has_relative_attention_bias=True, layer_idx=layer_idx)

class UMT5LayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        self.EncDecAttention = UMT5Attention(config, has_relative_attention_bias=False, layer_idx=layer_idx)
```

`True` unconditionally for self-attention, `False` for cross-attention. Exactly as planned.

**From the actual `google/umt5-small` checkpoint** (evidence E-05):

```
encoder: 8 bias tables, shape (32, 6) each
  identical-to-layer0: [True, False, False, False, False, False, False, False]
  all distinct pairwise: True
decoder: 8 bias tables, shape (32, 6) each
  identical-to-layer0: [True, False, False, False, False, False, False, False]
  all distinct pairwise: True
```

**Negative control:**
```
t5-small        encoder: 1 bias table   decoder: 1 self-attn bias table
google/mt5-small encoder: 1 bias table  decoder: 1 bias table
```

⇒ `T5Loader.set_stack`'s `if i > 0` bias-copy (`transformers.py:1294-1302`) would discard
14 of the 16 real tables. **Phase 2 (the C++ change) is genuinely required.** The plan's
"if this turns out shared, stop and delete Phase 2" branch is now closed: it does not apply.

---

## F-02 ✅ CONFIRMED — module names are identical to T5, so the loader can be inherited

| Item | UMT5 | Matches `T5Loader` expectation |
|---|---|---|
| `.SelfAttention`, `.EncDecAttention` | yes | ✅ `set_self_attention` / `set_cross_attention` |
| `.q .k .v .o` (all `bias=False`) | yes | ✅ `set_attention:1334-1337` |
| `.DenseReluDense.wi_0 / .wi_1 / .wo` | yes (`UMT5DenseGatedActDense`) | ✅ `set_ffn:1313-1317` |
| `.layer_norm`, `.final_layer_norm` (`.weight`, no bias) | yes (`UMT5LayerNorm`) | ✅ `set_layer_norm:1352` |
| `.block`, `.layer`, `.embed_tokens`, `.lm_head` | yes | ✅ `set_stack` |
| `has_relative_attention_bias` attribute | yes | ✅ `set_attention:1345` guard works |

**Q5 answered, and better than hoped** — `UMT5Attention.__init__` contains:

```python
# UMT5 folds the relative position bias into the attention scores and does not scale the query/key dot product.
self.scaling = 1.0
```

This is an explicit upstream confirmation that CT2's `spec.queries_scale = 1.0`
(`transformers.py:1331`) is correct for UMT5. Leave it alone.

---

## F-03 🔴 NEW BLOCKER — `AutoConfig` fails on stock `google/umt5-*` before the registry is ever consulted

**Not in the original plan. Registering `UMT5Config` in `_MODEL_LOADERS` is NOT sufficient.**

`google/umt5-small`'s `config.json` on the Hub has **no `model_type` key**:

```json
{ "architectures": ["UMT5ForConditionalGeneration"], "d_ff": 1024, ...
  "tokenizer_class": "T5Tokenizer", "transformers_version": "4.31.0.dev0", ... }
```

So the very first line of `TransformersConverter._load` (`transformers.py:107-109`) raises:

```
ValueError: Unrecognized model in google/umt5-small.
Should have a `model_type` key in its config.json.
```

…**before** reaching the `_MODEL_LOADERS.get(config_name)` lookup at line 113.

Meanwhile the class itself defines it (`configuration_umt5.py:34` → `model_type = "umt5"`),
so the explicit path works fine:

```
transformers.AutoConfig.from_pretrained("google/umt5-small")   -> ValueError
transformers.UMT5Config.from_pretrained("google/umt5-small")   -> OK, model_type: umt5
```

**⚠️ This means there are TWO different errors a user can hit, and they need different fixes:**

| Their config.json | Error they see | Fixed by |
|---|---|---|
| no `model_type` (stock Google repos) | `Unrecognized model ... Should have a model_type key` | F-03 work below |
| has `model_type: "umt5"` (re-saved by transformers ≥4.31) | `No conversion is registered for the model configuration UMT5Config` | Phase 1 registry entry |

**RESOLVED for this user.** They reported the second error (`No conversion is ... for the
model configuration UMT5Config`), which is raised at `transformers.py:115` — reachable only if
`AutoConfig` already succeeded. So their `config.json` **does** carry `model_type: "umt5"`
and **N-01 does not block them**. It still blocks anyone converting stock `google/umt5-*`.

⇒ F-03 is **downgraded from blocking to optional** (D-12): do W1/W2 first, add the dispatch
fallback afterwards as an independent, upstreamable commit.

**Options for Phase 1 (pick one, record in `06-DECISIONS.md`):**

- **F-03a — document only.** Tell users to add `"model_type": "umt5"` to `config.json`.
  Zero code. Leaves stock Google repos broken out of the box.
- **F-03b — fall back to `architectures[0]`.** In `_load`, catch the `ValueError` and
  re-dispatch using `config.json`'s `architectures` entry, which **is** present
  (`"UMT5ForConditionalGeneration"`). Helps every legacy repo with a missing `model_type`,
  not just UMT5. ~15 lines, touches shared converter code.
- **F-03c — hybrid (recommended).** F-03b, but scoped: only attempt the fallback when
  `AutoConfig` fails, and raise a message naming both the architecture found and the fix.

F-03b/c is arguably the more valuable contribution of the two — it is generic, it is in the
"Hugging Face integrations" category `CONTRIBUTING.md` explicitly invites, and it needs no
C++ at all.

---

## F-04 🟠 NEW SILENT TRAP — transformers 5.x forces `tie_word_embeddings = True` for UMT5

`transformers/models/umt5/configuration_umt5.py:64-78`:

```python
def __post_init__(self, **kwargs):
    ...
    kwargs.pop("tie_word_embeddings", None)
    self.tie_word_embeddings = True   # force it for T5 family
```

It is unconditional. Measured:

```
config.json says tie_word_embeddings   = False
loaded  config  tie_word_embeddings    = True
UMT5Config(tie_word_embeddings=False)  -> True      # even an explicit kwarg is discarded
UMT5Config.from_dict({...: False})     -> True
```

But the checkpoint is **genuinely untied**:

```
lm_head.weight exists, shape (256384, 512)
equal to shared.weight?  False
max abs diff:            116.65
```

and HF detects the conflict at load and refuses to tie:

> *"The tied weights mapping and config for this model specifies to tie shared.weight to
> lm_head.weight, but both are present in the checkpoints with different values, so we will
> NOT tie them. You should update the config with `tie_word_embeddings=False`…"*

**So the weights are fine — but the flag is wrong, and two places read the flag:**

- `modeling_umt5.py:1052-1055` — HF **applies** `sequence_output * d_model**-0.5` before `lm_head`.
- `transformers.py:1252-1253` — CT2 **applies** `spec.decoder.scale_outputs = d_model**-0.5`.

They read the same lying flag, so **CT2 and HF agree and parity will be green.** That is the
good news and also the trap: a green parity harness will *not* flag this.

Consequences to write down now:

1. **Conversion output is transformers-version-dependent.** A transformers 4.x conversion that
   honoured `tie_word_embeddings: false` produces a model **without** `scale_outputs`;
   ours produces one **with** it. Same checkpoint, different CT2 model.
2. If transformers ever fixes `__post_init__`, previously-converted models silently change.
3. Greedy `argmax` is invariant under a positive scalar on the logits, so this will not show
   up as garbage text — it shifts **scores**, softmax sharpness, sampling temperature and
   beam ranking with length penalty. It is exactly the kind of bug that passes a smoke test.

**Decision needed (D-11):** match HF (inherit the behaviour, document loudly, pin the
transformers version in the conversion instructions) — recommended, since "match HF" is the
converter contract — **or** honour `config.json` over the config object. Do **not** decide this
silently. Record the transformers version in the converted model's directory either way.

---

## F-05 🔴 K-12 CONFIRMED REAL — the vocabulary padding would add 84 duplicate tokens

```
config.vocab_size       : 256384
len(tokenizer)          : 256300      (T5Tokenizer, fast and slow agree)
gap                     :     84
<extra_id_*> already in vocab: 300
pad / eos / unk         : <pad> </s> <unk>
```

`T5Loader.get_vocabulary` (`transformers.py:1257-1264`) appends
`<extra_id_0>` … `<extra_id_83>` to close an 84-token gap — but the tokenizer **already
contains all 300 `<extra_id_*>` tokens**. The result is a 256384-entry vocabulary (so the
embedding shape matches and **nothing raises**) containing **84 duplicated token strings**,
with the padding slots 256300-256383 carrying names that collide with the real sentinels.

Latent for plain translation; a genuine bug for any sentinel/span-infilling use — which is
UMT5's own pretraining objective.

**Fix is now mandatory, not conditional.** Override `get_vocabulary` in `UMT5Loader` to pad
with non-colliding placeholders:

```python
    def get_vocabulary(self, model, tokenizer):
        tokens = ModelLoader.get_vocabulary(self, model, tokenizer)
        # UMT5 vocabularies are padded to a multiple of 128; the sentinel <extra_id_*>
        # tokens are already part of the tokenizer, so pad with unused placeholders
        # instead of appending duplicates.
        for i in range(model.config.vocab_size - len(tokens)):
            tokens.append("<unused_%d>" % i)
        if len(tokens) != model.config.vocab_size:
            raise ValueError(
                "Vocabulary size mismatch: tokenizer produced %d tokens but the model "
                "expects %d" % (len(tokens), model.config.vocab_size)
            )
        return tokens
```

> Confirm the placeholder name does not already exist in the vocab before settling on
> `<unused_%d>`, and confirm `ModelLoader.get_vocabulary` is the right super-call
> (`T5Loader.get_vocabulary` must be skipped, not chained — chaining would re-add the
> `<extra_id_*>` padding you are trying to avoid).

---

## F-06 🟡 Version drift — we are ahead of the repo's own CI

`python/tests/requirements.txt` pins:

```
transformers==5.9.0.*;platform_system=='Linux'
torch==2.12
```

We installed **transformers 5.17.0 / torch 2.14.0**. Same major version, so the v5 APIs the
converter uses (`dtype=` kwarg at `transformers.py:134-140`) are right — CTranslate2 4.8.2 is
already a transformers-v5 codebase, which is reassuring.

But F-04's forced `tie_word_embeddings` must be re-checked on **5.9.0**, since that is what
CI will actually run. If 5.9.0 does *not* force it, the converted model differs between your
machine and CI, and the test expectation in Phase 3 §3.6 would be unreproducible.

**Action:** before Phase 1, run the F-04 three-line check under `transformers==5.9.0.*` and
record both results. If they differ, pin the conversion instructions explicitly.

---

## F-07 ℹ️ Harmless oddities — noted so nobody re-investigates them

- **`"scalable_attention": true`** in `google/umt5-small`'s `config.json` appears **nowhere**
  in the entire `transformers` package (`grep -rn "scalable_attention" transformers/` →
  no hits). It is a dead field carried over from the original T5X export. HF ignores it, so
  it cannot cause a CT2/HF divergence. **Ignore it.**
- **`t5-small` carries an unused cross-attention bias** in the checkpoint:
  `decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight`. HF builds
  cross-attention with `has_relative_attention_bias=False`, so it is never loaded, and CT2's
  `set_attention` guard skips it too. Do not be confused when a decoder appears to have
  "2 bias tables" — one is dead weight. `google/mt5-small` does not have it.
- **`num_decoder_layers`** — `UMT5Config.__post_init__` defaults it to `num_layers` when
  `None`, and `google/umt5-small` sets it to `8` explicitly. **K-10 cannot fire.**
- **`dense_act_fn = "gelu_new"`**, `is_gated_act = True` — `"gelu_new"` is in
  `_SUPPORTED_ACTIVATIONS` (`transformers.py:33`) → `Activation.GELUTanh`.
  **K-11 cannot fire.**

---

## Confirmed `google/umt5-small` config

```
d_model 512 | num_layers 8 | num_decoder_layers 8 | num_heads 6 | d_kv 64 | d_ff 1024
feed_forward_proj 'gated-gelu' | dense_act_fn 'gelu_new' | is_gated_act True
vocab_size 256384 | relative_attention_num_buckets 32 | relative_attention_max_distance 128
tie_word_embeddings False (in json; forced True by transformers 5.17 — see F-04)
decoder_start_token_id 0 | pad 0 | eos 1 | layer_norm_epsilon 1e-06
tokenizer_class T5Tokenizer | 206 tensors | pytorch_model.bin only (no safetensors)
```

---

## Net effect on the plan

| Item | Before | Now |
|---|---|---|
| VERIFY-1 / U-01 (premise) | ⚠️ assumed | ✅ **proven** — Phase 2 is required |
| K-10 `num_decoder_layers` | conditional branch | ❌ **deleted** — cannot fire |
| K-11 `dense_act_fn` | conditional branch | ❌ **deleted** — cannot fire |
| K-12 vocabulary padding | conditional branch | 🔴 **confirmed real** — fix is mandatory |
| Module-name check (Q4) | checklist item | ✅ **done** — inheritance is safe |
| `queries_scale = 1.0` (Q5) | to verify | ✅ **confirmed by an upstream comment** |
| — | — | 🟠 **F-03 NEW**: `AutoConfig` dispatch failure — blocks stock `google/umt5-*`, **not** this user (O-1) |
| — | — | 🟠 **F-04 NEW**: forced `tie_word_embeddings`, silent and version-dependent |
| — | — | 🟡 **F-06 NEW**: pin transformers to match CI |

Phase 1 grew by one required fix (F-05) and one design decision (F-03). Phase 2 is unchanged
and confirmed necessary. Nothing found so far contradicts the C++ analysis, but none of it is
verified at runtime yet either.
