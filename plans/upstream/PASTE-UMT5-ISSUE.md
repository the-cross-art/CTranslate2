UMT5 support: per-layer relative attention bias without regressing T5/mT5

---

### Context

[#1478](https://github.com/OpenNMT/CTranslate2/issues/1478) requested UMT5 support and was closed without a fix. Its one comment identified the root cause correctly and proposed removing the `position_bias->empty()` guard in `dot_product_attention`, but noted:

> However, this approach may lead to performance degradation in T5 and MT5 models.

That objection is what stalled it — the fix would make every existing T5/mT5 model recompute the position bias in every layer to support one new architecture.

This proposes an approach without that cost. I have it implemented and validated, and would like to confirm the design before opening a PR.

### The architecture difference

T5 and mT5 keep one relative attention bias table, in layer 0, shared by all layers. UMT5 gives every layer its own. Measured on the checkpoints:

| model | encoder tables | decoder tables |
|---|---|---|
| `t5-small` | 1 | 1 |
| `google/mt5-small` | 1 | 1 |
| `google/umt5-small` | 8, all pairwise distinct | 8, all pairwise distinct |

### Two problems

**1. Converter.** `_MODEL_LOADERS` has no `UMT5Config` entry, so conversion fails with `No conversion is registered for the model configuration UMT5Config` — the error in #1478.

Registering `UMT5Config` against `T5Loader` is not sufficient, and fails silently: `T5Loader.set_stack` (`converters/transformers.py:1294-1302`) reads each layer's bias, then overwrites layers 1..N with layer 0's. For UMT5 that discards 14 of 16 real tables, converts without error, and produces fluent but incorrect output.

**2. Runtime.** `TransformerEncoder::operator()` (`src/layers/transformer.cc:460`) and `TransformerDecoder::operator()` (`:731`) create one `position_bias` buffer per forward pass and thread it through every layer. `attention.cc:237` fills it only when empty, so layer 0's bias is used everywhere.

### Proposed approach

The per-layer path already exists — `attention.cc:233-235` falls back to a local buffer when the caller passes `nullptr`. It is simply never reached. So rather than disabling the cache for all models, detect which regime a model is in and pass `nullptr` only for the per-layer case.

Detection needs no new config flag and no format change. `_alias_variables` (`specs/model_spec.py:169-189`) already serializes byte-identical tensors as aliases, and `register_variable_alias` (`models/model.cc:279-283`) resolves an alias to the same `shared_ptr<StorageView>` — preserved across device copies by `Model::copy_to` (`model.cc:809-831`). Comparing resolved pointers therefore distinguishes the two cases:

```cpp
static bool has_shared_position_bias(const models::Model& model,
                                     const std::string& scope,
                                     const size_t num_layers) {
  const auto bias_name = [&scope](size_t i) {
    return scope + "/layer_" + std::to_string(i) + "/self_attention/relative_attention_bias";
  };
  const StorageView* first = model.get_variable_if_exists(bias_name(0));
  if (!first)
    return true;  // no relative attention bias: the shared buffer is never filled
  for (size_t i = 1; i < num_layers; ++i) {
    if (model.get_variable_if_exists(bias_name(i)) != first)
      return false;
  }
  return true;
}
```

Confirmed on disk: `t5-small` stores 2 bias variables plus 10 aliases; `google/umt5-small` stores 16 variables and 0 aliases. Models with no relative attention bias take the early return and are untouched, as are models with fewer than two layers.

I believe this is sound in every case: identical tables (aliased) make the cache valid by definition, distinct tables take the per-layer path, and a partially-aliased model still takes the per-layer path and computes each layer correctly. If the aliasing behaviour ever changed, T5 would fall back to the per-layer path — slower, but never incorrect.

### No-regression evidence

Since the performance objection is what closed #1478, this is the part I would most like reviewed:

- **Detection:** `umt5-small` → `shared_position_bias=0` on both stacks; `t5-small` and `mt5-small` → `1` on both stacks. T5 and mT5 keep the shared buffer and the existing code path exactly.
- **`t5-small` `model.bin` is byte-identical** (same sha256) before and after the converter change.
- **C++ suite on macOS/CPU:** 197 ran, 196 passed, 1 skipped — identical to baseline.
- **`t5-small` numerical parity vs Hugging Face: `max |diff| = 1.07e-06`**, unchanged.

### Does it work?

Same model, same harness, only the runtime differs:

| branch | greedy 20 tokens vs Hugging Face |
|---|---|
| converter fix only | diverges at token 1 |
| + runtime fix | exact match, all 20 tokens |

### Validation on a production checkpoint and on GPU

Beyond the small models, this was run on Linux + CUDA (NVIDIA L4) against a 2.9B-parameter UMT5 checkpoint (24 encoder + 24 decoder layers):

- Correct conversion: 32128/32128 vocabulary entries, 0 duplicates; per-layer bias tables matching the HF checkpoint exactly.
- End-to-end: 139/140 prompts valid on a production benchmark at `float32`, replacing a prior 100%-failure path.
- **C++ test suite on Linux + CUDA: 344/345 passed, 1 pre-existing skip, 0 failures** — including every suite exercising the changed code (Attention, Transformer, Model, Translator, LayerDevice, CPU and GPU). Two further tests crashed for environment reasons (`Conv1DGroupNoBiasQuantized` on CPU; `Conv1D/float16` hitting a cuBLASLt symbol issue); `git diff` confirms neither is touched by the two changed files, and excluding them the run is clean.

### A float16 caveat that this change does not cause

At `compute_type=float16` on GPU, the 2.9B checkpoint degenerates to `<unk>`. I traced it rather than assume, and it appears to be a pre-existing T5-family fp16 dynamic-range problem:

- Reproduces at every beam size including beam=1 under forced-synchronous CUDA, so it is not a race.
- `t5-small` at fp16/GPU is correct, so it is not fp16 in general.
- **`google/mt5-small`, which takes the old untouched shared-bias path, also hard-crashes at fp16/GPU — and reproduces identically on the stock unpatched CTranslate2 4.8.2 PyPI wheel.** That is the decisive control: the failure class predates and is independent of this change.
- Temporary instrumentation in `attention.cc` showed bias values clean at every layer (no NaN/Inf, sane magnitudes), while attention output went from 0 NaN, to 992/8192 NaN at layer 4, to 100% NaN from layer 5 — progressive activation overflow, not a bias-computation bug.
- `bfloat16` makes beam=1 exactly correct, but beam≥2 degenerates into repetition — a separate issue I have not explored.

Suggest documenting `compute_type=float32` for large T5-family models on GPU. Happy to file the fp16 overflow separately if it is news to you.

### One thing I have not resolved

Greedy and beam output match Hugging Face exactly, but teacher-forced per-token log-probs for UMT5 still differ by up to 0.605 (mean 0.206), against a 1.07e-06 noise floor measured on `t5-small` with the same harness, in plain float32.

I do not believe this is caused by the change above, and ruled out the obvious candidates:

- A 1-layer UMT5 — where `shared_position_bias=1`, i.e. the original code path — still differs (0.022).
- Zeroing every bias table still differs (0.025), so it is not bias-related at all.
- The error is flat in sequence length (0.148 at 2 tokens, 0.219 at 17), so it is not accumulating through attention.
- Not the activation (CTranslate2's `gelu_tanh_func` matches HF's `NewGELUActivation` exactly; swapping to gated-ReLU changes nothing), not `head_dim` (injecting `d_kv=64` instead of the default `d_model/num_heads` = 512/6 = 85 changed nothing), and not a scalar factor on the logits (no α fits).

Since `argmax` is unaffected, generation is correct and only scores/sampling are affected. If this is a known characteristic of the T5-family path I would rather understand it before opening a PR.

### Also worth knowing

Stock `google/umt5-*` repositories publish a `config.json` with no `model_type` key, so `AutoConfig.from_pretrained` raises `Unrecognized model` at `converters/transformers.py:107`, before the loader registry is consulted. Registering the loader alone therefore does not make Google's own checkpoints convertible without adding that key locally. I would be happy to add a fallback that dispatches on `config.json`'s `architectures` entry — it would help any pre-4.31 repository, not just UMT5 — but that is a separate change.

### Questions

1. Is the pointer-identity detection acceptable, or would you prefer an explicit spec attribute? A flag would need the converter to set it and old models to default to shared. Neither requires a `spec_revision` bump — and I would argue against one, since `TransformerSpec` is shared by T5, NLLB, BART, Marian, M2M-100 and Pegasus, so bumping it would make every newly converted model of all of those unloadable on existing deployments.
2. Any insight on the log-prob residual above?
3. Would you want the `model_type` fallback in the same PR or separately?

The implementation is done and validated as described; happy to open a PR on confirmation. The diff is ~99 lines across `converters/transformers.py`, `src/layers/transformer.cc` and `include/ctranslate2/layers/transformer.h`.

### Environment

```
CTranslate2 4.8.2 (master), transformers 5.17.0, torch 2.14.0, Python 3.12

Development / parity  : macOS arm64, CPU float32 (Accelerate + Ruy, no OpenMP)
Production validation : Linux + CUDA, NVIDIA L4, 2.9B-param UMT5 (24+24 layers)
```

### Disclosure

I investigated and implemented this with AI assistance (Claude). I reproduced every number above myself, ran the test suites on both machines, and verified the source references before filing. I am responsible for the correctness and design of the change and happy to discuss any part of it.
