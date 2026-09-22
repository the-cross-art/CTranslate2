UMT5 support: per-layer relative attention bias without regressing T5/mT5

---

The request for UMT5 support in issue #1478 was closed without resolution. One remark accurately pinpointed the underlying issue and suggested eliminating the `position_bias->empty()` safeguard in `dot_product_attention`, yet observed:

> However, this approach may lead to performance degradation in T5 and MT5 models.

That objection is what hindered progress — the modification would require every current T5/mT5 model to recalculate the position bias in each layer to accommodate a new architecture.

This suggests a method devoid of that expense. I have executed and verified the implementation, and I wish to affirm the design prior to submitting a pull request.

The architectural distinction is that T5 and mT5 utilise a singular relative attention bias table in layer 0, which is uniformly shared across all levels. UMT5 allocates a distinct entity to each layer. Evaluated at the designated markers:

| model | encoder tables | decoder tables |
|---|---|---|
| `t5-small` | 1 | 1 |
| `google/mt5-small` | 1 | 1 |
| `google/umt5-small` | 8, all pairwise distinct | 8, all pairwise distinct |

### A pair of issues

**1. Converter.** The `_MODEL_LOADERS` lacks an entry for `UMT5Config`, resulting in a conversion failure with the message `No conversion is registered for the model configuration UMT5Config` — the mistake referenced in #1478.

Registering `UMT5Config` with `T5Loader` is inadequate and results in silent failure: `T5Loader.set_stack` (`converters/transformers.py:1294-1302`) accesses the bias of each layer and subsequently overwrites layers 1..N with layer 0's. For UMT5 this eliminates 14 out of 16 actual tables, processes without mistakes, and generates articulate yet erroneous results.

**2. Execution.** `TransformerEncoder::operator()` (`src/layers/transformer.cc:460`) and `TransformerDecoder::operator()` (`:731`) generate a singular `position_bias` buffer for each forward pass and propagate it through all layers. `attention.cc:237` populates solely when devoid of content, so the bias of layer 0 is universally applied.

The pathway for each layer is already established — `attention.cc:233-235` reverts to a local buffer when the caller provides `nullptr`. It is simply never reached. Instead of deactivating the cache for all models, identify the regime of each model and provide `nullptr` solely for the per-layer scenario.

Detection requires neither a new configuration flag nor a modification in format. The function `_alias_variables` (`specs/model_spec.py:169-189`) serialises byte-identical tensors as aliases, while `register_variable_alias` (`models/model.cc:279-283`) links an alias to the identical `shared_ptr<StorageView>`, maintained throughout device transfers via `Model::copy_to` (`model.cc:809-831`). Evaluating resolved pointers so differentiates the two scenarios:

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

Confirmed on disk: `t5-small` contains 2 bias variables and 10 aliases; `google/umt5-small` contains 16 variables and no aliases. Models devoid of relative attention bias achieve early returns and remain unaffected, as do models with less than two layers.

I contend that this holds true universally: identical tables (aliased) inherently validate the cache, disparate tables follow the per-layer trajectory, and a partially-aliased model also adheres to the per-layer path while accurately computing each layer. Should the aliasing behaviour alter, T5 would revert to the per-layer approach — less efficient, yet consistently accurate.

### No-regression evidence

Given that the performance concern is what led to the closure of #1478, this is the aspect I would prefer to be evaluated:

- **Identification:** `umt5-small` exhibits `shared_position_bias=0` across both stacks; whereas `t5-small` and `mt5-small` display `1` on both stacks. T5 and mT5 maintain the identical shared buffer and the current code pathway.
- The `t5-small` `model.bin` remains byte-identical (same sha256) prior to and after the converter modification.
- **C++ suite on macOS/CPU:** 197 executed, 196 succeeded, 1 omitted — consistent with baseline.
- **`t5-small` numerical parity compared to Hugging Face: `max |diff| = 1.07e-06`**, remains constant.

### Does it work?

Identical model, identical harness, but the runtime varies:

| branch | greedy 20 tokens versus Hugging Face |
|---|---|
| converter fix only | diverges at token 1 |
| + runtime fix | exact match, all 20 tokens |

### Validation at a production checkpoint and utilising GPU resources

In addition to the compact models, this was executed on Linux + CUDA (NVIDIA L4) utilising a 2.9B-parameter UMT5 checkpoint (24 encoder + 24 decoder layers):

- Accurate transformation: 32128/32128 vocabulary entries, no duplication; per-layer bias tables correspond precisely with the HF checkpoint.
- Comprehensive: 139 out of 140 prompts are valid on a production benchmark at `float32`, superseding a previous path with a 100% failure rate.
- **C++ test suite on Linux with CUDA: 344 out of 345 tests passed, 1 pre-existing skip, and 0 failures** — encompassing all suites that engage the modified code (Attention, Transformer, Model, Translator, LayerDevice, CPU, and GPU). Two further tests failed due to environmental factors (`Conv1DGroupNoBiasQuantized` on CPU; `Conv1D/float16` encountering a cuBLASLt symbol problem); `git diff` verifies that neither is affected by the two modified files, and excluding them results in a clean run.

### A float16 warning that this modification does not induce

With `compute_type=float16` on GPU, the 2.9B checkpoint deteriorates to `<unk>`. I investigated it instead of making assumptions, and there seems to be an existing issue related to the T5-family fp16 dynamic range.

- It replicates at all beam sizes, including beam=1, under enforced synchronous CUDA, hence it is not a race condition.
- The `t5-small` model operates correctly at fp16 on GPU, indicating that it is not universally fp16.
- **The `google/mt5-small` model, which follows the original unaltered shared-bias route, similarly experiences a complete failure at fp16/GPU — and delivers the same results on the standard unmodified CTranslate2 4.8.2 PyPI package.** The failure class exists prior to and independently of this alteration, representing the critical control.
- Provisional apparatus in `attention.cc` showed the bias values were consistently clean across all layers (no NaN/Inf, reasonable magnitudes), however the attention output transitioned from 0 NaN, to 992/8192 NaN at layer 4, and ultimately to 100% NaN from layer 5 — indicative of gradual activation overload, rather than a bias-computation error.
- The `bfloat16` format makes beam=1 precisely accurate, although beam≥2 deteriorates into redundancy — a distinct matter I have not investigated.

Recommend recording `compute_type=float32` for substantial T5-family models utilising GPU. Pleased to submit the fp16 overflow independently if this is unfamiliar to you.

### One unresolved issue

Teacher-forced per-token log-probs for UMT5 can differ by as much as 0.605 (mean 0.206), despite greedy and beam outputs aligning perfectly with Hugging Face, against a noise floor of 1.07e-06 observed on `t5-small` using the same framework in plain float32. I am convinced that this variation is not attributable to the aforementioned change and have eliminated the obvious possibilities.

- A single-layer UMT5, with `shared_position_bias=1`, which corresponds to the original code path, exhibits a difference of 0.022.
- The discrepancy persists when zeroing each bias table (0.025), indicating that it is not associated with bias.
- The error remains constant in sequence length (0.148 at 2 tokens, 0.219 at 17), indicating it is not compounding via attention.
- Not the activation function (CTranslate2's `gelu_tanh_func` aligns perfectly with HF's `NewGELUActivation`; altering to gated-ReLU yields no difference), nor `head_dim` (substituting `d_kv=64` for the standard `d_model/num_heads` = 512/6 = 85 produced no change), nor a scalar multiplier on the logits (no α is suitable).

As `argmax` remains unchanged, the generation is accurate, with only the scores and sampling being influenced. If this is a recognised trait of the T5-family path, I would prefer to comprehend it prior to submitting a PR.

### Also worth noting

Stock `google/umt5-*` repositories provide a `config.json` that lacks a `model_type` key, resulting in `AutoConfig.from_pretrained` triggering an `Unrecognized model` error at `converters/transformers.py:107`, prior to the consultation of the loader registry. Merely registering the loader does not render Google's checkpoints convertible unless the key is added locally. I would gladly implement a fallback that operates based on the `architectures` entry in `config.json` — this would benefit all pre-4.31 repositories, not solely UMT5 — however, that is a distinct modification.

### Enquiries

1. Is the detection of pointer identity satisfactory, or would you want a specific specification attribute? A flag would require the converter to set it, while older models would default to shared. Neither necessitates an increase in `spec_revision` — and I would contend against such an action, since `TransformerSpec` is utilised by T5, NLLB, BART, Marian, M2M-100 and Pegasus, thus elevating it would render every freshly converted model from all these frameworks incompatible with current deployments.
2. Do you have any observations regarding the aforementioned log-prob residual?
3. Would you prefer the `model_type` fallback to be included in the same pull request or in a separate one?

The execution has been completed and verified as outlined; I am pleased to initiate a pull request upon confirmation. The difference comprises approximately 99 lines within `converters/transformers.py`, `src/layers/transformer.cc` and `include/ctranslate2/layers/transformer.h`.

### Environment

```
CTranslate2 4.8.2 (master), transformers 5.17.0, torch 2.14.0, Python 3.12

Development / parity   : macOS arm64, CPU float32 (Accelerate + Ruy, no OpenMP)
Production verification: Linux + CUDA, NVIDIA L4, 2.9B-parameter UMT5 (24+24 layers)
```

### Disclosure

I examined and executed this with the aid of AI (Claude). I independently replicated each number mentioned, executed the test suites on both systems, and confirmed the source citations prior to submission. I am accountable for the accuracy and design of the modification and am eager to discuss any aspect of it.
