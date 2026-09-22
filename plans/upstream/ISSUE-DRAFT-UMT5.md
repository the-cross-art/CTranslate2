# Issue draft — UMT5 support

**Repo:** https://github.com/OpenNMT/CTranslate2/issues/new
**Title:** `UMT5 support: per-layer relative attention bias without regressing T5/mT5`
**Labels to request:** `enhancement`

> **Open a NEW issue, don't comment on #1478.** It was closed 2024-11-18 and is ~2 years
> old; comments on stale closed issues rarely get seen. Link it prominently instead — the
> title above deliberately answers the objection that killed it.

---

## Context

[#1478](https://github.com/OpenNMT/CTranslate2/issues/1478) asked for UMT5 support and was
closed without a fix. The single comment on it identified the root cause correctly and
proposed commenting out the `position_bias->empty()` guard in `dot_product_attention`, but
noted:

> *"However, this approach may lead to performance degradation in T5 and MT5 models."*

That objection is why it stalled — the fix would make every existing T5/mT5 model recompute
the position bias in every layer, to support one new architecture.

**This proposes a fix that doesn't have that cost.** I have it working locally and would like
to check the approach before opening a PR.

## The architecture difference

T5 and mT5 have one relative attention bias table, in layer 0, shared by all layers. UMT5
gives **every** layer its own. Measured on the checkpoints:

| model | encoder tables | decoder tables |
|---|---|---|
| `t5-small` | 1 | 1 |
| `google/mt5-small` | 1 | 1 |
| `google/umt5-small` | **8, all pairwise distinct** | **8, all pairwise distinct** |

## Two problems

**1. Converter.** `_MODEL_LOADERS` has no `UMT5Config` entry, so conversion fails with
`No conversion is registered for the model configuration UMT5Config` — the error in #1478.
But simply registering `UMT5Config` against `T5Loader` is **not** enough, and fails silently:
`T5Loader.set_stack` (`transformers.py:1294-1302`) reads each layer's bias and then
overwrites layers 1..N with layer 0's. For UMT5 that discards 14 of 16 real tables, converts
cleanly, and produces fluent but incorrect output.

**2. Runtime.** `TransformerEncoder::operator()` (`src/layers/transformer.cc:460`) and
`TransformerDecoder::operator()` (`:731`) create one `position_bias` buffer per forward pass
and thread it through every layer. `attention.cc:237` fills it only when empty, so layer 0's
bias is used for all layers.

## Proposed approach

The per-layer path **already exists**: `attention.cc:233-235` falls back to a local buffer
when the caller passes `nullptr`. It's simply never reached. So rather than disabling the
cache for everyone, detect which regime a model is in and pass `nullptr` only for the
per-layer case.

Detection needs no new config flag and no format change. `_alias_variables`
(`specs/model_spec.py:169-189`) already serializes byte-identical tensors as aliases, and
`register_variable_alias` (`models/model.cc:279-283`) resolves an alias to **the same
`shared_ptr<StorageView>`** — preserved across device copies by `Model::copy_to`
(`model.cc:812-829`). So comparing resolved pointers distinguishes the two cases:

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

Confirmed on disk — `t5-small`: 2 bias variables + **10 aliases**; `google/umt5-small`: 16
variables + **0 aliases**. Models with no relative attention bias take the early return and
are untouched.

## No-regression evidence

Since the perf objection is what closed #1478, this is the part I'd most like reviewed:

- **Detection logs:** `umt5-small` → `shared_position_bias=0` (both stacks);
  `t5-small` / `mt5-small` → `1` (both stacks). T5 and mT5 keep the shared buffer and the
  existing code path exactly.
- **`t5-small` `model.bin` is byte-identical** (same sha256) before and after the converter
  change.
- **C++ test suite unchanged:** 197 ran, 196 passed, 1 skipped — identical to baseline.
- **`t5-small` numerical parity vs Hugging Face: `max |diff| = 1.07e-06`**, unchanged.

## Does it actually work?

Falsification check — same model, same harness, only the runtime differs:

| branch | greedy 20 tokens vs HF |
|---|---|
| converter fix only | **diverges at token 1** |
| + runtime fix | **exact match, all 20 tokens** |

## Validation on a real checkpoint and on GPU

Beyond the toy models, this was run against a production 2.9B-parameter UMT5 checkpoint
(24 encoder + 24 decoder layers) on an NVIDIA L4:

- **Conversion:** correct vocabulary (32128/32128, 0 duplicates), per-layer bias tables
  matching the HF checkpoint exactly.
- **End-to-end:** 139/140 prompts valid on a production benchmark at `float32`, replacing a
  prior 100%-failure path. CPU `float32` greedy decode also coherent.
- **C++ test suite on Linux + CUDA:** 344/345 passed, 1 pre-existing skip, 0 failed —
  including every suite touching the changed code (Attention, Transformer, Model, Translator,
  LayerDevice, CPU and GPU). Two further tests crashed for environment reasons
  (`Conv1DGroupNoBiasQuantized` on CPU; `Conv1D/float16` hitting a cuBLASLt symbol issue);
  `git diff` confirms neither is touched by the two changed files, and excluding them the
  run is clean.

### A float16 caveat that is NOT caused by this change

At `compute_type=float16` on GPU, this 2.9B checkpoint degenerates to `<unk>`. I chased it
down rather than assume, and it is a **pre-existing T5-family fp16 dynamic-range problem**:

- Reproduces at every beam size including beam=1 under forced-synchronous CUDA, so it isn't a
  race.
- `t5-small` at fp16/GPU is perfectly correct — so it isn't fp16 generally.
- **`google/mt5-small` — which takes the old, untouched shared-bias path — also hard-crashes
  at fp16/GPU, and reproduces identically on the stock unpatched CTranslate2 4.8.2 PyPI
  wheel.** That is the decisive control: the failure class predates and is independent of this
  change.
- Temporary instrumentation in `attention.cc` showed bias values clean at every layer (no
  NaN/Inf, sane magnitudes); attention output goes 0 NaN → 992/8192 NaN at layer 4 → 100% NaN
  from layer 5. Progressive activation overflow, not a bias-computation bug.
- `bfloat16` (wider exponent) makes beam=1 exactly correct; beam≥2 degenerates into
  repetition — a separate issue I haven't explored.

I'd suggest documenting `compute_type=float32` for large T5-family models on GPU. Happy to
file the fp16 overflow separately if it's news to you.

## One thing I have not resolved — would value your input

Greedy/beam output matches Hugging Face exactly, but **teacher-forced per-token log-probs for
UMT5 still differ by up to 0.605** (mean 0.206), against a `1.07e-06` noise floor measured on
`t5-small` with the same harness.

I do **not** believe this is caused by the change above, and I ruled out the obvious
candidates:

- A **1-layer** UMT5 — where `shared_position_bias=1`, i.e. the original code path — still
  differs (0.022).
- **Zeroing every bias table** still differs (0.025), so it isn't bias-related at all.
- Error is **flat in sequence length** (0.148 at 2 tokens, 0.219 at 17), so it isn't
  accumulating through attention.
- Not the GELU (CT2's `gelu_tanh_func` matches HF's `NewGELUActivation` exactly; swapping to
  gated-ReLU changes nothing), not `head_dim` (injecting `d_kv=64` instead of the default
  `d_model/num_heads = 512/6 = 85` changed nothing), and not a scalar logit factor (no α fits).

Since `argmax` is unaffected, generation is correct and scores are not. If this is a known
characteristic of the T5-family path I'd rather understand it before opening a PR — and if
it's news to you, it may be worth a separate look.

(Separately, I think I've found a related pre-existing issue affecting mT5 under
`transformers` 5.x — I'll file that on its own.)

## Also worth knowing

Stock `google/umt5-*` repositories publish a `config.json` with **no `model_type` key**, so
`AutoConfig.from_pretrained` raises `Unrecognized model` at `transformers.py:107`, before the
loader registry is consulted. Registering the loader alone therefore doesn't make Google's own
checkpoints convertible; they need the key added locally. Happy to add a fallback that
dispatches on `config.json`'s `architectures` entry if you'd want it — it would help any
pre-4.31 repo, not just UMT5 — but that's a separate change.

## Questions

1. Is the pointer-identity detection acceptable, or would you prefer an explicit spec
   attribute? (A flag would need the converter to set it and old models to default to
   shared — no `spec_revision` bump either way, but it touches the spec.)
2. Any insight on the log-prob residual above?
3. Would you want the `model_type` fallback in the same PR or separately?

Happy to open a PR on confirmation — the implementation is done and validated as above.
Diff is ~99 lines across `converters/transformers.py`, `src/layers/transformer.cc`,
`include/ctranslate2/layers/transformer.h`.

## Environment

```
CTranslate2 4.8.2 (master), transformers 5.17.0, torch 2.14.0, Python 3.12

Development/parity : macOS arm64, CPU float32 (Accelerate + Ruy, no OpenMP)
Production validation: Linux + CUDA, NVIDIA L4, 2.9B-param UMT5 (24+24 layers)
```

---

## AI-use disclosure — REQUIRED, and it matters more here

`CONTRIBUTING.md` lines 31-37, plus:

> *"Please contribute within your area of expertise. If you are not familiar with the core
> codebase, consider contributing to documentation, examples, or Hugging Face integrations."*

This change touches `src/layers/` — **the area that guidance steers newcomers away from.**
That's the main risk to this landing, independent of whether the code is right. Which is why:

- Lead with the no-regression evidence, not the diff.
- Disclose the AI use plainly, in your own words.
- Be honest about the unresolved residual — hiding it would be worse than the residual.

Be ready to explain, unaided: why the shared `position_bias` buffer exists, what
`_alias_variables` does and why aliasing makes pointer comparison sound, why no format
version bump is needed, and why a `spec_revision` bump would be harmful (it would make every
newly converted model, plain T5 included, unloadable on older CTranslate2 builds).

If you can't yet, don't file this one — file B-01 first and come back to it.
