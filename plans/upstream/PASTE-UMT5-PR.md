Add UMT5 support (per-layer relative attention bias)

---

Closes #2102. Follows up on #1478, which requested UMT5 support and was closed without a fix.

## What this does

UMT5 is structurally identical to mT5 except for one thing: every self-attention layer owns its own relative attention bias table, where T5 and mT5 compute the bias once in layer 0 and share it. CTranslate2 currently cannot convert UMT5 at all, and its runtime assumes the shared-table layout.

Two commits, 96 lines:

1. **`UMT5Loader`** — registers `UMT5Config` and overrides `set_stack` to keep each layer's own bias table. `T5Loader.set_stack` reads each layer's bias and then overwrites layers 1..N with layer 0's, which for UMT5 discards 14 of 16 real tables, converts without error, and produces fluent but incorrect output. It also overrides `get_vocabulary`: UMT5 tokenizers already contain the `<extra_id_*>` sentinels, so the inherited padding would append duplicates.

2. **Per-layer position bias in the Transformer stacks** — `TransformerEncoder::operator()` and `TransformerDecoder::operator()` create one `position_bias` buffer per forward pass and thread it through every layer; `attention.cc` fills it only when empty, so layer 0's bias is used everywhere.

## Why not just remove the cache

The comment on #1478 proposed removing the `position_bias->empty()` guard, and noted it "may lead to performance degradation in T5 and MT5 models". That is what stalled it — it would make every existing T5/mT5 model recompute the bias in every layer.

Instead, this detects which layout a model has and only takes the per-layer path when needed. The per-layer path already exists in `MultiHeadAttention`: when the caller passes `nullptr`, each layer falls back to a local buffer and computes from its own table. This change makes that branch reachable rather than adding new computation.

Detection uses pointer identity and needs no new config flag, no format change, and no `spec_revision` bump. `_alias_variables` already serializes byte-identical tensors as aliases; `register_variable_alias` resolves an alias to the same `shared_ptr<StorageView>`, and `Model::copy_to` preserves that across device copies. So for T5 every layer's bias resolves to one pointer, and for UMT5 they differ.

Models with no relative attention bias hit the early return, and models with fewer than two layers take the previous path, so both are byte-for-byte unchanged.

## No regressions

- `t5-small` `model.bin` is **byte-identical** (same sha256) before and after the converter change.
- Detection: `umt5-small` → per-layer on both stacks; `t5-small` and `mt5-small` → shared on both stacks, i.e. the existing code path exactly.
- `t5-small` numerical parity vs Hugging Face unchanged at `max |diff| = 1.07e-06`.
- C++ test suite, macOS/CPU: 197 ran, 196 passed, 1 skipped — identical to baseline.
- C++ test suite, Linux + CUDA: 344/345 passed, 1 pre-existing skip, 0 failures, covering Attention, Transformer, Model, Translator and LayerDevice on both CPU and GPU.

## Verification

Same model and harness, only the runtime differs:

| | greedy 20 tokens vs Hugging Face |
|---|---|
| converter commit only | diverges at token 1 |
| both commits | exact match, all 20 tokens |

Also validated on a 2.9B-parameter UMT5 checkpoint (24 encoder + 24 decoder layers) on an NVIDIA L4: correct conversion, and 139/140 prompts valid on a production benchmark at `float32`, replacing a path that previously failed 100% of the time.

## Known limitations

- **Teacher-forced per-token log-probs for UMT5 differ from Hugging Face by up to 0.605** (mean 0.206), against a `1.07e-06` noise floor measured on `t5-small` with the same harness in float32. Greedy and beam output match exactly, so generation is correct and only scores/sampling are affected. This is not introduced by these commits — a 1-layer UMT5 on the unchanged shared path still differs (0.022), zeroing every bias table still differs (0.025), and the error is flat in sequence length. I ruled out the activation, `head_dim`, and a scalar logit factor. Detail and reasoning are in #2102; I would appreciate a pointer if this is a known characteristic of the T5-family path.
- **`float16` on GPU is unusable for large T5-family models**, but this predates the change: `google/mt5-small`, which takes the untouched shared-bias path, fails identically on the stock unpatched 4.8.2 PyPI wheel. Instrumentation showed clean bias values with progressive activation overflow from layer 4 onward. `float32` is fine; `bfloat16` is correct at beam=1.
- **Stock `google/umt5-*` repositories omit `model_type` from `config.json`**, so `AutoConfig.from_pretrained` raises before the loader registry is consulted and the key has to be added locally. A fallback dispatching on `config.json`'s `architectures` entry would fix that for any pre-4.31 repository, but it touches shared converter code so I left it out of this PR. Happy to add it here or separately.

## Not included

Docs and CHANGELOG entries — tell me the form you'd like and I'll add them to this PR.

## Disclosure

I investigated and implemented this with AI assistance (Claude). I ran the GPU/CUDA validation and the test suites myself and verified the source references before submitting. I am responsible for the correctness and design of the change and happy to discuss any part of it.
