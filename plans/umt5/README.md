# UMT5 support in CTranslate2 — plan index

Base: `master` @ `d44d2d06` (v4.8.2).

**Short answer to "is it possible?" — yes.** ~95 lines of Python and ~25 lines of C++.
It is *not* a one-line converter registration; that shortcut converts and runs and is
silently numerically wrong.

**Status:** the HF half of Phase 0 is **done and the premise is proven** — see
`07-VERIFIED-FINDINGS.md`. CTranslate2 itself has **not** been built, so every C++ claim
below is read from source and unverified at runtime.

Read in order:

| File | What it is |
|---|---|
| [`00-OVERVIEW.md`](00-OVERVIEW.md) | Feasibility, root cause with file:line evidence, design decision, branch strategy, explicit in/out of scope |
| [`01-PHASE-0-baseline.md`](01-PHASE-0-baseline.md) | Build the tree, reproduce the error, capture HF ground-truth tensors |
| [`02-PHASE-1-converter.md`](02-PHASE-1-converter.md) | `UMT5Loader` in `transformers.py` |
| [`03-PHASE-2-runtime.md`](03-PHASE-2-runtime.md) | Per-layer position bias in `transformer.cc` |
| [`04-PHASE-3-validation.md`](04-PHASE-3-validation.md) | Numerical parity harness, tests, docs |
| [`05-ERROR-CATALOG.md`](05-ERROR-CATALOG.md) | 14 known-known errors, 12 known-unknown probes, 10 unknown-unknown experiments |
| [`06-DECISIONS.md`](06-DECISIONS.md) | Decision log + evidence log — **fill in as you go** |
| [`07-VERIFIED-FINDINGS.md`](07-VERIFIED-FINDINGS.md) | **What was actually measured.** Wins any conflict with 00-05. |

`scripts/` holds the harness written during Phase 0 (`hf_reference.py`,
`inspect_ct2_model.py`, `parity.py`). `artifacts/` (git-excluded) holds the captured tensors.

## The three bugs, in one paragraph each

**Converter.** `T5Loader.set_stack` (`python/ctranslate2/converters/transformers.py:1294-1302`)
reads each layer's relative attention bias and then, for every layer past the first,
overwrites it with layer 0's — correct for T5/mT5, where only layer 0 has one, and
destructive for UMT5, where every layer has its own.

**Config dispatch** (found by measurement, not in the first draft). Stock `google/umt5-*`
repositories publish a `config.json` with **no `model_type` key**, so
`AutoConfig.from_pretrained` at `transformers.py:107` raises `Unrecognized model` *before* the
loader registry at `:113` is consulted. Registering `UMT5Config` alone therefore does not make
Google's own checkpoints convertible.

**Runtime.** `TransformerEncoder::operator()` (`src/layers/transformer.cc:460`) and
`TransformerDecoder::operator()` (`:731`) create **one** `position_bias` buffer per forward
pass and thread it through every layer; `attention.cc:237` fills it only when empty, so
layer 0's bias is used for all layers. The per-layer path already exists at
`attention.cc:233-235` (when the caller passes `nullptr`) — it is simply never taken.

## Design decision to confirm before writing C++

Detect "shared vs per-layer" by **pointer identity** of the resolved bias variables: the
converter serializes identical tables as aliases (`model_spec.py:169-189`) and the loader
resolves an alias to the *same* `shared_ptr<StorageView>` (`model.cc:279-283`), preserved
across device copies (`model.cc:812-829`). No new model attribute, no format version bump,
no behaviour change for any existing model. Phase 1 gate **G1.4** and probe **U-10** are what
confirm this; a flag-based fallback is specified in Phase 2 §2.6.

## Confirmed by measurement (2026-09-20)

- 8 encoder + 8 decoder bias tables in `google/umt5-small`, **all pairwise distinct**;
  `t5-small` and `mt5-small` have exactly one each. The C++ change is genuinely required.
- UMT5 module names are identical to T5 → the loader can inherit almost everything.
- Two planned conditional fixes (`num_decoder_layers`, `dense_act_fn`) **cannot fire** — deleted.
- The vocabulary-padding bug **is real**: 84 duplicate `<extra_id_*>` tokens, silent.
- **New:** transformers 5.x hard-forces `tie_word_embeddings=True` for UMT5, overriding the
  checkpoint. HF and CT2 read the same wrong flag, so parity stays green — a trap the harness
  cannot catch. Conversion output is transformers-version-dependent.

## Before starting

Three open items in `06-DECISIONS.md` need answers: **O-1** (what's in your model's
`config.json` — decides which of the two errors you hit), **D-11** (the `tie_word_embeddings`
decision) and **D-12** (how to handle the missing `model_type`). Upstream-vs-fork is settled:
local fork for now.
