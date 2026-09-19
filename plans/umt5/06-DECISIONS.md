# Decision log & evidence log

Fill this in **as you go**, not at the end. An unfilled evidence slot means the gate it
belongs to has not passed, regardless of what the terminal printed.

---

## Part A — Decisions

| ID | Decision | Rationale | Status |
|---|---|---|---|
| D-01 | UMT5 support requires **both** a converter change and a runtime change | `T5Loader.set_stack` discards per-layer biases (`transformers.py:1294-1302`) **and** the encoder/decoder share one `position_bias` buffer (`transformer.cc:460`, `:731`). Fixing either alone leaves the model silently wrong. | **Decided** |
| D-02 | Detect share-vs-per-layer by **pointer identity** (Option R), not a new model flag | `_alias_variables` (`model_spec.py:169-189`) + `register_variable_alias` (`model.cc:279-283`) make T5's per-layer lookups return one shared `StorageView*`. Zero Python change, zero format change, zero behaviour change for existing models. | **Proposed** — confirm with E-08 and U-10 before locking |
| D-03 | Do **not** bump `TransformerSpec.revision` (currently 7) or `binary_version` (6) | A revision bump makes every newly converted model, plain T5 included, unloadable on older CTranslate2 builds. The cost lands on users who aren't converting UMT5 at all. | **Decided** |
| D-04 | Development target is `google/umt5-small`; the **user's** model is the acceptance target | Fast gates during development; correctness where it counts. | **Decided** |
| D-05 | No perf work on `compute_relative_bias` | Going 1× → N× is real but small next to N FFN passes. Measure (U-12), record, defer. | **Decided** |
| D-06 | The decoder's sliding-window chunk loop reusing one `position_bias` across chunks (`transformer.cc:731` outside the chunk loop at `:757`) is **pre-existing and out of scope** | No shipped model combines `decoder/sliding_window` with relative attention bias. The per-layer path sidesteps it entirely. | **Decided** — add a code comment, do not fix |
| D-07 | Flash attention + relative attention bias is **out of scope** | `WITH_FLASH_ATTN` is OFF by default (`CMakeLists.txt:25`) and `model.cc:198` disables it outside float16/bfloat16. If `flash_attention.cc` ignores the bias, that is a pre-existing T5 issue to report, not to fix here. | **Pending** — resolve with E-11 |
| D-08 | Whether `google/umt5-small` (~1.5 GB) belongs in CI | Compare against the sizes already downloaded by `test_transformers.py`. If out of line, propose a smaller UMT5 checkpoint or `@pytest.mark.slow`. | **Open** |
| D-09 | Encoder-only `UMT5EncoderModel` support | Deferred so Phase 3 has one parity target. Natural follow-up. | **Deferred** |
| D-10 | int8 / float16 **parity** is not asserted; smoke-tested only | Parity work belongs on float32 CPU where differences are attributable. | **Decided** |
| D-11 | How to handle transformers forcing `tie_word_embeddings=True` (N-02 / F-04) | Recommended: **match HF** — it is the converter contract, and diverging makes the parity harness meaningless. Cost: conversion output is transformers-version-dependent, and parity will be green either way so the harness cannot catch a wrong choice. Must be decided consciously and documented. | **Open — decide before Phase 1** |
| D-12 | How to handle the missing `model_type` (N-01 / F-03) | Options: (a) docs only, (b) fall back to `config.json`'s `architectures[0]`, (c) hybrid. Recommended **(c)**. **Downgraded from blocking to optional:** O-1 established the user's own model carries `model_type`, so this is not needed to unblock them — it is needed to support stock `google/umt5-*` and every other pre-4.31 repo. Given the stated open-source goal it is the most independently upstreamable piece of the project. Keep it in its own commit, after W1/W2 work. | **Open — not blocking; defer to after Gate 1** |
| D-13 | Pin `transformers==5.9.0.*` / `torch==2.12` to match repo CI (N-03) | We are on 5.17.0/2.14.0. N-02's behaviour must be confirmed on 5.9.0 before any test expectation is generated. | **Open** |

### Decisions still genuinely open — needing the user

- **O-1 — What is in your model's `config.json`?** ✅ **PARTLY ANSWERED.** The user reported
  `ValueError: No conversion is ... for the model configuration UMT5Config`, i.e. the error
  raised at `transformers.py:115`. Reaching that line means `AutoConfig.from_pretrained`
  **succeeded** and returned a `UMT5Config` — so their `config.json` **does** carry
  `model_type: "umt5"` and was saved by transformers ≥ 4.31.
  ⇒ **N-01 does not block this user.** W1 (the loader) is what unblocks them.
  **Still unknown for their custom checkpoint:** vocabulary shape (K-12/K-13), whether its
  bias tables are actually per-layer (it may be a fine-tune of something T5-shaped), and its
  `tie_word_embeddings` state. Run `scripts/diagnose_model.py <their path>` to close this.
- **O-2 — Local fork or upstream PR?** ✅ **ANSWERED: local fork for now.** Docs/CHANGELOG and
  the AI-disclosure work drop out of Phase 3 scope until this is revisited. The user's stated
  goal is an eventual open-source contribution covering broad model support, so keep commits
  clean and separable — D-12's fallback in particular is independently upstreamable.
- **O-3 — Is CUDA in scope?** Nothing in Phases 0-3 runs on GPU on this machine. If you need
  GPU, someone with the hardware must re-run E-12 and the Phase 3 parity checks there.
- **O-4 — Which `transformers` version did your failing conversion use?** If it predates UMT5
  support, your error may have a second cause on top of K-08.

---

## Part B — Evidence log

Paste **actual output**. Not "OK". Not "passed". The literal text.

| ID | What | Phase | Value |
|---|---|---|---|
| E-00 | Python / torch / transformers versions | 0.2 | ✅ python 3.12.11, torch 2.14.0, **transformers 5.17.0**, macOS arm64, uv venv `.venv-umt5`. ⚠️ repo CI pins `transformers==5.9.0.*`, `torch==2.12` — see N-03 |
| E-01 | `ctranslate2_test ./tests/data` baseline on `master` | 0.3b | |
| E-02 | VERIFY-1 answers Q1–Q7 | 0.4 | ✅ **All answered.** Q1 yes (`has_relative_attention_bias=True` unconditional). Q2 yes. Q3 cross-attn `False`. Q4 names identical to T5. Q5 `self.scaling = 1.0` w/ upstream comment. Q6 per-config. Q7 `layer_idx` present. → `07-VERIFIED-FINDINGS.md` F-01/F-02 |
| E-03 | `UMT5Config` field dump | 0.4 | ✅ d_model 512, layers 8/8, heads 6, d_kv 64, d_ff 1024, gated-gelu→`gelu_new`, vocab 256384, buckets 32, max_dist 128, tie_word_embeddings **False in json / True as loaded (N-02)**, decoder_start 0. **K-10 and K-11 cannot fire.** |
| E-04 | The verbatim conversion `ValueError` | 0.5 | ✅ **Both reproduced.** Stock `google/umt5-small` → `Unrecognized model ... Should have a model_type key` (N-01, at `transformers.py:107`). **User's model** → `No conversion is [registered] for the model configuration UMT5Config` (`:115`) ⇒ their config has `model_type`, AutoConfig succeeded, **N-01 does not affect them**. |
| E-05 | Per-layer bias identity matrix | 0.6 | ✅ **DECISIVE.** umt5-small: encoder 8 tables `(32,6)` all pairwise distinct; decoder 8 tables all pairwise distinct. t5-small: 1 encoder + 1 decoder self-attn table. mt5-small: 1 + 1. **Premise proven; Phase 2 required.** |
| E-06 | Vocab-size diagnostic | 1.3 | 🔴 **K-12 confirmed real.** vocab_size 256384, len(tokenizer) 256300, gap **84**, `<extra_id_*>` already present: **300**. Inherited loop would append 84 duplicates. Fix now mandatory. |
| E-06b | `len(vocabulary) == embedding rows`, encoder and decoder | G1.5 | |
| E-07 | Converted per-layer bias fingerprints vs Phase-0 `.npy`, exact match | G1.3 | |
| E-08 | Aliasing check: UMT5 not aliased, t5-small aliased | G1.4 | |
| E-09 | `shasum` of t5-small/mt5-small/t5gemma `model.bin`, before vs after the Phase-1 edit | G1.6 | |
| E-10 | UMT5 translation output on the Phase-1 branch ("pre-fix", expected wrong) | G1.8 | |
| E-11 | Does `flash_attention.cc` use its `position_bias` argument? | 2.4.d | |
| E-12 | Detection log: `shared_position_bias=` for umt5 / t5-small / mt5-small / bart-base | 2.5 | |
| E-13 | t5-small + mt5-small output, `master` vs Phase-2, `diff` result | G2.4 | |
| E-14 | **Falsification**: P2 fails on Phase-1, passes on Phase-2 | 3.3 | |
| E-15 | Parity numbers P1–P4, with the tolerance actually used | G3.1 | |
| E-16 | X-01 … X-10 probe results, including the ones that found nothing | 3 | |

---

## Part C — Running notes

Append dated entries. Record surprises, dead ends, and anything that contradicted this plan.
A plan that survives contact with the code unchanged usually means nobody checked.

```
[date] [phase] [what happened] [what it changed about the plan]
```

---

## Part D — Honest status summary (rewrite at the end of every phase)

> **Current status (2026-09-20):** Planning complete. **Phase 0's HF half is done and the
> premise is proven** — see `07-VERIFIED-FINDINGS.md`. torch 2.14 / transformers 5.17 are
> installed in `.venv-umt5`; `google/umt5-small` weights are downloaded and inspected.
>
> **Verified:** every UMT5 layer owns a distinct relative attention bias (8 encoder + 8
> decoder, all pairwise distinct; t5-small/mt5-small have one each) — so the C++ change is
> genuinely required. Module names match T5, so loader inheritance is safe. K-10 and K-11
> cannot fire. K-12 is real.
>
> **Newly found by measurement:** N-01 (`AutoConfig` fails on stock `google/umt5-*` before the
> registry is reached — a second, independent blocker) and N-02 (transformers hard-forces
> `tie_word_embeddings=True`, silently and version-dependently).
>
> **Still entirely unverified:** everything C++. CTranslate2 has **not** been built. The Phase 2
> analysis (shared `position_bias` at `transformer.cc:460`/`:731`, the `nullptr` escape hatch at
> `attention.cc:233-235`, and the pointer-identity detection) is read from source and is
> internally consistent, but **no line of it has been run.** Do not treat Phase 2 as
> established until Gate 2's evidence E-12 exists.
>
> **O-1 resolved:** the user's error was the registry error at `transformers.py:115`, so their
> `config.json` carries `model_type` and N-01 does not block them. W1 (the loader) is the
> unblocking change; W3 (dispatch fallback) is deferred to after Gate 1 as an independent,
> upstreamable improvement.
>
> **Blocking on the user:** D-11 (tie_word_embeddings). Wanted but not blocking: a
> `scripts/diagnose_model.py` run against their actual checkpoint, to confirm it is genuinely
> UMT5-shaped rather than a T5-shaped fine-tune, and to capture its vocabulary numbers.
