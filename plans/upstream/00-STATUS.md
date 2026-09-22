# Upstream status — is anyone already on these?

Researched 2026-09-20 against `OpenNMT/CTranslate2` with `gh`.
(Your `origin` is the fork `the-cross-art/CTranslate2`, so you can push branches freely.)

---

## Summary

| Item | Existing issue? | Anyone working on it? | Verdict |
|---|---|---|---|
| **B-01** mT5 wrongly rescaled under transformers 5.x | **none found** | no | 🟢 **Open field.** Highest impact. |
| **B-02** T5 vocabulary duplicate `<extra_id_*>` | **none found** | no | 🟢 **Open field.** Smallest, cleanest. |
| **UMT5 support** | **#1478, CLOSED** | no PR, no assignee | 🟡 Known want; naive fix was rejected on performance grounds — **ours answers that objection** |

Searches run: `mt5`, `scale_decoder_outputs`, `transformers 5`, `extra_id`, plus the 15
most recent open issues and a PR search for `umt5`. No open PR touches any of this.

---

## The UMT5 issue — #1478, and why it matters to us

**"Support for UMT5"**, opened by `QLutz`, closed 2024-11-18, one comment by `soocheolnoh`.

The reporter hit **your exact error**:

```
ValueError: No conversion is registered for the model configuration UMT5Config
```

The single comment gets the root cause **exactly right** — it points at
`src/layers/attention.cc#L236-L253` and proposes:

> To use the correct `position_bias` for every layer, simply disable the if condition:
> ```c++
> // if (position_bias->empty()) {
>   *position_bias = compute_relative_bias(...);
> // }
> ```
> **However, this approach may lead to performance degradation in T5 and MT5 models.**

That last sentence is why the issue died. The fix was known; it was just unacceptable,
because commenting out that cache makes **every** T5 and mT5 model recompute the position
bias in every layer — a permanent slowdown for thousands of existing users, to support one
new architecture.

**Our implementation is a strictly better answer to that exact objection.** Instead of
disabling the cache globally, we detect per model which regime applies:

- T5/mT5 → pointers are aliased → **keep the shared buffer, zero change, zero slowdown**
- UMT5 → pointers differ → **per-layer path**

No new config flag, no file-format change, no `spec_revision` bump, and we **proved** T5 is
untouched: `model.bin` byte-identical (same sha256), parity 1.07e-06, C++ suite 196/197
identical to baseline.

So the honest framing for any upstream conversation is: *"#1478 identified the cause but the
only known fix regressed T5. Here's one that doesn't, with the T5 no-regression evidence."*

**Caveat:** an old closed issue is not a commitment to merge. And `CONTRIBUTING.md` steers
newcomers away from core C++ (`src/layers/`) and toward "documentation, examples, or Hugging
Face integrations". B-01 and B-02 sit squarely in the welcome category; UMT5's C++ half does
not. That's the main reason to lead with B-01/B-02.

---

## Why B-01 and B-02 are the stronger openings

|  | B-01 (mT5) | B-02 (vocab) | UMT5 |
|---|---|---|---|
| Language | Python only | Python only | Python **+ core C++** |
| Lines | ~10 | ~10 | ~99 |
| `CONTRIBUTING.md` category | ✅ HF integration | ✅ HF integration | ⚠️ core, perf-critical |
| Breaks something users have today | **yes** — mT5 is broken now | yes, for sentinel tasks | no, adds new support |
| Prior art to argue with | none | none | a closed issue + a rejected fix |
| Reproducible in one command | yes | yes | needs a C++ build |

B-01 in particular is a **regression in shipped code**: mT5 works on transformers 4.x and is
broken on 5.x, and `python/tests/requirements.txt` already pins `transformers==5.9.0.*`, so
CI is running the broken combination right now.

---

## Recommended sequence

1. **Reproduce both yourself** (`plans/onboarding/02-HANDS-ON.md`, Steps 4 and 5). Do not
   open anything upstream you haven't personally reproduced.
2. **File B-01 as an issue first**, with the four-line evidence block. Issues are cheap;
   a maintainer reply tells you whether a PR is welcome before you spend the effort.
3. **B-02 as a second issue** — or fold it into the first if a maintainer prefers.
4. Only then raise UMT5, referencing #1478 and leading with the no-regression evidence.

Detail: `B-01-mt5-rescaling.md`, `B-02-vocabulary-duplicates.md`.
**Ready-to-paste issue text: `ISSUE-DRAFT-B-01.md`, `ISSUE-DRAFT-B-02.md`.**

### Version check — resolved 2026-09-22

B-01's open question ("does this affect transformers 5.9.0, the version CI pins?") is
**closed: yes.** Checked the upstream source at both tags:

| file | v5.9.0 | v5.17.0 |
|---|---|---|
| `modeling_t5.py` | 1 rescale, guarded by `scale_decoder_outputs` | same |
| `modeling_mt5.py` | **0 rescale sites** | **0 rescale sites** |
| `modeling_umt5.py` | 1 rescale, guarded by `tie_word_embeddings` | same |

So the bug is present on the pinned CI version too, and the issue can state that.

> Both write-ups contain a proposed fix, but **neither is implemented**. That is deliberate —
> reproduce first, decide the fix second. B-01 especially has a real open question
> (older `transformers` versions) that should be settled before code is written.
