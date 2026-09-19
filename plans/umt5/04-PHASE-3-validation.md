# Phase 3 — Validation, tests, docs

**Branch:** `umt5/phase3-validation` (from `umt5/phase2-runtime`)
**Touches:** `plans/umt5/scripts/`, `python/tests/test_transformers.py`,
`docs/guides/transformers.md`, `CHANGELOG.md`.
**Must NOT touch:** `src/`, `include/`, `python/ctranslate2/` — if Phase 3 finds a bug,
you go **back** to Phase 1 or 2's branch and fix it there, then re-run its gate.

---

## 3.1 The parity harness — the real deliverable of this phase

"It generates sensible French" is not verification. Write
`plans/umt5/scripts/parity.py` that compares CTranslate2 against the Phase-0 HF artefacts
numerically.

### What it compares, in this order (fail fast, cheapest first)

| # | Comparison | How to get the CT2 side | Tolerance |
|---|---|---|---|
| P1 | Per-layer bias tables | `inspect_ct2_model.py` fingerprints vs `*_bias_<i>.npy` | **exact** (`array_equal`) |
| P2 | Encoder final hidden state | `ctranslate2.Encoder(...).forward_batch()` → `last_hidden_state` | see §3.2 |
| P3 | Decoder step-1 logits | `Translator.score_batch` / `Generator`, or `Translator` with `return_scores` | see §3.2 |
| P4 | Greedy 20 tokens, beam=1 | `Translator.translate_batch(beam_size=1)` | **exact token match** |
| P5 | Beam 4 output | `translate_batch(beam_size=4)` | sanity read, not asserted |

P1 is the one that catches a broken Phase 1. P2 is the one that catches a broken Phase 2 —
run it **with all layers** and, if it fails, bisect by truncating the encoder (see §3.4).

### Critical setup rules for a fair comparison

- CT2 model converted with **no quantization** (`--quantization` omitted), loaded with
  `compute_type="float32"`, `device="cpu"`, `inter_threads=1, intra_threads=1`.
- HF model in `torch.float32`, `.eval()`, `torch.no_grad()`.
- Same tokenizer object, same prompt string from Phase 0 §0.7.
- **Feed CT2 the exact token list HF produced.** Do not let each side tokenize
  independently — a tokenizer difference will masquerade as a numerical bug.
  Convert HF's `input_ids` → tokens via `tokenizer.convert_ids_to_tokens` and pass those
  strings to CT2, which works on token strings.
- Single sequence, no padding, in the first run. Batch/padding effects are a **separate**
  test (§3.5), not a confound in the primary one.

---

## 3.2 Tolerances — decide them, write them down, don't tune them to pass

State them in `parity.py` as named constants with a comment justifying each:

- **P1 (bias tables):** exact. These are copied, never computed. Any difference is a bug.
- **P2 (encoder hidden state):** `max_abs_diff` and `max_rel_diff`. Start with
  `atol=1e-4, rtol=1e-3` for float32 on CPU. Encoder hidden states in T5-family models have
  large magnitudes (RMSNorm without a final scale-down), so **report the max absolute diff
  alongside the max magnitude** and judge the ratio, not the raw number.
- **P3 (logits):** `atol=1e-3` on the logit vector, **plus** two stronger, scale-free checks
  that matter more than any epsilon:
  - `argmax` matches.
  - Spearman/Kendall agreement on the **top-20** token ranking matches exactly.
- **P4 (greedy tokens):** exact for 20 steps. Divergence at step *k* localises the bug to
  the decoder cache path rather than the forward pass.

> Honesty rule: if a tolerance has to be loosened to make a test pass, that is a **finding**,
> not a fix. Record the loosening, the number it needed, and why, in `06-DECISIONS.md`. Do
> not quietly bump `atol` until green.

---

## 3.3 The falsification test (run this before believing anything)

A parity harness that has never failed proves nothing. Prove it can fail:

1. Check out `umt5/phase1-converter` (converter fixed, runtime **not**).
2. Run `parity.py`. **P1 must PASS** (the tables are on disk correctly) and
   **P2 must FAIL** (the runtime shares layer 0's bias).
3. Check out `umt5/phase2-runtime`. **P2 must now PASS.**

Record both runs as evidence **E-14**. This single before/after pair is the strongest
evidence the project produces — it demonstrates the runtime change is load-bearing and that
the harness is sensitive to exactly the defect it was built for. Without it you cannot
distinguish "fixed" from "harness is blind".

---

## 3.4 If P2 fails on the Phase-2 branch — bisection procedure

Do not guess. In order:

1. **Is the change live?** Re-run Phase 0 checkpoint 0.3c and the E-12 log check. → **U-05**.
2. **Is it layer 0 only?** Build a 1-encoder-layer toy: load the HF model, truncate
   `model.encoder.block` to 1, convert, compare. If the 1-layer case passes and the full one
   fails, the bug is in *sharing*, i.e. Phase 2. If the 1-layer case also fails, the bug is
   upstream of position bias entirely — suspect FFN activation (`is_gated_act`), RMSNorm, or
   embedding scaling; check them one at a time against HF module outputs.
3. **Is it the bias at all?** Zero out **every** bias table in both HF and CT2 and re-run P2.
   If they now agree, the discrepancy is definitely the bias. If they still disagree, the
   bias is a red herring and you are chasing the wrong defect.
4. **Which layer diverges?** Use HF `output_hidden_states=True` for the per-layer reference,
   and truncate the CT2 encoder layer-by-layer (re-convert with `block[:k]`) to find the
   first `k` where they part. That `k` is your bug's address.
5. **Encoder-only vs decoder-only.** Feed CT2 the *HF* encoder output directly if the API
   allows, to isolate which stack is at fault.

Log every bisection step and its result. A bisection you didn't write down is one you will
repeat.

---

## 3.5 Secondary correctness checks (run after P1–P4 are green)

- **Batching / padding.** Translate `[prompt, short_prompt]` as a batch of 2 and compare each
  against its single-sequence result. Relative position bucketing interacts with the padder
  (`attention.cc:112-120` uses `query_length`/`key_length` from the *padded* tensors, and
  `Padder` removes padding before the layers). Any mismatch here is real and must be
  reported, not averaged away.
- **Incremental decoding beyond step 1** (**U-06**). P4's 20-step exactness covers it, but if
  it fails at exactly step 2, look at `attention.cc:244`'s `with_cache ? key_length - 1 : 0`.
- **Long input.** One input longer than `relative_attention_max_distance` (typically 128) so
  the log-bucketing branch at `attention.cc:83-94` is exercised. Compare against HF.
- **Quantization smoke test.** Convert with `--quantization int8` and confirm it loads,
  runs, and produces readable output. **Do not assert parity**; int8 parity is explicitly
  out of scope (`00-OVERVIEW.md` §7). Confirm only that
  `relative_attention_bias` is *not* quantized — `Model::is_quantizable`
  (`src/models/model.cc:289-291`) returns true only for names ending in `weight`, and our
  variable ends in `relative_attention_bias`. Verify the dtype after load. → **U-08**.
- **float16 smoke test** on CPU if supported by the build; otherwise note as untested.
- **CUDA**: out of scope on this machine. If the user has a CUDA box, the only extra check
  needed is that `Model::copy_to`'s alias preservation (`model.cc:812-829`) keeps the
  pointer-identity property after the device copy — i.e. re-run the E-12 detection log on GPU.
  **Until someone runs it, write "CUDA untested" in the docs and the PR.**

---

## 3.6 The unit test

Add to `python/tests/test_transformers.py`, in `_TRANSFORMERS_TRANSLATION_TESTS`
(the list ending at line ~128), following the existing tuple shape
`(model, source_tokens, target_tokens, expected_tokens, kwargs)`:

```python
    (
        "google/umt5-small",
        "<source tokens from the Phase 0 tokenizer dump>",
        "",
        "<expected target tokens, taken from the HF greedy reference>",
        dict(),
    ),
```

Rules:
- The expected tokens must come from the **HF reference** (Phase 0 artefacts), not from
  whatever CT2 currently emits. Copying CT2's own output into the expectation makes the test
  assert only that the code does what it does.
- Note the decorator on the test function: `@test_utils.only_on_linux`
  (`test_transformers.py:~131`). Your macOS run will **skip** it. Run the body manually and
  record the result; say plainly in the PR that CI is what will actually exercise it.
- `google/umt5-small` is ~1.5 GB (300M params). Check whether the existing test models are
  comparably sized before adding a download this big to CI; if it is out of line, say so and
  propose a smaller community UMT5 checkpoint, or mark it `@pytest.mark.slow`. → **D-08**.

---

## 3.7 Docs

`docs/guides/transformers.md`:
1. Supported-model list (around line 28-30, where `T5`, `T5Gemma`, `T5Gemma2` appear) — add
   `UMT5` in the list's existing alphabetical/grouped position.
2. Line 570 currently reads *"The variants T5v1.1, mT5, and FLAN-T5 are also supported."*
   Decide: extend that sentence, **or** add a separate `## UMT5` section after `## T5`
   (line 563-594). Recommend a **separate section**, because UMT5 needs a caveat the other
   variants do not:

```markdown
## UMT5

[UMT5](https://huggingface.co/docs/transformers/model_doc/umt5) is a multilingual T5 variant
pretrained with the UniMax sampling strategy. Unlike T5 and mT5, each UMT5 self-attention
layer has its own relative attention bias, which requires CTranslate2 >= <version>.

<conversion + translation example, mirroring the T5 section's structure>
```

3. Fill `<version>` with the release this actually lands in. If unknown at PR time, write
   the PR number and fix it before merge — do not leave a placeholder in merged docs.

`CHANGELOG.md`: add under the unreleased section, in the existing style. Two entries, because
they are two user-visible things:
- *New features*: "Support converting UMT5 models from Hugging Face Transformers"
- *Fixes / improvements*: "Support Transformer models where each layer has its own relative
  attention bias"

---

## 3.8 Lint and formatting

```bash
cd python
black --check ctranslate2/converters/transformers.py tests/test_transformers.py
isort --check ctranslate2/converters/transformers.py tests/test_transformers.py
flake8 ctranslate2/converters/transformers.py tests/test_transformers.py
```

For the C++ side, match surrounding style by eye: 2-space indent, `_member` naming,
`const` on members, brace style as in `transformer.cc`. There is no clang-format config in
the repo — do **not** introduce one.

---

## 3.9 GATE 3 — release readiness

- [ ] G3.1 P1–P4 all pass on `umt5/phase3-validation`, numbers recorded as **E-15**.
- [ ] G3.2 The falsification test (§3.3) recorded as **E-14**: P2 fails on Phase 1, passes on Phase 2.
- [ ] G3.3 t5-small, mt5-small, t5gemma outputs unchanged vs `master` (re-confirm E-13 on the
      merged branch, not just on Phase 2).
- [ ] G3.4 `./build-umt5/tests/ctranslate2_test ./tests/data` still matches E-01.
- [ ] G3.5 `pytest python/tests/` — same pass/skip/fail set as on `master`, plus the new test
      (skipped on macOS). Record both runs.
- [ ] G3.6 Batch/padding, long-input, and int8-smoke results recorded — including any that
      **failed**. A gate with no failures listed and no statement that none occurred is
      incomplete.
- [ ] G3.7 Docs + CHANGELOG written; every "untested" surface (CUDA, tensor-parallel, flash
      attention, int8 parity) named explicitly in the PR description.
- [ ] G3.8 Lint clean.

---

## 3.10 Upstreaming note — read `CONTRIBUTING.md` before opening a PR

`CONTRIBUTING.md:31-37` sets an explicit policy for this repository:

> *"Use of AI tools for brainstorming or minor assistance is acceptable, but contributors
> must explicitly disclose how AI was used and remain fully responsible for correctness,
> performance, and design. Submissions that appear generated without deep understanding will
> be declined."*
> *"Mandatory Deep Understanding: Contributors must fully understand their code and be
> prepared to justify the purpose of part of the code base."*

If this is going upstream to OpenNMT/CTranslate2:
- **Disclose** the AI involvement in the PR description. This is required, not optional.
- Be ready to defend, in your own words: why the shared `position_bias` buffer exists at all,
  why pointer identity is a sound discriminator, what `_alias_variables` does, and why no
  format version bump is needed. If you cannot, do not open the PR yet.
- Lead with the evidence, especially **E-12** (detection log) and **E-14** (falsification),
  not with the diff.

If this is staying local to the user's fork, none of that is required — but E-14 is still the
only thing that tells you the work is actually done.
