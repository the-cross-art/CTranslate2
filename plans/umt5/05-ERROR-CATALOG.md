# Error catalog

Three tiers:

- **K-xx — known-known.** We can predict the symptom *and* the fix. Environment, API,
  config-shape failures. Loud. Cheap.
- **U-xx — known-unknown.** We know the *question* but not the answer; it depends on the
  checkpoint, the `transformers` version, or a code path nobody has exercised. Each has a
  **probe** to run and a **decision rule**. Several of these are silent.
- **X-xx — unknown-unknown probes.** Not errors — deliberate experiments designed to surface
  failures we have not thought of. These are what stop the project from shipping something
  that merely *looks* correct.

The dangerous ones are **U-04, U-05, U-09, N-02** — all silent. A loud crash costs an hour;
a silent wrong bias costs a week and ships bad translations.

> **Status after measurement (2026-09-20, see `07-VERIFIED-FINDINGS.md`):**
> **U-01 RESOLVED** (premise proven). **K-10, K-11 CANNOT FIRE** (deleted).
> **K-12 CONFIRMED REAL** (84 duplicate tokens). **Two new entries: N-01, N-02.**

---

# K — Known-known errors

## K-01 · cmake configure fails: submodules missing
**Phase** 0 · **Loud**
**Symptom:** `add_subdirectory ... does not contain CMakeLists.txt` for `cpu_features`,
`cxxopts`, `spdlog`, `ruy`, or `googletest`.
**Cause:** verified on this machine — `git submodule status` shows all seven entries prefixed
`-` (not initialised). The repo was cloned without `--recursive`.
**Fix:** `git submodule update --init --recursive`
**Verify:** re-run `git submodule status`; no leading `-`.

## K-02 · cmake fails to find Intel MKL
**Phase** 0 · **Loud**
**Symptom:** `Could NOT find MKL` / missing `mkl.h`.
**Cause:** `CMakeLists.txt:10` sets `WITH_MKL=ON` by default; MKL is x86-only and this is
Apple Silicon.
**Fix:** `-DWITH_MKL=OFF -DWITH_ACCELERATE=ON -DWITH_RUY=ON`.
**Verify:** configure output lists Accelerate and Ruy as enabled backends.

## K-03 · `transformers.models.umt5` does not exist
**Phase** 0 · **Loud**
**Symptom:** `ModuleNotFoundError` / `KeyError: 'umt5'` from `AutoConfig`.
**Cause:** `transformers` older than 4.30 (UMT5 landed mid-2023).
**Fix:** `pip install -U "transformers>=4.39"`.
**Note:** if the user's own environment has an older version, **their** conversion will fail
differently from ours. Ask which version they used. Record it.

## K-04 · AppleClang / OpenMP link failure
**Phase** 0 · **Loud**
**Symptom:** `library not found for -lomp`, or `clang: error: unsupported option '-fopenmp'`.
**Fix:** `-DOPENMP_RUNTIME=NONE`. (Or `brew install libomp` + `-DOPENMP_RUNTIME=COMP`, but
single-threaded is better for parity work anyway — it removes reduction-order nondeterminism.)

## K-05 · C++17 / CMake too old
**Phase** 0 · **Loud**
**Symptom:** `CMake 3.15 or higher is required`, or C++17 syntax errors.
**Fix:** `brew install cmake`. Confirm `cmake --version` and the compiler std in the configure log.

## K-06 · `import ctranslate2` → `dlopen` cannot find `libctranslate2.dylib`
**Phase** 0 · **Loud**
**Symptom:** `ImportError: dlopen(...): Library not loaded: @rpath/libctranslate2.4.dylib`.
**Fix:** `export DYLD_LIBRARY_PATH="$PWD/install-umt5/lib:$DYLD_LIBRARY_PATH"` (macOS SIP
strips this from some shells — if it is ignored, use the `cmake --install` prefix route in
Phase 0 §0.4 and re-run `pip install -e .` so the rpath is baked in).
**Watch for:** the version number in the dylib name changing between rebuilds.

## K-07 · A PyPI wheel shadows the local build
**Phase** 0/2 · **SILENT — the expensive one**
**Symptom:** everything imports and runs, and your C++ change has no effect whatsoever.
**Cause:** a previously `pip install ctranslate2`'d wheel is earlier on `sys.path`.
**Detect:** `python -c "import ctranslate2; print(ctranslate2.__file__)"` — must print a path
**inside this repo**.
**Fix:** `pip uninstall -y ctranslate2` (repeat until "not installed"), then reinstall the
editable build.
**This check is mandatory at the start of Phase 1, 2, and 3.** See also U-05.

## K-08 · `ValueError: No conversion is registered for the model configuration UMT5Config`
**Phase** 0 (expected) / 1 (must be gone) · **Loud**
**Source:** `python/ctranslate2/converters/transformers.py:115-120`.
**Cause:** `_MODEL_LOADERS` has no `UMT5Config` key.
**Fix:** Phase 1.
**Note:** this is the user's reported error. Reproducing it verbatim is Phase 0 gate G0.6.

## K-09 · Wrong architecture name at load time
**Phase** 1 · **Loud**
**Symptom:** converter raises about the architecture, or `from_pretrained` picks the wrong class.
**Cause:** `architecture_name` must be the exact HF class name.
**Fix:** `"UMT5ForConditionalGeneration"`. Confirm with
`python -c "from transformers import UMT5ForConditionalGeneration"`.

## K-10 · ~~`num_decoder_layers is None`~~ — ❌ CANNOT FIRE
**RESOLVED by measurement.** `UMT5Config.__post_init__` (`configuration_umt5.py:64-68`)
defaults `num_decoder_layers` to `num_layers`, and `google/umt5-small` sets it to `8`
explicitly. No `get_model_spec` override needed. Kept for reference only.
**Phase** 1 · **Loud (but late)**
**Symptom:** `TypeError: 'NoneType' object cannot be interpreted as an integer` inside
`TransformerDecoderSpec.__init__`'s `range(num_layers)`.
**Cause:** `T5Loader.get_model_spec:1238` passes `config.num_decoder_layers` through unguarded.
**Detect:** Phase 0 evidence E-03.
**Fix:** Phase 1 §1.3.a.

## K-11 · ~~`KeyError` on `dense_act_fn`~~ — ❌ CANNOT FIRE
**RESOLVED by measurement.** `dense_act_fn == "gelu_new"`, which is in
`_SUPPORTED_ACTIVATIONS` (`transformers.py:33`) → `Activation.GELUTanh`; `is_gated_act` is
`True`. Kept for reference only.
**Phase** 1 · **Loud**
**Symptom:** `KeyError: 'gated-gelu'` (or similar) from
`_SUPPORTED_ACTIVATIONS[model.config.dense_act_fn]` at `transformers.py:1241`.
**Cause:** HF stores `feed_forward_proj="gated-gelu"` and derives `dense_act_fn="gelu_new"`;
if the checkpoint's config is hand-written, `dense_act_fn` may be absent or raw.
**Detect:** E-03.
**Fix:** add the mapping to `_SUPPORTED_ACTIVATIONS` (`transformers.py:30-38`), not a
loader-local hack.

## K-12 · Vocabulary padding adds duplicate `<extra_id_*>` tokens — 🔴 CONFIRMED REAL
**Phase** 1 · **SILENT — nothing raises**
**Measured:** `config.vocab_size 256384`, `len(tokenizer) 256300`, gap **84**, and the
tokenizer **already contains all 300** `<extra_id_*>` sentinels. The inherited loop appends
`<extra_id_0..83>` as duplicates. The vocabulary ends up the right *size*, so no exception —
but 84 names collide with the real sentinels. Latent for translation, a genuine bug for
span-infilling (UMT5's own pretraining task).
**Fix is mandatory:** Phase 1 §1.3 (pad with `<unused_%d>`, assert no duplicates).
**Regression test:** gate G1.5's `len(set(tokens)) == len(tokens)`.
**Symptom:** either a shape error when registering the vocabulary, or — worse — a converted
model whose token ids are offset from the tokenizer's, producing fluent gibberish.
**Cause:** `T5Loader.get_vocabulary:1257-1264` pads with `<extra_id_i>` assuming a T5-style
sentinel gap.
**Detect:** the diagnostic in Phase 1 §1.3.c (evidence E-06).
**Fix:** per the decision rule there. **Assert**
`len(tokens) == model.get_input_embeddings().weight.shape[0]`.

## K-13 · Tokenizer larger than the embedding matrix
**Phase** 1 · **Loud**
**Symptom:** negative `extra_ids`, or a vocabulary longer than the embedding rows.
**Cause:** a fine-tuned checkpoint with added tokens but an unresized embedding, or vice versa.
**Fix:** this is a broken checkpoint, not a CTranslate2 bug. Report it to the user with both
numbers; do not truncate silently.

## K-14 · `black`/`isort`/`flake8` failures
**Phase** 1/3 · **Loud** · Run them (`CONTRIBUTING.md:59-64`). Trivial.

---

# U — Known-unknowns: the probe list

## U-01 · Does every UMT5 layer actually own a bias? — ✅ RESOLVED: YES
**Was the load-bearing assumption. Now measured — see `07-VERIFIED-FINDINGS.md` F-01.**
Source: `UMT5LayerSelfAttention` passes `has_relative_attention_bias=True` unconditionally;
cross-attention gets `False`. Weights: `google/umt5-small` has 8 encoder + 8 decoder tables,
shape `(32, 6)`, **all pairwise distinct**. Negative control: `t5-small` and `mt5-small` have
exactly one each. **Phase 2 is required.** The "stop and re-scope" branch is closed.
Original text kept below for the probe method.
**Probe:** Phase 0 §0.5 Q1/Q2 (read `UMT5LayerSelfAttention.__init__` source) **and**
Phase 0 §0.7 (the per-layer identity matrix on the real checkpoint weights).
**Decision rule:**
- All layers differ → proceed as planned.
- All identical / only layer 0 present → **Phase 2 is unnecessary.** Stop, re-scope with the
  user, ship Phase 1 alone.
- Encoder differs, decoder does not → plan still works (detection is per-stack); update the
  Phase 3 expectations.
**Guard if it silently regresses later:** the explicit `raise ValueError` in `UMT5Loader.set_stack`
(Phase 1 §1.2) turns a future `transformers` rename into a converter-time error.

## U-02 · `-Wreorder` from member-initialisation order
**Phase** 2 · **Loud (warning) → SILENT (if ignored)**
**Symptom:** `warning: '_shared_position_bias' will be initialized after '_layers'`.
**Why it matters:** if `_shared_position_bias` is initialised **before** `_layers`, it reads
`_layers.size()` on an empty vector → always `true` → the fix silently does nothing.
**Rule:** declare after `_layers` in the header; treat any `-Wreorder` warning as a gate failure.

## U-03 · Does `MT5Loader` need the same treatment?
**Phase** 1 · **Probe:** run Phase 0's bias-identity script against `google/mt5-small`.
**Expected:** all identical → mT5 is genuinely T5-shaped → `MT5Loader` stays a 4-line subclass.
**If not:** mT5 has been silently mis-converted all along — a much bigger finding. Escalate
to the user before touching it; **do not** fold an mT5 fix into this project.

## U-04 · Does `Padder` interact with relative position bucketing? ⚠️ SILENT
**Phase** 3
**Why:** `attention.cc:112-120` derives buckets from `queries.dim(2)`/`keys.dim(2)`, but
`Padder` (`transformer.cc:447-451`) removes padding *before* the layers run, and
`compute_relative_bias` is computed once per stack in the shared path.
**Probe:** Phase 3 §3.5 — batch of two different-length sequences vs each alone.
**Decision rule:** if single and batched disagree beyond P2 tolerance, this is real. It may
well be **pre-existing** (T5 has the same structure) — check `t5-small` the same way before
attributing it to UMT5. Report either way.

## U-05 · The C++ change isn't actually loaded ⚠️ SILENT — most common time sink
**Phase** 2
**Symptom:** UMT5 output identical before and after the Phase 2 change (Gate G2.5 fails).
**Causes, in order of likelihood:** stale wheel (K-07); Python extension not rebuilt after
the dylib was; `DYLD_LIBRARY_PATH` pointing at an older install prefix; `build-umt5` vs a
second stray `build/` directory.
**Probe:** the `spdlog::warn` positive control in Phase 2 §2.5 (evidence E-12). Do this
**before** debugging numerics, every time.

## U-06 · Incremental decoding offset
**Phase** 3
**Why:** `attention.cc:244` passes `with_cache ? key_length - 1 : 0` as `query_offset`, and
on the per-layer path each layer now allocates its own buffer every step.
**Probe:** Phase 3 P4 — 20 greedy steps, exact token match.
**Decision rule:** divergence at exactly step 2 ⇒ cache/offset. Divergence at step 1 ⇒ the
forward pass, not the cache (look at P2/P3 instead).

## U-07 · Flash attention silently ignores relative attention bias
**Phase** 2 · **Probe:** read `src/layers/flash_attention.cc`; does it use its `position_bias`
parameter (`flash_attention.h:31`)?
**Context:** `WITH_FLASH_ATTN` is OFF by default (`CMakeLists.txt:25`) and `model.cc:198`
forces it off unless float16/bfloat16, so our builds never take this path.
**Decision rule:** if it ignores the bias, that is a **pre-existing T5 bug**, recorded as
D-07 and reported — **not** fixed in this project. Never claim UMT5 + flash attention works.

## U-08 · Quantization touching the bias table
**Phase** 3 · **Probe:** convert with `--quantization int8`, load, print the dtype of a
`relative_attention_bias` variable.
**Expected:** unquantized. `Model::is_quantizable` (`model.cc:289-291`) matches names ending
in `weight` only; ours ends in `relative_attention_bias`.
**If quantized anyway:** parity will degrade a lot. Report; do not add an exception without
understanding why the name matched.

## U-09 · Aliasing doesn't fire where we assume ⚠️ SILENT — Option R's single point of failure
**Phase** 1/2 · **Probe:** Phase 1 gate **G1.4 / evidence E-08** — confirm t5-small's layers
1..N *are* aliased to layer 0 and UMT5's are *not*.
**Decision rule:** if t5-small does **not** alias, Option R's detection returns `false` for
T5 and every T5 model switches to the per-layer path. Output stays *correct* (each layer
holds the same table) but you lose the shared-computation optimisation and **G2.4 will fail
on timing, not on values**. If this happens, switch to **Option F** (Phase 2 §2.6).
This is exactly why G1.4 exists before any C++ is written.

## U-10 · `dtype` conversion breaking pointer identity
**Phase** 2 · **Probe:** run the E-12 detection log with `compute_type="float16"` (and
`int8_float16` if the build supports it) on t5-small.
**Why:** `Model::set_compute_type` / `process_linear_weights` may convert variables in place
or replace them in `_variable_index`. If it replaces the aliased entry with a **fresh**
`StorageView` per name, pointer identity breaks and T5 falls to the per-layer path (correct
output, lost optimisation — same consequence as U-09).
**Decision rule:** if float32 detects `true` but float16 detects `false` for t5-small, Option
R is dtype-fragile → switch to Option F. **Check this before finalising Phase 2.**

## U-11 · The user's model is not stock `google/umt5-*`
**Phase** 0 · **Probe:** ask which model id/path, run the E-03 config dump and the E-05 bias
matrix against **their** model.
**Decision rule:** a fine-tune with a resized vocabulary triggers K-12/K-13. A merged or
quantized checkpoint may not be UMT5-shaped at all. Their model is the acceptance target;
`umt5-small` is only the development target.

## U-12 · Per-layer bias cost on long inputs
**Phase** 3 · **Probe:** time the encoder on a 512-token input, N layers, shared vs per-layer
(t5-small gives you both by construction).
**Decision rule:** `compute_relative_bias` is a `Gather` + `Transpose` over
`L_q × L_k × heads`. Going from 1× to N× is real but should be small next to N FFN passes. If
it is >10% end-to-end, record it in `06-DECISIONS.md` as a follow-up — **do not** optimise it
in this project (`00-OVERVIEW.md` §7).

---

# N — Found by measurement, not in the original plan

## N-01 · `AutoConfig` fails before the loader registry is consulted 🔴 BLOCKER
**Phase** 1 · **Loud, but misleading** · Full detail: `07-VERIFIED-FINDINGS.md` F-03
**Symptom:**
```
ValueError: Unrecognized model in google/umt5-small.
Should have a `model_type` key in its config.json.
```
**Cause:** stock `google/umt5-*` repos publish a `config.json` with **no `model_type` key**
(exported by `transformers 4.31.0.dev0`). `TransformersConverter._load` calls
`AutoConfig.from_pretrained` at `transformers.py:107`, which raises **before** the
`_MODEL_LOADERS.get(...)` lookup at line 113.
**Consequence:** registering `UMT5Config` in the registry does **not**, on its own, make
stock Google UMT5 checkpoints convertible.
**Two distinct user-visible errors now exist:**

| config.json | Error | Fixed by |
|---|---|---|
| no `model_type` | `Unrecognized model ... model_type key` | N-01 / Phase 1 §1.4 (W3) |
| `model_type: "umt5"` | `No conversion is registered ... UMT5Config` | Phase 1 §1.2 (W1) |

**Fix:** Phase 1 §1.4 — decision D-12 (docs-only / architecture fallback / hybrid).
**Workaround available today:** add `"model_type": "umt5"` to the model's `config.json`.
**Note:** `UMT5Config.from_pretrained(...)` works fine — only `AutoConfig` dispatch fails.

## N-02 · transformers forces `tie_word_embeddings = True` for UMT5 ⚠️ SILENT
**Phase** 1/3 · Full detail: `07-VERIFIED-FINDINGS.md` F-04
**Cause:** `configuration_umt5.py:75-76`:
```python
kwargs.pop("tie_word_embeddings", None)
self.tie_word_embeddings = True   # force it for T5 family
```
Unconditional — even `UMT5Config(tie_word_embeddings=False)` returns `True`. Meanwhile
`google/umt5-small`'s `config.json` says `false` and the checkpoint is genuinely untied
(`lm_head.weight` vs `shared.weight`, max abs diff **116.65**); HF warns and refuses to tie.
**Effect:** both `modeling_umt5.py:1052-1055` (HF) and `transformers.py:1252-1253` (CT2) read
the lying flag and apply `d_model**-0.5` to the decoder output.
**Why it's dangerous:** they agree, so **the parity harness will be green and will not flag
this.** Greedy `argmax` is invariant under a positive scalar, so no garbage text either — it
shifts scores, softmax sharpness, sampling, and beam ranking with length penalty.
**Also:** conversion output becomes **transformers-version-dependent**. A 4.x conversion that
honoured `false` yields a model *without* `scale_outputs`; ours has it.
**Decision:** D-11 — recommended is to match HF and document loudly, but decide consciously.
**Probe before Phase 1:** re-run the three-line check under `transformers==5.9.0.*` (what repo
CI pins, F-06). If 5.9.0 differs from 5.17.0, you have a reproducibility problem first.

## N-03 · Version drift from the repo's own CI 🟡
**Phase** 0 · `python/tests/requirements.txt` pins `transformers==5.9.0.*` and `torch==2.12`;
we installed **5.17.0 / 2.14.0**. Same major version and CT2 4.8.2 is already a
transformers-v5 codebase (it uses the v5 `dtype=` kwarg at `transformers.py:134-140`), so the
APIs are right. But N-02's behaviour must be confirmed on 5.9.0, and the Phase 3 test
expectation must be generated under whatever CI runs.

---

# X — Unknown-unknown probes

Deliberate attempts to break our own work. Run **all** of them in Phase 3 and record the
results, including the boring ones.

## X-01 · The falsification test
Phase 3 §3.3. Confirm the harness FAILS on the Phase-1 branch and PASSES on Phase-2.
**If it passes on both, the harness is blind and every green result so far is meaningless.**
This is the most important single check in the project.

## X-02 · Negative control on a non-T5 model
Convert and run `facebook/bart-base` (no relative attention bias at all) on the Phase-2
branch. Output must be character-identical to `master`. Proves the `!first → return true`
early exit keeps unrelated architectures on the old path.

## X-03 · One-layer model
Truncate a UMT5 to a single encoder and single decoder layer, convert, run. Exercises the
`num_layers < 2` branch of the detection helper. Must not crash; must match HF.

## X-04 · Shuffle the bias tables
Take the converted UMT5 spec, **swap** layer 0's and layer 5's bias tables, convert, run.
Output **must change**. If it does not, per-layer bias is still not being used and you have
a false green. (A nastier variant of X-01, and it catches failures X-01 can miss.)

## X-05 · Make two layers identical
Force layers 3 and 4 to share one table, convert, run. Aliasing will collapse them; detection
must still return per-layer (because other layers differ); output must still match an HF model
patched the same way. Probes the aliasing edge case from `00-OVERVIEW.md` §4.

## X-06 · Empty and single-token input
Translate `""` and a one-token input. `query_length == key_length == 1` exercises the
bucketing boundary at `attention.cc:83-94`. Compare against HF; watch for an exception from
`compute_relative_bias` on a degenerate shape.

## X-07 · Input longer than `relative_attention_max_distance`
Feed >128 tokens so the logarithmic bucket branch (`attention.cc:86-92`) runs, in both the
encoder (bidirectional) and the decoder (unidirectional). Compare against HF. Different code
path from every short-input test above.

## X-08 · Round-trip a converted model through `copy_to`
Load on CPU, and if any second device is available, copy and re-run. `Model::copy_to`
(`model.cc:809-831`) is the one place pointer identity is explicitly maintained; if that
guarantee ever breaks, this is where it shows. On a CPU-only box, at least load the same
model twice in one process and confirm identical output.

## X-09 · Two models in one process
Load t5-small and umt5-small simultaneously and translate with both, interleaved. Catches any
accidental static/global state in the detection or bias path.

## X-10 · Re-convert and diff
Convert the same UMT5 checkpoint twice into different directories and `shasum` both
`model.bin`. They must be identical. A difference means nondeterminism in conversion (dict
ordering, aliasing order) and undermines every reproducibility claim above.
