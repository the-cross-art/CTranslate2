# Phase 0 — Baseline: build it, break it, capture ground truth

**Branch:** `umt5/phase0-baseline` (from `master` @ `d44d2d06`)
**Touches:** `plans/umt5/scripts/` only. **No change to `src/`, `include/`, or `python/ctranslate2/`.**
**Goal:** end Phase 0 able to (a) build and import CTranslate2 from this working tree,
(b) reproduce the user's exact error, (c) hold a saved HF reference forward pass to compare
against later. Without (c) every later phase is guesswork.

---

## 0.1 Facts established about this machine (already checked)

- Platform: macOS (Darwin 25.5.0), Apple Silicon (`/opt/homebrew`).
- Default interpreter: `/opt/homebrew/Caskroom/miniforge/base/bin/python` (miniforge base).
- `ctranslate2` is **not** importable. `transformers` is **not** installed. `torch` is not installed.
- **Git submodules are NOT initialised.** `git submodule status` shows all seven entries
  prefixed with `-` (`cpu_features`, `cutlass`, `cxxopts`, `googletest`, `ruy`, `spdlog`, `thrust`).
  A cmake configure will fail until this is fixed. → error **K-01** in `05-ERROR-CATALOG.md`.
- `CMakeLists.txt:10` defaults `WITH_MKL=ON`. **MKL does not exist for Apple Silicon.**
  → error **K-02**.

Conclusion: the user has never built this tree. Phase 0 is mostly environment work. Budget
for it honestly; do not let the agent blast through it.

---

## 0.2 Step 0.1 — isolated Python environment

Do **not** install into miniforge base.

```bash
cd /Users/imrannazir/Documents/CTranslate2
python -m venv .venv-umt5
source .venv-umt5/bin/activate
python -m pip install -U pip wheel setuptools
```

Add `.venv-umt5/` to `.git/info/exclude` (**not** to `.gitignore` — do not dirty the repo's
tracked ignore file for a local venv):

```bash
echo ".venv-umt5/" >> .git/info/exclude
echo "plans/umt5/artifacts/" >> .git/info/exclude
```

**Checkpoint 0.1:** `which python` prints a path under `.venv-umt5/bin`.

---

## 0.3 Step 0.2 — install the reference stack (torch + transformers)

```bash
pip install "torch" "transformers>=4.39" "sentencepiece" "protobuf" "numpy<2.3" "pytest" "accelerate"
python - <<'PY'
import torch, transformers, sys
print("python     ", sys.version.split()[0])
print("torch      ", torch.__version__)
print("transformers", transformers.__version__)
from transformers.models.umt5 import modeling_umt5
print("umt5 module ok:", modeling_umt5.__file__)
PY
```

**Record all four version strings into `06-DECISIONS.md` § Evidence E-00.** Every later
"it worked on my machine" claim is meaningless without them.

If `from transformers.models.umt5 import modeling_umt5` raises → **K-03**.

**Checkpoint 0.2:** versions printed and recorded.

---

## 0.4 Step 0.3 — build CTranslate2 from this tree

```bash
cd /Users/imrannazir/Documents/CTranslate2
git submodule update --init --recursive        # fixes K-01

cmake -S . -B build-umt5 \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_MKL=OFF \
  -DWITH_ACCELERATE=ON \
  -DWITH_RUY=ON \
  -DBUILD_CLI=OFF \
  -DBUILD_TESTS=ON \
  -DOPENMP_RUNTIME=NONE

cmake --build build-umt5 -j"$(sysctl -n hw.ncpu)"
```

Notes for the implementing agent:
- `-DWITH_MKL=OFF -DWITH_ACCELERATE=ON -DWITH_RUY=ON` is the Apple Silicon combination.
  Accelerate supplies float32 GEMM; Ruy supplies the int8 path.
- `-DOPENMP_RUNTIME=NONE` avoids the libomp-vs-AppleClang mess. If you *have* `brew install libomp`
  and want threads, use `-DOPENMP_RUNTIME=COMP` and expect to debug it. Phase 0–3 gates do not
  need OpenMP; single-threaded parity checks are fine and are in fact *more* deterministic.
- `-DBUILD_TESTS=ON` gives you `build-umt5/tests/ctranslate2_test`, needed in Phase 2.
- **Do not** `sudo make install`. We install the Python wrapper against the build tree
  (next step) so that rebuilds are cheap and nothing pollutes `/usr/local`.

Expected failure modes here: **K-01, K-02, K-04, K-05** — see `05-ERROR-CATALOG.md`.

**Checkpoint 0.3a:** `ls build-umt5/libctranslate2*.dylib` succeeds.
**Checkpoint 0.3b:** `./build-umt5/tests/ctranslate2_test ./tests/data` passes on a clean
`master` build. **Record the pass/fail counts as evidence E-01** — this is the regression
baseline for Phase 2. If it already fails on `master`, that is a pre-existing condition:
record which tests fail and treat that exact set as the allowed baseline.

### Build the Python wrapper against the build tree

```bash
cd python
pip install -r install_requirements.txt
CTRANSLATE2_ROOT=/Users/imrannazir/Documents/CTranslate2/build-umt5 \
  pip install -e . --no-build-isolation
cd ..
```

If `CTRANSLATE2_ROOT` pointing at the build tree is not accepted by `setup.py` (it may expect
an *install* prefix with `include/` and `lib/`), fall back to a local install prefix:

```bash
cmake --install build-umt5 --prefix "$PWD/install-umt5"
cd python
CTRANSLATE2_ROOT="$PWD/../install-umt5" pip install -e . --no-build-isolation
cd ..
export DYLD_LIBRARY_PATH="$PWD/install-umt5/lib:$DYLD_LIBRARY_PATH"
```

→ error **K-06** covers the `dlopen`/`DYLD_LIBRARY_PATH` failure.

**Checkpoint 0.3c:**
```bash
python -c "import ctranslate2; print(ctranslate2.__version__, ctranslate2.__file__)"
```
prints a version and a path **inside this repo** (not site-packages of a pip wheel). If it
prints a site-packages path, you have a stale wheel shadowing the build → **K-07**.

> ⚠️ This is the single most important check in Phase 0. If you are silently testing against
> a PyPI wheel, Phase 2's C++ change will appear to do nothing and you will waste hours.
> Re-run this check at the start of **every** later phase.

**Write down the rebuild incantation** you will use after every C++ edit:
```bash
cmake --build build-umt5 -j"$(sysctl -n hw.ncpu)" && \
  python -c "import ctranslate2; print('reimport ok')"
```
(With an editable install against the build tree, rebuilding the dylib is usually enough;
if the pybind extension itself changed you must re-run `pip install -e .`.)

---

## 0.5 Step 0.4 — VERIFY-1: read the actual UMT5 source

**This step is non-negotiable and comes before any code is written.**

```bash
python - <<'PY'
import inspect
from transformers.models.umt5 import modeling_umt5 as m
for name in ("UMT5Attention", "UMT5LayerSelfAttention", "UMT5LayerCrossAttention",
             "UMT5LayerFF", "UMT5Block", "UMT5Stack"):
    cls = getattr(m, name, None)
    print("="*70); print(name, "->", "MISSING" if cls is None else "")
    if cls is not None:
        print(inspect.getsource(cls.__init__))
PY
```

Answer and record in `06-DECISIONS.md` § Evidence **E-02**:

| # | Question | Where it matters |
|---|---|---|
| Q1 | Does `UMT5LayerSelfAttention.__init__` pass `has_relative_attention_bias=True` unconditionally? | the entire premise |
| Q2 | Does `UMT5Attention` expose an attribute literally named `has_relative_attention_bias`? | `T5Loader.set_attention:1345` reads it |
| Q3 | Does cross-attention (`UMT5LayerCrossAttention`) have a bias? (expected: no) | Phase 1 §1.4 |
| Q4 | Are the submodule attribute names identical to T5 — `.SelfAttention`, `.EncDecAttention`, `.DenseReluDense.wi_0/.wi_1/.wo`, `.layer_norm`, `.q/.k/.v/.o`, `.final_layer_norm`, `.embed_tokens`, `.block`, `.layer`? | Phase 1 §1.3 |
| Q5 | Is there any query scaling in `UMT5Attention` that T5 lacks (T5 uses none; CT2 sets `queries_scale = 1.0`)? | Phase 1 §1.5 |
| Q6 | Is `relative_attention_max_distance` per-layer or per-config? | Phase 1 §1.4 |
| Q7 | Does `UMT5Attention` use `layer_idx` / new-style cache in this `transformers` version? | only affects the reference script, not CT2 |

Then dump the config of the actual target model:

```bash
python - <<'PY'
from transformers import AutoConfig
c = AutoConfig.from_pretrained("google/umt5-small")
print(type(c).__name__)
for k in ("d_model","num_layers","num_decoder_layers","num_heads","d_kv","d_ff",
          "feed_forward_proj","dense_act_fn","is_gated_act","vocab_size",
          "relative_attention_num_buckets","relative_attention_max_distance",
          "tie_word_embeddings","decoder_start_token_id","pad_token_id","eos_token_id",
          "layer_norm_epsilon","dropout_rate"):
    print(f"{k:35s} {getattr(c, k, '<absent>')!r}")
PY
```

Record as evidence **E-03**. Specifically flag:
- `num_decoder_layers is None` → **K-10** (T5Loader passes it straight into `from_config`).
- `dense_act_fn` not in `_SUPPORTED_ACTIVATIONS` (`transformers.py:30-38`) → **K-11**.
- `tie_word_embeddings` value — decides whether `scale_outputs` is set (`transformers.py:1252-1253`).

> If the user's model is **not** `google/umt5-small` — e.g. a fine-tune, or `umt5-base/xl/xxl`,
> or a third-party UMT5 — run this block against **their** model id too and record both.
> Use `google/umt5-small` as the development target regardless: it is the smallest and every
> gate runs in seconds on CPU.

---

## 0.6 Step 0.5 — reproduce the reported error, verbatim

```bash
ct2-transformers-converter --model google/umt5-small \
  --output_dir /tmp/ct2-umt5-should-fail
```

Expected, from `transformers.py:115-120`:

```
ValueError: No conversion is registered for the model configuration UMT5Config
(supported configurations are: ..., T5Config, ..., MT5Config, ...)
```

**Paste the full message into `06-DECISIONS.md` § Evidence E-04.** If the error text differs
from this, something about the user's setup is different from ours and Phase 1's premise
must be re-checked before proceeding.

---

## 0.7 Step 0.6 — capture the HF ground truth (the most valuable artefact of this phase)

Create `plans/umt5/scripts/hf_reference.py`. It must:

1. Load `UMT5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32)`, `.eval()`.
2. Encode one fixed prompt with the matching tokenizer (`use_fast=False` to pin the
   sentencepiece path; record which one you used).
3. Run a forward pass with `decoder_input_ids = [[decoder_start_token_id]]` and
   `output_hidden_states=True`.
4. Save, as `.npy` under `plans/umt5/artifacts/`:
   - `input_ids.npy`, `decoder_input_ids.npy`
   - `encoder_last_hidden_state.npy`
   - `logits.npy` (first decoder step)
   - **`enc_bias_<i>.npy` for every encoder layer** — the raw
     `model.encoder.block[i].layer[0].SelfAttention.relative_attention_bias.weight`
   - **`dec_bias_<i>.npy` for every decoder layer** — same for the decoder
5. Print, for the biases: `shape`, and a pairwise "is layer i identical to layer 0"
   boolean matrix.
6. Also save `greedy_20.npy` / a text file with a 20-token greedy decode of the same prompt,
   `num_beams=1, do_sample=False`, for the end-to-end gate in Phase 3.

Fixed prompt for all phases (do not change it later — it invalidates the artefacts):
```
"Translate English to French: The house is wonderful."
```

**Checkpoint 0.6 / Evidence E-05 — the decisive observation:**

The printed identity matrix must show **all layers differ from layer 0**. That is the
empirical proof of §3.1 in `00-OVERVIEW.md`, measured on the real checkpoint rather than
assumed.

- If all encoder layers differ → the premise holds. Proceed.
- If only layer 0 has a bias / all are identical → **STOP.** The model is structurally T5
  and the whole Phase 2 is unnecessary; Phase 1 alone (registering the loader) would suffice.
  Log this in `06-DECISIONS.md` and re-scope with the user before writing any C++.
- If encoder differs but decoder does not (or vice versa) → log it; Phase 2 detection is
  per-stack anyway, so it handles this correctly, but the test expectations change.

Also run the same script against `t5-small` and confirm the identity matrix is the opposite
(all layers identical / only layer 0 present). That is your **negative control** and proves
the detection logic you will write in Phase 2 can actually discriminate.

---

## 0.8 GATE 0 — do not open Phase 1 until all of these are true

- [ ] G0.1 `python -c "import ctranslate2"` resolves to a module built from **this** tree.
- [ ] G0.2 `./build-umt5/tests/ctranslate2_test ./tests/data` result recorded as E-01.
- [ ] G0.3 `transformers` / `torch` / Python versions recorded as E-00.
- [ ] G0.4 VERIFY-1 questions Q1–Q7 answered from source, recorded as E-02.
- [ ] G0.5 Target model config dumped, recorded as E-03; K-10/K-11 checked.
- [ ] G0.6 The `ValueError` reproduced verbatim, recorded as E-04.
- [ ] G0.7 `plans/umt5/artifacts/` holds the HF reference tensors, and the per-layer bias
      identity matrix for UMT5 (all-differ) **and** for t5-small (all-same) is recorded as E-05.
- [ ] G0.8 Commit on `umt5/phase0-baseline`: scripts only. `git diff master --stat` must show
      **zero** files under `src/`, `include/`, or `python/ctranslate2/`.

Paste the evidence, then open `02-PHASE-1-converter.md`.
