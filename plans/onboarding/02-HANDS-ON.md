# Hands-on: convert a model, run it, break it

Every command below was run on this machine and works. Do them in order.
Expect ~30 minutes the first time.

---

## Step 0 — the one line you need every time

Everything lives in a virtual environment in the repo, and the C++ library has to be
findable. Run this **once per terminal window**:

```bash
cd /Users/imrannazir/Documents/CTranslate2
export DYLD_LIBRARY_PATH="$PWD/install-umt5/lib:$DYLD_LIBRARY_PATH"
export PATH="$PWD/.venv-umt5/bin:$PATH"
```

Check it worked:

```bash
python -c "import ctranslate2; print(ctranslate2.__version__, ctranslate2.__file__)"
```

You must see a path **inside this repo**:
```
4.8.2 /Users/imrannazir/Documents/CTranslate2/python/ctranslate2/__init__.py
```

> If it says `site-packages` instead, you're running a downloaded copy, not the one we built —
> your changes will appear to do nothing. This is the single most common way to waste an
> afternoon. Check it every session.

Make sure you're on the right branch:

```bash
git branch --show-current     # expect: umt5/phase2-runtime
```

---

## Step 1 — convert your first model

`t5-small` is ~250 MB and downloads once.

```bash
ct2-transformers-converter --model t5-small --output_dir /tmp/my-first-model
ls -la /tmp/my-first-model
```

Three files:
- `model.bin` — all the weights in CTranslate2's format
- `shared_vocabulary.json` — the token list
- `config.json` — which token means "start", "end", "unknown"

**What just happened:** Python downloaded the PyTorch model, `T5Loader` walked every weight
and copied it into a spec under CTranslate2 names, and the spec was serialized to `model.bin`.

---

## Step 2 — run it

```bash
python - <<'PY'
import ctranslate2, transformers
tok = transformers.AutoTokenizer.from_pretrained("t5-small")
tr  = ctranslate2.Translator("/tmp/my-first-model", device="cpu")

text = "translate English to German: The house is wonderful."
tokens = tok.convert_ids_to_tokens(tok(text).input_ids)
print("input tokens :", tokens)

out = tr.translate_batch([tokens])[0].hypotheses[0]
print("output tokens:", out)
print("decoded      :", tok.decode(tok.convert_tokens_to_ids(out)))
PY
```

Expected: `Das Haus ist wunderbar.`

Note you pass **tokens**, not a string. CTranslate2 has no tokenizer — that's deliberate,
it's a pure inference engine. Tokenizing stays in Python.

---

## Step 3 — look inside the model file

```bash
python plans/umt5/scripts/read_model_bin.py /tmp/my-first-model relative_attention_bias
```

You'll see **2 real tables and 10 aliases** — layers 1-5 pointing at layer 0. That's the T5
sharing from the guide, visible on disk.

Now the same for UMT5:

```bash
python plans/umt5/scripts/read_model_bin.py /tmp/ct2-umt5 relative_attention_bias
```

**16 real tables, 0 aliases.** That difference is what the C++ fix detects.

> If `/tmp/ct2-umt5` is gone (a reboot clears `/tmp`), rebuild it — see Step 7.

---

## Step 4 — reproduce bug B-02 (vocabulary duplicates) yourself

```bash
python - <<'PY'
import json
from transformers import AutoTokenizer, AutoConfig
cfg = AutoConfig.from_pretrained("t5-small")
tok = AutoTokenizer.from_pretrained("t5-small", use_fast=False)
hf  = tok.get_vocab()
ct2 = json.load(open("/tmp/my-first-model/shared_vocabulary.json"))

print("model expects       :", cfg.vocab_size)
print("tokenizer provides  :", len(hf))
print("gap CT2 must pad    :", cfg.vocab_size - len(hf))
print("<extra_id_*> already in tokenizer:", sum(1 for k in hf if k.startswith("<extra_id_")))
print("DUPLICATES in CT2 vocabulary:", len(ct2) - len(set(ct2)))

idx = {t: i for i, t in enumerate(ct2)}
for t in ("<extra_id_0>", "<extra_id_5>"):
    print(f"  {t}: HuggingFace id {hf[t]}  ->  CTranslate2 index {idx[t]}   MISMATCH")
PY
```

You should see **28 duplicates** and `<extra_id_0>` at HF id 32099 but CT2 index **32100**.

**Why it happens:** `T5Loader.get_vocabulary` (`transformers.py:1257-1264`) pads the
vocabulary up to the model's size by appending `<extra_id_0>`, `<extra_id_1>`, … — but those
names are *already in the tokenizer*. So the padding slots get names that collide with real
tokens.

**Why nobody noticed:** ordinary translation never emits a sentinel token, so the wrong names
sit at the end of the list doing nothing. It breaks the moment you use T5 for span-infilling
(the fill-in-the-blank task T5 was actually pretrained on).

Read the buggy code:
```bash
sed -n '1257,1264p' python/ctranslate2/converters/transformers.py
```

---

## Step 5 — reproduce bug B-01 (mT5 rescaling) yourself

This one is dramatic. `google/mt5-small` is ~2.3 GB.

```bash
ct2-transformers-converter --model google/mt5-small --output_dir /tmp/ct2-mt5
python plans/umt5/scripts/parity.py google/mt5-small /tmp/ct2-mt5
```

Look at P2:

```
HF  log-probs: [-23.28 -38.27 -65.91 ...]
CT2 log-probs: [-12.03 -11.93 -12.86 ...]
max |diff| = 67.79
```

HF is confident; CTranslate2 is near-uniform (`log(1/250112) = -12.4` — i.e. "every one of
250,000 words is equally likely", which means it's learned nothing).

Now the same script on `t5-small`:

```bash
python plans/umt5/scripts/parity.py t5-small /tmp/my-first-model
```

`max |diff| = 1.07e-06`. **Same code, same script — t5 perfect, mt5 catastrophically wrong.**

Now see why:

```bash
python - <<'PY'
from transformers import AutoConfig
for m in ("t5-small", "google/mt5-small"):
    c = AutoConfig.from_pretrained(m)
    print(f"{m:18s} tie_word_embeddings={c.tie_word_embeddings}  "
          f"scale_decoder_outputs={getattr(c,'scale_decoder_outputs','ABSENT')}")
PY
```

```
t5-small           tie_word_embeddings=True  scale_decoder_outputs=True
google/mt5-small   tie_word_embeddings=True  scale_decoder_outputs=ABSENT
```

`transformers` v5 added a dedicated `scale_decoder_outputs` flag. T5 uses it; **mT5 doesn't
rescale at all any more.** But CTranslate2 still reads the old `tie_word_embeddings` proxy:

```bash
sed -n '1250,1254p' python/ctranslate2/converters/transformers.py
```

So CTranslate2 multiplies mT5's decoder output by `1/sqrt(512) = 0.044`, flattening the
logits into mush — while Hugging Face does nothing. Full write-up:
`plans/upstream/B-01-mt5-rescaling.md`.

---

## Step 6 — see the UMT5 fix doing its job

The strongest demonstration in the project: same model, same harness, only the C++ differs.

```bash
# WITHOUT the C++ fix (converter only)
git checkout umt5/phase1-converter
cmake --build build-umt5 -j8 && cmake --install build-umt5 --prefix "$PWD/install-umt5"
python plans/umt5/scripts/parity.py /tmp/umt5-small-local /tmp/ct2-umt5 | grep -A3 "P4"
```
→ CT2 produces **one token** then stops. Diverges immediately.

```bash
# WITH the fix
git checkout umt5/phase2-runtime
cmake --build build-umt5 -j8 && cmake --install build-umt5 --prefix "$PWD/install-umt5"
python plans/umt5/scripts/parity.py /tmp/umt5-small-local /tmp/ct2-umt5 | grep -A3 "P4"
```
→ **All 20 tokens match Hugging Face exactly.**

> Each rebuild takes ~1 minute (only changed files recompile).
> A test that has never failed proves nothing. This pair is why we know the fix is real.

---

## Step 7 — rebuilding the UMT5 test model (if `/tmp` was cleared)

Stock `google/umt5-*` needs one patch: Google published those configs without a
`model_type` key, so `AutoConfig` can't identify them.

```bash
python - <<'PY'
import json, os, shutil
from huggingface_hub import hf_hub_download
dst = "/tmp/umt5-small-local"; os.makedirs(dst, exist_ok=True)
for f in ("config.json","generation_config.json","special_tokens_map.json",
          "spiece.model","tokenizer.json","tokenizer_config.json","pytorch_model.bin"):
    src = hf_hub_download("google/umt5-small", f); tgt = os.path.join(dst, f)
    if os.path.lexists(tgt): os.remove(tgt)
    os.symlink(os.path.realpath(src), tgt) if f.endswith(".bin") else shutil.copy(src, tgt)
p = os.path.join(dst, "config.json"); c = json.load(open(p))
c["model_type"] = "umt5"; json.dump(c, open(p, "w"), indent=2)
print("ready:", dst)
PY

ct2-transformers-converter --model /tmp/umt5-small-local --output_dir /tmp/ct2-umt5
```

---

## Step 8 — check your own custom UMT5

```bash
python plans/umt5/scripts/diagnose_model.py /path/to/your/model
```

Add `--skip-weights` to check config and tokenizer only. Look for:

```
encoder: N self-attention bias tables
  all pairwise distinct: True
  >>> UMT5-STYLE (per-layer bias). Phase 2 (C++) REQUIRED.
```

If it says **T5-STYLE** instead, your model is structurally a T5 and doesn't need the C++
fix at all.

---

## Useful commands

```bash
git branch                          # where am I
git diff master -- src include python   # the actual code changes
git checkout <branch>               # switch versions
./build-umt5/tests/ctranslate2_test ./tests/data   # C++ tests (expect 196 passed, 1 skipped)
cmake --build build-umt5 -j8        # rebuild after editing C++
cmake --install build-umt5 --prefix "$PWD/install-umt5"   # publish the rebuild
```

Editing **Python** needs no rebuild. Editing **C++** needs both cmake commands.

---

## Suggested order of work

1. Steps 0-3 — get comfortable converting and running.
2. Step 4 — reproduce B-02. Smallest, clearest bug. Try fixing it yourself; the pattern is
   already in `UMT5Loader.get_vocabulary` on this branch.
3. Step 5 — reproduce B-01. Higher impact, needs a decision about older `transformers`.
4. Step 6 — understand why the UMT5 fix matters.
5. Then come back and we'll finish UMT5 (the unresolved score discrepancy, and the
   `model_type` fallback).

New to C++? Read `03-CPP-FOR-PYTHON-DEVS.md` — it maps every Python habit (venv, pip,
imports, PYTHONPATH) onto its C++ equivalent using this repo, and explains exactly why the
`DYLD_LIBRARY_PATH` line in Step 0 is needed.
