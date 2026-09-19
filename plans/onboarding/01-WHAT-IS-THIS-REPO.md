# CTranslate2, explained from zero

No ML background assumed. Read once, then do `02-HANDS-ON.md`.

---

## 1. The 60-second version

A **model** is a big pile of numbers (called **weights**) plus a recipe for what to do with
them. Researchers train those numbers on GPUs for weeks. That's **training**.

Afterwards you just want to *use* it — put text in, get text out. That's **inference**.

Training happens once. Inference happens millions of times. So inference is where speed and
memory matter, and the tools built for training (PyTorch, Hugging Face `transformers`) are
flexible but not especially fast.

**CTranslate2 is an inference-only engine.** It throws away everything needed for training
and re-implements just the forward pass in hand-tuned C++. Result: several times faster,
much less memory, no Python needed at runtime.

The trade: CTranslate2 must be taught each model architecture **by hand**. That's why your
UMT5 failed. Nobody had taught it UMT5 yet.

---

## 2. The words you need

**Token.** Models don't see letters or words — they see tokens, roughly word-pieces.
`"The house is wonderful"` → `['▁The', '▁house', '▁is', '▁wonderful']`. The `▁` marks a
space. The **tokenizer** does this split, and every token has an integer **id**.

**Vocabulary.** The token↔id lookup table. `t5-small` has 32128 entries; `umt5-small` has
256384 (it covers 100+ languages).

> This matters more than it sounds. If the model says "token 32099" and your vocabulary says
> that's a different word than the tokenizer meant, you get fluent nonsense. **That is a real
> bug in this repo right now** — bug B-02, which you'll reproduce yourself.

**Embedding.** Each token id becomes a vector (a list of ~512 numbers). `d_model` is how
long that vector is. This is the model's "meaning space".

**Layer.** The model transforms those vectors through a stack of identical blocks.
`t5-small` has 6; `umt5-small` has 8. Each layer has two parts:

- **Attention** — lets each token look at other tokens. "is" needs to know it belongs to
  "house". Attention is split into **heads** (6 or 8 parallel views), each of size `d_kv`.
- **Feed-forward (FFN)** — a small per-token computation. `d_ff` is its internal width.

**Encoder / decoder.** Two flavours of stack:
- **Encoder** reads the whole input at once, every token seeing every other. Good for
  *understanding*.
- **Decoder** writes output one token at a time, each token seeing only what came before
  (it can't peek at the future — that would be cheating).

T5, mT5 and **UMT5 are encoder-decoder**: encoder reads English, decoder writes French.
GPT-style models are decoder-only.

**Logits.** The decoder's raw score for every token in the vocabulary — 256384 numbers
saying "how good would each word be next". **Softmax** turns them into probabilities.
**Greedy decoding** just takes the highest every time. **Beam search** keeps the best few
running candidates.

**Quantization.** Storing weights as 8-bit integers instead of 32-bit floats. ~4× smaller,
faster, slightly less accurate. That's what `--quantization int8` does.

---

## 3. Relative attention bias — the thing this whole project is about

Attention alone has no notion of order: "dog bites man" and "man bites dog" would look the
same. Models need position information.

T5's approach: for each pair of positions, add a learned number to the attention score
depending on how far apart they are. Nearby words get one nudge, distant words another.
Those numbers live in a small table — for `umt5-small`, **32 distance buckets × 6 heads**.

The whole difference between T5 and UMT5 is *how many of these tables exist*:

| | tables |
|---|---|
| **T5 / mT5** | **one**, in layer 0, reused by every layer |
| **UMT5** | **one per layer** — 8 encoder + 8 decoder |

That's it. That one difference is why UMT5 needed a C++ change, not just a converter entry.
CTranslate2 computed the position nudge **once** and reused it for all layers — correct for
T5, wrong for UMT5, and silently so: it produced fluent-looking but incorrect output.

---

## 4. How this repo is laid out

Two halves that meet in the middle.

```
  Hugging Face model                  CTranslate2 model
  (PyTorch weights)                   (model.bin)
        │                                   │
        │   ── PYTHON: the converter ──►    │   ── C++: the runtime ──►  output
        │      python/ctranslate2/          │      src/
        │                                   │
        └────────── the "spec" is the contract between them ──────────┘
```

### The Python half — `python/ctranslate2/`

| Path | What it is |
|---|---|
| `converters/transformers.py` | **The big one (~4000 lines).** One "loader" class per architecture. Reads HF weights, writes them into a spec. **`UMT5Loader` lives here now.** |
| `specs/transformer_spec.py` | Describes the *shape* of a model: how many layers, pre-norm or post-norm, gated FFN or not. |
| `specs/attention_spec.py` | Shape of one attention block — including `relative_attention_bias`. |
| `specs/model_spec.py` | Serializes a spec to `model.bin`. Also does **aliasing** (see below). |

A **loader** is a translator between naming conventions. Hugging Face calls something
`encoder.block.0.layer.0.SelfAttention.q.weight`; CTranslate2 calls it
`encoder/layer_0/self_attention/linear_0/weight`. The loader maps one to the other.

Adding a new architecture = adding a loader. ~60 lines when the architecture is close to an
existing one.

### The C++ half — `src/` and `include/`

| Path | What it is |
|---|---|
| `src/models/model.cc` | Loads `model.bin` into memory. |
| `src/layers/transformer.cc` | The encoder and decoder stacks. **Your Phase-2 change is here.** |
| `src/layers/attention.cc` | Attention maths, including the position bias. |
| `src/ops/` | Primitive operations — matrix multiply, GELU, softmax. |
| `src/cpu/`, `src/cuda/` | Hardware-specific fast paths. |

### One clever detail worth knowing: **aliasing**

When saving, if two weights are byte-identical, CTranslate2 stores one copy and an **alias**
(`model_spec.py:169-189`). On load, both names point to the *same memory*
(`model.cc:279-283`).

You already saw this: T5 stores 2 bias tables + 10 aliases; UMT5 stores 16 with 0 aliases.

That's exactly what your fix keys off — if layer 0 and layer 5 resolve to the same pointer,
it's T5 (share it); different pointers means UMT5 (compute per layer). No new file format,
and every existing model behaves identically.

---

## 5. Why your UMT5 conversion failed

`converters/transformers.py` keeps a registry keyed by the Hugging Face config class name:

```python
config_name = config.__class__.__name__     # "UMT5Config"
loader = _MODEL_LOADERS.get(config_name)    # None — nobody registered it
if loader is None:
    raise ValueError("No conversion is registered for the model configuration %s" ...)
```

Nothing deep. A missing dictionary entry. But the *fix* wasn't shallow, because copying the
T5 loader would have thrown away 14 of UMT5's 16 bias tables without telling you.

---

## 6. What we changed

Three files, 99 lines, on branch `umt5/phase2-runtime`:

1. **`python/ctranslate2/converters/transformers.py`** (+59) — `UMT5Loader`: registers
   `UMT5Config`, keeps each layer's own bias table, fixes the vocabulary padding.
2. **`src/layers/transformer.cc`** (+34) — detect per-layer vs shared bias, and stop forcing
   the shared buffer when they differ.
3. **`include/ctranslate2/layers/transformer.h`** (+6) — one flag per stack.

See them with:

```bash
git diff master -- src include python
```

---

## 7. Git: why your folder "looked unchanged"

The work is on **branches**, not on `master`:

```
master                   ← untouched, original code
└─ umt5/phase0-baseline  ← planning docs only
   └─ umt5/phase1-converter  ← + the Python loader
      └─ umt5/phase2-runtime ← + the C++ fix   ← YOU ARE HERE
```

A branch is a parallel version of the whole project. Switching branches rewrites the files on
disk. If you `git checkout master`, `UMT5Loader` disappears; come back and it returns.

```bash
git branch                 # list branches, * marks current
git checkout master        # see the original
git checkout umt5/phase2-runtime   # back to our work
git log --oneline -5       # history
git diff master --stat     # what's different
```

Nothing is lost when files "vanish" — that is branches working as intended.

---

## 8. Next

`02-HANDS-ON.md` — convert a model yourself, run it, and reproduce both bugs with your own
eyes before trying to fix them.
