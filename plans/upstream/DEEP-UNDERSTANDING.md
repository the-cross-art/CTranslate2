# The three things you must be able to explain unaided

These are the questions a maintainer is most likely to ask, because each one is a place where
the change could have been done wrong. Read until you could answer from memory.

---

## 1. Why does the shared `position_bias` buffer exist?

### What the bias actually is

Attention computes a score for every (query position, key position) pair. T5-family models
add a learned number to each score based on the **distance** between the two positions.

`compute_relative_bias` (`src/layers/attention.cc:102-126`) builds that as a tensor of shape
`[num_heads, query_length, key_length]` in three steps:

1. **`get_relative_position_bucket`** — for every (i, j) pair, compute `j - i`, then map it
   into one of 32 buckets. Near distances get their own bucket; far ones are grouped
   logarithmically (`attention.cc:83-94`, with a `std::log` per element).
2. **`ops::Gather`** — look up each bucket in the `[32, num_heads]` table, producing
   `[query_length, key_length, num_heads]`.
3. **`ops::Transpose({2,0,1})`** — reorder to `[num_heads, query_length, key_length]`.

### Why caching it is a real optimization

The result depends on exactly five things: the **bias table**, `query_length`, `key_length`,
`is_decoder`, and the offset. For T5 **all layers share one table**, so all five inputs are
identical for every layer — meaning **the output tensor is bit-for-bit identical in every
layer.**

Computing it 24 times to get 24 identical answers is pure waste. Concretely, for a
512-token input with 8 heads:

- 262,144 bucket computations (each with a branch and a `log`)
- a gather producing 2.1M floats
- a transpose of those 2.1M floats
- a 2.1M-float allocation, then freed

Per layer. On a 24-layer encoder that's 24× all of it. That's why
`TransformerEncoder::operator()` declares **one** `StorageView position_bias` and threads a
pointer to it through every layer, and why `attention.cc:237` guards with
`if (position_bias->empty())` — only layer 0 finds it empty and fills it; layers 1..N find it
populated and skip straight to the add.

### Why it's wrong for UMT5

The cache is valid **only because the inputs are identical**. UMT5 breaks that premise: every
layer has its own table, so the correct bias tensor differs per layer. The cache silently
hands layer 0's answer to all 24 layers.

Not a crash. Not a warning. Just the wrong positional prior in 23 of 24 layers.

### What the fix does

`attention.cc:233-235` already has the per-layer path:

```cpp
StorageView local_position_bias(output.dtype(), output.device());
if (!position_bias)
  position_bias = &local_position_bias;   // a fresh, always-empty buffer
```

Pass `nullptr` and each layer gets its own local buffer, which is always empty, so each
computes from its own table. **We added no new computation — we made an existing branch
reachable.**

And the cost for UMT5 is inherent, not a regression: Hugging Face computes the bias per layer
too, because it has to.

> **If asked "why not just delete the `if`?"** — that's what #1478 proposed. It would force
> every T5 and mT5 model to redo all of the above N times for N identical answers. The
> detection exists precisely so T5 keeps the cache.

---

## 2. What does `_alias_variables` do, and why is pointer comparison sound?

### On write (`python/ctranslate2/specs/model_spec.py:169-189`)

Before saving, CTranslate2 walks every variable in sorted order and compares each against the
ones before it. If two are element-wise equal, the later one's **value is replaced by the
earlier one's name — a string**:

```python
if not value.is_scalar() and value.equal(other_value) and attr_name not in SKIP_CREATING_ALIAS:
    spec = index_spec(self, scope)
    setattr(spec, attr_name, other_name)   # now a str, not a tensor
```

`_serialize` (`model_spec.py:382-413`) then splits on type: anything still a tensor goes in
the variables section; anything now a string goes in an **alias table** written at the end as
a count followed by (alias_name, target_name) pairs.

This is deduplication. It's why `t5-small` stores 2 bias tensors plus 10 aliases while
`google/umt5-small` stores 16 tensors and 0 aliases — you can see both with
`plans/umt5/scripts/read_model_bin.py`.

### On read (`src/models/model.cc:274-283`)

```cpp
void Model::register_variable(std::string name, StorageView variable) {
  _variable_index.emplace(std::move(name), std::make_shared<StorageView>(std::move(variable)));
}

void Model::register_variable_alias(std::string alias, const std::string& variable_name) {
  auto it = _variable_index.find(variable_name);
  if (it == _variable_index.end()) return;
  _variable_index.emplace(std::move(alias), it->second);   // ← the SAME shared_ptr
}
```

`_variable_index` is `unordered_map<string, shared_ptr<StorageView>>`. An alias inserts a
second **name** pointing at the **same object**. So for T5,
`get_variable_if_exists("encoder/layer_0/.../relative_attention_bias")` and
`...layer_5/...` return the **identical raw pointer**. For UMT5 they return different ones.

`Model::copy_to` (`model.cc:809-831`) deliberately preserves this, using a
`unordered_map<const StorageView*, shared_ptr<StorageView>>` so that two names that shared an
object before a device copy still share one after. **That's why this works on GPU** — and
your L4 run is empirical confirmation.

### The soundness argument — this is the part to have ready

Enumerate every case. In each, ask: is the decision correct?

| Situation | Aliased? | Pointers | We decide | Correct? |
|---|---|---|---|---|
| T5: one table, layers 1..N alias layer 0 | yes | same | **shared** | ✅ identical inputs ⇒ identical bias ⇒ cache is valid |
| UMT5: all tables distinct | no | differ | **per-layer** | ✅ each computes its own |
| No relative attention bias at all (BART, NLLB, GPT…) | n/a | `first == nullptr` | **shared** (early return) | ✅ buffer is never filled; identical to old behaviour |
| Fewer than 2 layers | n/a | loop never runs | **shared** | ✅ nothing to share *with* |
| UMT5 where layers 3 and 4 happen to be byte-identical, others differ | those two alias | some differ | **per-layer** | ✅ the two identical ones each compute the same correct bias |
| UMT5 where *all* tables are byte-identical | all alias | same | **shared** | ✅ numerically indistinguishable from T5 |

**In every case the decision is correct.** There is no input for which pointer comparison
produces wrong output. That's the soundness claim, and it's the strongest thing you can say
about the design.

### The honest weakness

It depends on the aliasing machinery behaving as it does today. If a future change made
aliasing lossy, or made dtype conversion replace each name's `StorageView` independently,
then T5's pointers would stop matching and T5 would fall to the per-layer path.

**Note what that failure looks like: slower, not wrong.** The per-layer path computes the
same bias from the same shared table. So the worst case of the mechanism degrading is a
performance regression, never a correctness one. Say this if challenged — it's a genuine
design virtue, not a dodge.

The alternative (an explicit spec flag) doesn't depend on aliasing, but does touch the spec
and requires the converter to set it. Either is defensible; this one has a smaller blast
radius.

---

## 3. Why would a `spec_revision` bump be actively harmful?

### The version handshake

Every model file carries two version numbers (`model_spec.py:399-401`):

```python
model.write(struct.pack("I", CURRENT_BINARY_VERSION))   # = 6, the file format
_write_string(self.name)                                # e.g. "TransformerSpec"
model.write(struct.pack("I", self.revision))            # = 7, this spec's layout
```

On load, `model.cc:596-614` checks both against what the binary supports, via
`check_version` (`model.cc:460-472`):

```cpp
if (saved_version > current_version)
  throw std::runtime_error("Unsupported model " + version_type + ". This executable supports "
     "models with " + version_type + " v" + std::to_string(current_version) + " or below, "
     "but the model has ... (Forward compatibility is not guaranteed.)");
```

**A model whose revision exceeds the binary's is refused outright.** Not degraded — refused.

### Why bumping it is disproportionate

`revision` is a property of the **spec class**, not of an individual model
(`transformer_spec.py:585`). `TransformerSpec` is the shared seq2seq spec used by **T5, mT5,
FLAN-T5, NLLB, BART, mBART, Marian/OPUS-MT, M2M-100, Pegasus** — and UMT5.

So bumping `TransformerSpec.revision` from 7 to 8 stamps **8 on every model any of those
loaders produces.** Convert a plain `t5-small` with the new converter and it becomes
unloadable on every CTranslate2 build ≤ 4.8.2 — including the PyPI wheel everyone in
production is running.

That is an enormous blast radius for adding one architecture, imposed on users who never
touch UMT5.

### The honest cost of *not* bumping

State this before you're asked, because it's the real tradeoff:

**An old runtime loading a new UMT5 model will silently use layer 0's bias for all layers.**
No error. Wrong output.

Three things make that acceptable:

1. Old CTranslate2 **cannot convert UMT5 at all** — that's the error in #1478 — so this only
   arises if someone converts with a new version and deploys on an old binary.
2. **An explicit spec flag has exactly the same exposure.** An old binary reading an unknown
   attribute falls back to its default (shared). So bumping the revision is the *only* way to
   make old binaries hard-refuse — and it refuses everything else too.
3. There is **no per-model mechanism** to make old binaries reject only UMT5. `spec_revision`
   is per-spec-class by construction.

So the choice is: break every newly-converted T5/NLLB/BART model on older runtimes, or accept
a silent-wrongness window for an architecture those runtimes couldn't produce in the first
place. Every prior architecture addition in this repo made the same call.

### The one-sentence version

> *A `spec_revision` bump is per-spec-class, and `TransformerSpec` is shared by T5, NLLB,
> BART, Marian, M2M-100 and Pegasus — so bumping it to gate one new architecture would make
> every newly converted model of all of those unloadable on existing deployments, while an
> explicit flag wouldn't have prevented the same silent-fallback anyway.*

---

## Quick self-test

Answer out loud, without looking:

1. Why is caching the position bias valid for T5 but not UMT5? *(identical inputs across
   layers ⇒ identical output; UMT5 breaks the premise)*
2. What does an alias become in `model.bin`? *(a name→name pair; the tensor is stored once)*
3. What does `get_variable_if_exists` return for two aliased names? *(the same raw pointer —
   same `shared_ptr`)*
4. Name a case where pointer comparison gives the "wrong" answer. *(none produces wrong
   output; the degenerate case is all-identical UMT5 tables, where sharing is numerically
   correct anyway)*
5. What breaks if aliasing stops working? *(T5 gets slower — never incorrect)*
6. Which models would a `TransformerSpec.revision` bump break? *(every newly converted T5,
   mT5, NLLB, BART, mBART, Marian, M2M-100, Pegasus — on any older runtime)*

If any answer is shaky, re-read that section before filing.
