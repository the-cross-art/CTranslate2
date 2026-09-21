# C++ for a Python developer — using this repo as the example

You know Python. This maps every Python habit onto its C++ equivalent, then walks the actual
files in this repo. All output below is from your machine.

---

## Part 1 — Why `export DYLD_LIBRARY_PATH=...`?

Short answer: **it's C++'s `PYTHONPATH`, but for compiled libraries, and it's needed at the
moment you run — not when you build.**

### What actually happens

`import ctranslate2` runs `python/ctranslate2/__init__.py`, which does:

```python
from ctranslate2._ext import ...
```

`_ext` is not a `.py` file. It's `python/ctranslate2/_ext.cpython-312-darwin.so` — compiled
machine code. And that file does **not** contain CTranslate2. It's a thin bridge that calls
into the real library. Ask it what it needs:

```bash
otool -L python/ctranslate2/_ext.cpython-312-darwin.so
```
```
@rpath/libctranslate2.4.dylib     ← the actual engine, ~3 MB of compiled C++
@rpath/libc++.1.dylib
/usr/lib/libSystem.B.dylib
```

`@rpath` means *"look in my baked-in search paths."* Ask what those are:

```bash
otool -l python/ctranslate2/_ext.cpython-312-darwin.so | grep -A2 LC_RPATH | grep "path "
```
```
path /opt/homebrew/Caskroom/miniforge/base/lib
path /usr/local/lib
```

**Neither of those is where we put the library.** Ours is at
`/Users/imrannazir/Documents/CTranslate2/install-umt5/lib/`. Those baked-in paths came from
the Python interpreter's build defaults, not from us.

So without help, it fails:

```bash
env -u DYLD_LIBRARY_PATH python -c "import ctranslate2"
```
```
ImportError: dlopen(...): Library not loaded: @rpath/libctranslate2.4.dylib
```

`DYLD_LIBRARY_PATH` adds one more directory to that search list at runtime. That's all.

### The Python analogy

| | Python | C++ |
|---|---|---|
| "where do I find code?" | `PYTHONPATH` / `sys.path` | `DYLD_LIBRARY_PATH` (macOS), `LD_LIBRARY_PATH` (Linux) |
| checked when | on `import` | when the OS loads the `.so`, before your first line runs |
| what it finds | `.py` files | `.dylib` / `.so` machine code |

`DYLD` = **dyld**, macOS's *dynamic linker* — the program that stitches executables together
at launch. On Linux the same thing is `ld.so` and the variable is `LD_LIBRARY_PATH`.

### Why didn't we just avoid it?

Three options, and why we're here:

1. `sudo make install` into `/usr/local/lib` — already on the search path, no env var needed.
   But it writes to your system, needs `sudo`, and would collide with any other CTranslate2.
2. Bake the right `rpath` into the `.so` at build time. Correct, but means fighting
   `setup.py`, which hardcodes assumptions (see upstream issue #2082, which is literally
   about this).
3. **Set the env var.** Zero system changes, fully reversible, works immediately.

We chose 3 because we're developing, not deploying. A released wheel uses option 2.

> **This is also why it's `install-umt5/` and not `build-umt5/`.** Our first attempt pointed
> `CTRANSLATE2_ROOT` at the build tree and failed with
> `fatal error: 'ctranslate2/generator.h' file not found`. A build tree is scratch space with
> `.o` files scattered around; an *install prefix* is the tidy `include/` + `lib/` layout that
> other software expects. `cmake --install` creates that tidy copy.

---

## Part 2 — The mental model, Python vs C++

### Python: one step

```
foo.py  ──python──►  runs
```
The interpreter reads your source every time. Edit, re-run, done.

### C++: three steps, before anything runs

```
foo.cc  ──preprocess──►  ──compile──►  foo.o  ──link──►  program / library
```

1. **Preprocess** — every `#include "x.h"` is literally pasted in, producing one huge file.
2. **Compile** — that file becomes `foo.o`, machine code for *one* source file. It has holes
   where it calls things defined elsewhere.
3. **Link** — all `.o` files are stitched together, holes filled. Output: an executable or a
   library.

Your repo right now: **167 `.o` files** → linked into one **2.9 MB** `libctranslate2.4.8.2.dylib`,
from 1.4 MB of `src/` and 1.7 MB of `include/`.

**This is why editing C++ needs a rebuild and editing Python doesn't.** There's no
interpreter reading `transformer.cc` at runtime — that file was turned into machine code
hours ago. Change it and you must redo steps 1–3.

### The translation table

| Python | C++ | Notes |
|---|---|---|
| `.py` | `.cc` / `.cpp` | the implementation |
| — | `.h` / `.hpp` | **header**: declarations only. Like a `.pyi` stub, but mandatory |
| `import x` | `#include "x.h"` **+ link against x** | two separate steps — this trips everyone up |
| `__pycache__/*.pyc` | `.o` object files | cached compilation |
| package | **library** — `.dylib`/`.so` (dynamic) or `.a` (static) | |
| `pip install` | *no equivalent* | you compile it yourself, or a system package manager provides it |
| `venv` | *no equivalent* | closest thing: a build dir + install prefix |
| `site-packages/` | install prefix: `include/` + `lib/` | your `install-umt5/` |
| `PYTHONPATH` | `-I` (headers, compile) / `-L` (libs, link) / `DYLD_LIBRARY_PATH` (run) | three different paths for three phases |
| `setup.py` / `pyproject.toml` | `CMakeLists.txt` | |
| `pip` | `cmake` + `ninja` | |
| `python foo.py` | `./foo` after compiling | |
| `TypeError` at runtime | compile error, before running | C++ checks types up front |

### The single biggest difference

In Python, **a header doesn't exist** — `import x` gets you everything.

In C++, `#include "transformer.h"` only tells the compiler *what exists* ("there is a class
`TransformerEncoder` with a constructor"). It does **not** provide the code. The code lives
in `transformer.cc` → `transformer.o` → the library, and the **linker** connects them.

That's why our change touched **two** files: the declaration in `include/.../transformer.h`
and the implementation in `src/layers/transformer.cc`. In Python that would be one file.

---

## Part 3 — "Creating an env and running", the C++ way

Here's your Python workflow and the exact C++ counterpart in this repo.

### Python
```bash
python -m venv .venv          # 1. isolated environment
source .venv/bin/activate     # 2. activate
pip install -r requirements   # 3. dependencies
python foo.py                 # 4. run
```

### C++ (what we actually ran)

**1. Get dependencies.** No `pip`. This repo vendors them as **git submodules** — other git
repos nested inside. They weren't downloaded:

```bash
git submodule update --init --recursive third_party/spdlog third_party/cpu_features \
                                        third_party/ruy third_party/googletest
```
This is the closest thing to `pip install -r requirements.txt`. Note we fetched 4 of 7 —
`cutlass` and `thrust` are CUDA-only and you have no NVIDIA GPU.

**2. Configure.** `cmake` inspects your machine and writes build instructions:

```bash
cmake -S . -B build-umt5 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_MKL=OFF -DWITH_ACCELERATE=ON -DWITH_RUY=ON \
  -DBUILD_CLI=OFF -DBUILD_TESTS=ON -DOPENMP_RUNTIME=NONE
```

- `-S .` source, `-B build-umt5` where to put output. **Never build in the source dir** —
  `build-umt5/` is disposable; delete it any time.
- `-DWITH_MKL=OFF -DWITH_ACCELERATE=ON` — MKL is Intel's math library and doesn't exist for
  Apple Silicon. Accelerate is Apple's. These pick which matrix-multiply backend to use.
- `-DCMAKE_BUILD_TYPE=Release` — optimize. `Debug` compiles faster and is debuggable but runs
  much slower.
- cmake is a **generator**: it doesn't compile, it writes files for the thing that does.

**3. Build.** `ninja` runs the compiler across 167 files in parallel:

```bash
cmake --build build-umt5 -j8        # -j8 = 8 files at once
```

**4. Install** into a tidy prefix — this is what makes `install-umt5/` look like
`site-packages`:

```bash
cmake --install build-umt5 --prefix "$PWD/install-umt5"
```

**5. Build the Python bridge** against that prefix:

```bash
cd python && CTRANSLATE2_ROOT=.../install-umt5 pip install -e . --no-build-isolation
```

**6. Run** — and here's where `DYLD_LIBRARY_PATH` finally matters:

```bash
export DYLD_LIBRARY_PATH="$PWD/install-umt5/lib:$DYLD_LIBRARY_PATH"
python -c "import ctranslate2"
```

### Why there's no C++ "venv"

A venv works because Python resolves imports at runtime via `sys.path`, so you can swap the
search path. C++ bakes decisions in at **compile** and **link** time. Isolation instead means:
a separate build directory, a separate install prefix, and explicit paths. `build-umt5/` +
`install-umt5/` **are** your environment — delete both and you're back to a clean machine.

---

## Part 4 — How Python and C++ talk: pybind11

`python/cpp/` is the bridge — C++ files whose only job is exposing C++ classes to Python:

```
python/cpp/module.cc          ← defines the Python module
python/cpp/translator.cc      ← exposes ctranslate2.Translator
python/cpp/encoder.cc         ← exposes ctranslate2.Encoder
python/cpp/storage_view.cc    ← exposes ctranslate2.StorageView
...
```

Compiled together into `_ext.cpython-312-darwin.so`. The chain when you call
`translator.translate_batch([...])`:

```
your Python code
   └─ ctranslate2/translator.py        (thin Python wrapper)
        └─ _ext.so                     (pybind11 glue: Python objects → C++ types)
             └─ libctranslate2.dylib   (the real engine)
                  └─ src/translator.cc → src/layers/transformer.cc → src/ops/*
```

`_ext.so` is small glue. `libctranslate2.dylib` is the 3 MB engine. That's why two separate
files need finding, and why only the second one needs `DYLD_LIBRARY_PATH`.

---

## Part 5 — What each part of the repo does

### The Python half — pure `.py`, no compilation, edit and re-run

```
python/ctranslate2/
├── __init__.py              imports _ext, exposes the public API
├── converters/
│   ├── transformers.py      ★ ~4000 lines. One loader class per architecture.
│   │                          UMT5Loader is here (line ~1363).
│   ├── converter.py         base class: orchestrates load → spec → save
│   ├── marian.py, fairseq.py, opennmt_*.py   other source frameworks
│   └── utils.py             helpers e.g. fuse_linear (merges Q,K,V into one matrix)
└── specs/
    ├── model_spec.py        ★ serializes a spec to model.bin. Does the ALIASING
    │                          (lines 169-189) your C++ fix depends on.
    ├── transformer_spec.py  shape of a transformer: layers, pre/post-norm, gated FFN
    ├── attention_spec.py    shape of one attention block, incl. relative_attention_bias
    └── common_spec.py       Linear, LayerNorm, Embeddings; the Activation enum
```

**The `spec` is the contract.** A converter's only job is to fill a spec. `model_spec.py`
turns it into bytes. The C++ reads those bytes. Python and C++ never talk during conversion —
they communicate through `model.bin`.

### The C++ half — needs compiling

```
include/ctranslate2/          HEADERS — declarations ("what exists")
└── layers/transformer.h      ★ we added `_shared_position_bias` here

src/                          IMPLEMENTATIONS — actual code
├── models/
│   └── model.cc              ★ reads model.bin. register_variable_alias (line 279)
│                               makes aliased names share ONE StorageView — the
│                               pointer identity your fix tests for.
├── layers/
│   ├── transformer.cc        ★ encoder & decoder stacks. YOUR FIX IS HERE.
│   │                           line 409  has_shared_position_bias()  (new)
│   │                           line 486  encoder chooses shared vs per-layer
│   │                           line 759  decoder does the same
│   ├── attention.cc          attention maths. Line 233-246 is the position-bias
│   │                           branch — pass nullptr and each layer computes its own.
│   ├── common.cc             Dense (matrix multiply + bias), LayerNorm, Embeddings
│   └── decoder.cc            decoding state / KV cache
├── ops/                      primitives: matmul, softmax, gelu, gather, transpose...
├── cpu/                      CPU-specific fast paths (SIMD kernels)
│   ├── kernels.cc            ★ gelu_tanh_func lives here — we verified it matches HF
│   └── primitives.cc         dispatch to the right CPU instruction set
├── cuda/                     NVIDIA GPU versions (not compiled on your Mac)
├── translator.cc             top-level translate API
└── decoding.cc               beam search, sampling

python/cpp/                   pybind11 bridge (compiled into _ext.so)
third_party/                  vendored dependencies (git submodules)
tests/                        C++ tests → build-umt5/tests/ctranslate2_test
CMakeLists.txt                the build recipe
```

### A full request, end to end

```
1.  tokenizer (Python, HF)      "The house" → ['▁The','▁house']
2.  translator.py               Python wrapper
3.  _ext.so                     converts Python list → C++ StorageView
4.  src/translator.cc           orchestrates
5.  src/models/model.cc         weights already loaded in memory
6.  src/layers/transformer.cc   TransformerEncoder::operator()
      └─ per layer: attention.cc  ← YOUR FIX decides shared vs per-layer bias here
      └─ per layer: common.cc     FFN
7.  src/decoding.cc             beam search, calling TransformerDecoder each step
8.  back up through _ext.so     C++ → Python list of token strings
9.  tokenizer (Python)          tokens → "Das Haus"
```

---

## Part 6 — Practical recipes

### Edit Python
```bash
vim python/ctranslate2/converters/transformers.py
python -c "..."        # done. no rebuild.
```

### Edit C++
```bash
vim src/layers/transformer.cc
cmake --build build-umt5 -j8                                # recompile (~1 min: only changed files)
cmake --install build-umt5 --prefix "$PWD/install-umt5"     # publish — DON'T FORGET THIS
python -c "..."
```

> Forgetting the install step is the #1 "my change did nothing" trap: you rebuilt
> `build-umt5/`, but Python loads from `install-umt5/lib/`.

### Reading a compile error

C++ errors are long. **The first line is the real one**; the rest is context:

```bash
cmake --build build-umt5 -j8 2>&1 | grep -E "error" | head -5
```

You already hit two:
- `use of undeclared identifier 'spdlog'` — used something without `#include`ing its header.
  Python's equivalent of forgetting an import, but caught at compile time.
- `'ctranslate2/generator.h' file not found` — the compiler's header search path (`-I`)
  didn't include the right directory.

### Warnings worth caring about

`-Wreorder` in particular. C++ initializes members in **declaration** order, not the order you
write in the constructor. Get it wrong and you read a variable before it's set — no crash,
just wrong values. That's why the plan made "0 reorder warnings" a gate. You had 0.

### Start over
```bash
rm -rf build-umt5 install-umt5    # safe: nothing in them is source
```

### Debug C++ the way you'd use `print()`
```cpp
fprintf(stderr, "value=%d\n", (int)some_variable);
```
That's exactly what we did to capture the `shared_position_bias=0/1` evidence, then removed
it. There's no REPL — `printf` debugging is normal and respectable in C++.

---

## Part 7 — The five things that will bite you

1. **Header + implementation are separate files.** Change a class's members → edit `.h` *and*
   `.cc`.
2. **Editing C++ requires rebuild + install.** Two commands, both every time.
3. **Declaration order matters** for constructor initialization (see `-Wreorder`).
4. **Three different search paths** — `-I` for headers at compile, `-L` for libs at link,
   `DYLD_LIBRARY_PATH` for libs at run. Python has one `sys.path`.
5. **No garbage collector.** Memory is freed when a variable goes out of scope, or manually.
   This repo uses `std::unique_ptr` / `std::shared_ptr`, which free automatically — which is
   precisely why the aliasing trick works: `shared_ptr` lets two names own the same tensor,
   and your fix compares those pointers.

---

## Part 8 — The one-line setup, explained

```bash
export DYLD_LIBRARY_PATH="$PWD/install-umt5/lib:$DYLD_LIBRARY_PATH"
```

- `$PWD/install-umt5/lib` — where *our* freshly built engine lives
- `:$DYLD_LIBRARY_PATH` — **append the existing value**, don't clobber it; other software may
  rely on entries already there
- `export` — make it visible to programs this shell launches, not just the shell
- It vanishes when you close the terminal. That's a feature: nothing about your machine is
  permanently modified.

Verify you're on our build, not a downloaded wheel:

```bash
python -c "import ctranslate2; print(ctranslate2.__file__)"
# must print a path inside this repo
```
