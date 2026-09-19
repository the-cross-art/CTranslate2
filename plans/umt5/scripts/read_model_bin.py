#!/usr/bin/env python
"""Read a CTranslate2 model.bin header: variables and the alias table.

Mirrors ModelSpec._serialize in python/ctranslate2/specs/model_spec.py.
Usage: read_model_bin.py <model_dir> [name-filter]
"""
import struct, sys, os

def rd(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))[0]

def rstr(f):
    ln = rd(f, "H")
    b = f.read(ln)
    return b[:-1].decode("utf-8")

def main():
    d = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    path = d if os.path.isfile(d) else os.path.join(d, "model.bin")
    with open(path, "rb") as f:
        binver = rd(f, "I"); name = rstr(f); rev = rd(f, "I")
        nvar = rd(f, "I")
        print(f"binary_version={binver} spec={name} revision={rev} variables={nvar}")
        variables = []
        for _ in range(nvar):
            vn = rstr(f)
            rank = rd(f, "B")
            shape = [rd(f, "I") for _ in range(rank)]
            rd(f, "B")                      # dtype id
            nbytes = rd(f, "I")
            f.seek(nbytes, os.SEEK_CUR)     # skip payload
            variables.append((vn, shape))
        nalias = rd(f, "I")
        aliases = [(rstr(f), rstr(f)) for _ in range(nalias)]

    sel = [v for v in variables if filt in v[0]] if filt else variables
    print(f"\n--- variables matching {filt!r}: {len(sel)} ---")
    for vn, shape in sel:
        print(f"    {vn}  {tuple(shape)}")
    asel = [a for a in aliases if filt in a[0] or filt in a[1]] if filt else aliases
    print(f"\n--- aliases: {nalias} total, {len(asel)} matching {filt!r} ---")
    for a, t in asel:
        print(f"    {a}\n      -> {t}")

if __name__ == "__main__":
    main()
