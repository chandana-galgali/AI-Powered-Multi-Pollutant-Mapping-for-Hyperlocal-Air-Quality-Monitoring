"""Print the layout of the LISTOS airborne HDF5 files (groups, datasets, shapes, units).

Usage:  python validation/airborne/inspect_h5.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h5read import h5lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "listos2018", "*.h5")))
if not files:
    sys.exit("No .h5 files in validation/airborne/listos2018/")


def attrs(obj):
    out = {}
    for k, v in obj.attrs.items():
        if isinstance(v, bytes):
            v = v.decode(errors="replace")
        elif isinstance(v, np.ndarray) and v.size == 1:
            v = v.item()
            if isinstance(v, bytes):
                v = v.decode(errors="replace")
        s = str(v)
        out[k] = s[:120]
    return out


def show(name, obj):
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        line = f"  DATASET {name}  shape={obj.shape} dtype={obj.dtype}"
        a = attrs(obj)
        keep = {k: a[k] for k in a if k.lower() in
                ("units", "long_name", "description", "_fillvalue", "fillvalue",
                 "missing_value", "scale_factor", "valid_range")}
        if keep:
            line += f"  {keep}"
        if obj.size and obj.ndim <= 3 and obj.dtype.kind in "fiu":
            sl = tuple(slice(None, None, max(1, s // 200)) for s in obj.shape)
            d = np.asarray(obj[sl], dtype="float64")
            d = d[np.isfinite(d) & (np.abs(d) < 1e30)]
            if d.size:
                line += f"  sample[min={d.min():.4g} med={np.median(d):.4g} max={d.max():.4g}]"
        print(line)
    else:
        print(f"GROUP {name}  {attrs(obj) or ''}")


def walk(group, prefix=""):
    for key in group.keys():
        obj = group[key]
        name = f"{prefix}{key}"
        show(name, obj)
        if not (hasattr(obj, "shape") and hasattr(obj, "dtype")):
            walk(obj, name + "/")


print("Reader:", h5lib.__name__)
f = h5lib.File(files[0], "r")
print("FILE", os.path.basename(files[0]))
print("ROOT ATTRS", attrs(f))
walk(f)
print("\nOther files:", [os.path.basename(p) for p in files[1:]])
