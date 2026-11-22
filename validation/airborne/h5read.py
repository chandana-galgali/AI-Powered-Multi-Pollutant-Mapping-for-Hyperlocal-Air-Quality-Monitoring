"""Open HDF5 files with no compiled extensions.

h5py, and numcodecs (a dependency of new pyfive releases), ship DLLs that Windows
Application Control may block. This module tries h5py first. If that fails, it
installs a pure-Python stand-in for numcodecs and then uses pyfive.
"""
import sys
import types

import numpy as np


def _numcodecs_stub():
    mod = types.ModuleType("numcodecs")

    class _Codec:
        def __init__(self, *a, **k):
            self.args, self.kwargs = a, k

    class Fletcher32(_Codec):
        """HDF5 Fletcher-32 filter: the last 4 bytes are the checksum. Strip them."""
        codec_id = "fletcher32"

        def decode(self, buf, out=None):
            b = memoryview(bytes(buf))[:-4]
            if out is not None:
                np.copyto(np.frombuffer(out, dtype="u1"), np.frombuffer(b, dtype="u1"))
                return out
            return bytes(b)

        def encode(self, buf):
            raise NotImplementedError

    class Shuffle(_Codec):
        codec_id = "shuffle"

        def decode(self, buf, out=None):
            size = self.kwargs.get("elementsize", self.args[0] if self.args else 1)
            a = np.frombuffer(bytes(buf), dtype="u1")
            n = a.size // size
            body = a[: n * size].reshape(size, n).T.reshape(-1)
            return bytes(body) + bytes(a[n * size:])

    mod.Fletcher32 = Fletcher32
    mod.Shuffle = Shuffle

    def __getattr__(name):  # any other codec: a placeholder class
        return type(name, (_Codec,), {})

    mod.__getattr__ = __getattr__
    return mod


def get_lib():
    try:
        import h5py
        return h5py
    except (ImportError, OSError):
        pass
    try:
        import pyfive
        return pyfive
    except (ImportError, OSError):
        for k in [k for k in sys.modules if k == "numcodecs" or k.startswith("numcodecs.") or k.startswith("pyfive")]:
            del sys.modules[k]
        sys.modules["numcodecs"] = _numcodecs_stub()
        import pyfive
        return pyfive


h5lib = get_lib()


def is_dataset(obj):
    return hasattr(obj, "shape") and hasattr(obj, "dtype")
