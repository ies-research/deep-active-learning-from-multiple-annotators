from __future__ import annotations
from typing import Any, Sequence
import numpy as np


def as_int_array(y: Sequence[Any]) -> np.ndarray:
    out = np.empty((len(y),), dtype=np.int64)
    for i, v in enumerate(y):
        a = np.asarray(v)
        out[i] = int(a.reshape(-1)[0])
    return out


def stack_fixed(x: Sequence[Any], *, dtype=None) -> np.ndarray:
    arr = [np.asarray(v) for v in x]
    out = np.stack(arr)
    return out.astype(dtype, copy=False) if dtype is not None else out
