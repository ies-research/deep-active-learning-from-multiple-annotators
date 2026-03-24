from __future__ import annotations
from numbers import Number
from typing import Any, Sequence
import numpy as np


def infer_label_values(
    *label_sequences: Sequence[Any], feature: Any = None
) -> list[Any]:
    names = getattr(feature, "names", None)
    if names is not None:
        return [_to_scalar(name) for name in names]

    values = []
    for seq in label_sequences:
        values.extend(_to_scalar(v) for v in seq)

    if not values:
        return []

    return sorted(set(values), key=_sort_key)


def stack_tabular_columns(
    columns: Any, *, column_order: Sequence[str], dtype=np.float32
) -> np.ndarray:
    if isinstance(columns, dict):
        cols = [np.asarray(columns[name]) for name in column_order]
        out = np.column_stack(cols)
    else:
        out = np.asarray(columns)
    if out.ndim != 2:
        raise ValueError(
            "Expected tabular features with shape (n_samples, n_features), "
            f"got {out.shape}."
        )
    return out.astype(dtype, copy=False) if dtype is not None else out


def stack_fixed(x: Sequence[Any], *, dtype=None) -> np.ndarray:
    arr = [np.asarray(v) for v in x]
    out = np.stack(arr)
    return out.astype(dtype, copy=False) if dtype is not None else out


def _to_scalar(v: Any) -> Any:
    a = np.asarray(v)
    scalar = a.reshape(-1)[0]
    return scalar.item() if hasattr(scalar, "item") else scalar


def _sort_key(v: Any) -> tuple[str, str]:
    if isinstance(v, Number) and not isinstance(v, bool):
        return ("number", f"{float(v):.20g}")
    if isinstance(v, str):
        return ("string", v)
    return (type(v).__name__, repr(v))
