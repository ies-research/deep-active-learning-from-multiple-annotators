import numpy as np


def coerce_annotator_vector(annotator_indices, values, *, name):
    """
    Map a global annotator vector/dict to the local annotator order.

    Parameters
    ----------
    annotator_indices : array-like of shape (n_sel_annotators,)
        Global annotator ids in local assigner order.
    values : None, dict, or array-like
        Annotator-aligned quantity. Dicts are interpreted as
        ``global_annotator_id -> value``.
    name : str
        Used in error messages.
    """
    annotator_indices = np.asarray(annotator_indices, dtype=int)
    A = len(annotator_indices)

    if values is None:
        return None

    if isinstance(values, dict):
        out = np.zeros(A, dtype=int)
        for j, a_glob in enumerate(annotator_indices):
            out[j] = int(values.get(int(a_glob), 0))
        if (out < 0).any():
            raise ValueError(f"{name} must be non-negative.")
        return out

    arr = np.asarray(values)
    if arr.shape == (A,):
        arr = arr.astype(int, copy=False)
        if (arr < 0).any():
            raise ValueError(f"{name} must be non-negative.")
        return arr

    if arr.ndim == 1:
        max_idx = int(annotator_indices.max(initial=-1))
        if arr.shape[0] > max_idx:
            out = arr[annotator_indices].astype(int, copy=False)
            if (out < 0).any():
                raise ValueError(f"{name} must be non-negative.")
            return out

    raise ValueError(
        f"`{name}` must be dict, local array of shape ({A},), "
        f"or global array indexable by annotator id; got shape {arr.shape}."
    )
