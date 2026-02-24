from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def images_to_numpy(
    images: Sequence[Any],
    *,
    channels_first: bool,
    as_float32: bool,
    scale_01: bool,
) -> np.ndarray:
    """
    Convert a sequence of image-like objects into a single NumPy array.

    This function stacks a list/sequence of images into a batch array. Images
    may be provided as NumPy arrays, PIL images, or any object supported by
    ``np.asarray``. Grayscale images are promoted to have a singleton channel
    dimension.

    Parameters
    ----------
    images : sequence of Any
        Sequence of image-like objects. Each element must be convertible via
        ``np.asarray`` to an array of shape ``(H, W)`` (grayscale) or
        ``(H, W, C)`` (color). All images are assumed to have the same spatial
        dimensions and number of channels as the first image.
    channels_first : bool
        If True, return array in ``(N, C, H, W)`` format. If False, return
        array in ``(N, H, W, C)`` format.
    as_float32 : bool
        If True, convert images to ``np.float32``. If False, the output dtype
        is ``np.uint8``.
    scale_01 : bool
        If True and ``as_float32`` is True, scale values by ``1/255`` to map
        typical 8-bit image ranges to approximately ``[0, 1]``. Ignored if
        ``as_float32`` is False.

    Returns
    -------
    X : numpy.ndarray
        Batched image array with shape:
        - ``(N, C, H, W)`` if ``channels_first=True``
        - ``(N, H, W, C)`` if ``channels_first=False``

        The dtype is ``np.float32`` if ``as_float32=True``, otherwise
        ``np.uint8``.

    Notes
    -----
    - This function assumes all images match the shape of the first image. If
      later images differ, assignment into the preallocated batch array will
      raise a broadcasting or shape error.
    - If you pass float images in range ``[0, 1]`` and also set
      ``scale_01=True``, you will incorrectly scale them down again.
    - No color space conversion is performed (e.g., RGB vs BGR), and no
      resizing is performed.

    Examples
    --------
    >>> X = images_to_numpy(images, channels_first=True, as_float32=True, scale_01=True)
    >>> X.shape
    (N, C, H, W)
    """
    # Convert first image to infer expected shape (H, W, C) for the whole
    # batch. We intentionally use the first element as the "template" and
    # assume all other images match it. If they don't, numpy will complain
    # later.
    first = np.asarray(images[0])

    # Promote grayscale (H, W) to (H, W, 1) so everything has an explicit
    # channel dim.
    if first.ndim == 2:
        first = first[..., None]

    # Expected spatial and channel dimensions inferred from first image.
    h, w, c = first.shape

    # Batch size.
    n = len(images)

    # Preallocate output array for speed and to ensure a contiguous result.
    # Shape depends on requested channel order; dtype depends on as_float32.
    X = np.empty(
        (n, c, h, w) if channels_first else (n, h, w, c),
        dtype=np.float32 if as_float32 else np.uint8,
    )

    # Convert each image and write it into the preallocated batch array.
    for i, im in enumerate(images):
        # Convert to numpy array (may copy depending on input type).
        a = np.asarray(im)

        # Promote grayscale to have an explicit channel dimension.
        if a.ndim == 2:
            a = a[..., None]

        # Optional dtype conversion + scaling. Scaling only makes sense for
        # typical 8-bit images; for other ranges it's on you to not be wrong.
        if as_float32:
            a = a.astype(np.float32, copy=False)
            if scale_01:
                a *= 1.0 / 255.0

        # Optional layout change from (H, W, C) to (C, H, W).
        if channels_first:
            a = np.transpose(a, (2, 0, 1))

        # Store into batch array. If shapes don't match the template,
        # this errors.
        X[i] = a

    return X
