from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Sequence
import numpy as np
from ._base import Embedder
from ._utils import images_to_numpy


@dataclass
class IdentityImageEmbedder(Embedder):
    channels_first: bool = True
    as_float32: bool = True
    scale_01: bool = True

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "type": "identity_image",
            "channels_first": self.channels_first,
            "as_float32": self.as_float32,
            "scale_01": self.scale_01,
        }

    def embed(self, x: Sequence[Any]) -> np.ndarray:
        return images_to_numpy(
            x,
            channels_first=self.channels_first,
            as_float32=self.as_float32,
            scale_01=self.scale_01,
        )


@dataclass
class IdentityTabularEmbedder(Embedder):
    as_float32: bool = True
    standardize: bool = False
    eps: float = 1e-12

    def __post_init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: Sequence[Any]) -> "IdentityTabularEmbedder":
        X = np.asarray(x, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                "IdentityTabularEmbedder.fit expects a 2D array of shape "
                f"(n_samples, n_features), got {X.shape}."
            )
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.maximum(std, float(self.eps))
        return self

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "type": "identity_tabular",
            "as_float32": self.as_float32,
            "standardize": self.standardize,
            "eps": self.eps,
            "fitted": self.mean_ is not None and self.scale_ is not None,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
        }

    def embed(self, x: Sequence[Any]) -> np.ndarray:
        X = np.asarray(x, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                "IdentityTabularEmbedder.embed expects a 2D array of shape "
                f"(batch_size, n_features), got {X.shape}."
            )

        if self.standardize:
            if self.mean_ is None or self.scale_ is None:
                raise RuntimeError(
                    "IdentityTabularEmbedder must be fitted before calling "
                    "embed when standardize=True."
                )
            X = (X - self.mean_) / self.scale_

        if self.as_float32:
            return X.astype(np.float32, copy=False)
        return X
