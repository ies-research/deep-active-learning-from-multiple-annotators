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
