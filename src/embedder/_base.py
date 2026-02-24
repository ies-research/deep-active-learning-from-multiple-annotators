from __future__ import annotations
from typing import Any, Dict, Protocol, Sequence
import numpy as np


class Embedder(Protocol):
    def fingerprint(self) -> Dict[str, Any]: ...
    def embed(self, x: Sequence[Any]) -> np.ndarray: ...
