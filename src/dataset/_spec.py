from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Literal


@dataclass(frozen=True)
class HFDatasetSpec:
    source: str
    source_kind: Literal["hub_or_local_script", "from_disk"] = (
        "hub_or_local_script"
    )
    dataset_config: Optional[str] = None
    revision: Optional[str] = None
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[Dict[str, Any]] = None

    train_splits: Sequence[str] = ("train",)
    test_split: str = "test"

    x_key: str | Sequence[str] = "image"
    y_key: str = "label"
    z_key: Optional[str] = None

    local_signature: bool = True
