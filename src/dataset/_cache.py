from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import asdict, is_dataclass
import hashlib, json, os
import numpy as np


def to_plain(obj: Any) -> Any:
    """
    Convert OmegaConf containers to plain Python types and recursively make objects JSON-serializable.

    Handles:
    - OmegaConf DictConfig/ListConfig
    - dataclasses (including nested dataclasses)
    - dict/list/tuple/set
    - numpy scalars/arrays
    - pathlib.Path
    """
    # OmegaConf -> plain
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            obj = OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass

    # dataclass -> dict (recursive)
    if is_dataclass(obj):
        obj = asdict(obj)

    # pathlib.Path -> str
    if isinstance(obj, Path):
        return str(obj)

    # containers
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, set):
        return [to_plain(v) for v in sorted(obj, key=lambda x: str(x))]

    # numpy types
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    return obj


def sha1_json(obj: Dict[str, Any]) -> str:
    plain = to_plain(obj)
    s = json.dumps(plain, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha1(s).hexdigest()


def sha1_bytes(b: bytes) -> str:
    """Return SHA1 hex digest of bytes."""
    return hashlib.sha1(b).hexdigest()


def npz_load(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def npz_save(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def file_signature(path: str) -> Dict[str, Any]:
    p = Path(path)
    try:
        st = p.stat()
        return {
            "abs": str(p.resolve()),
            "size": int(st.st_size),
            "mtime_ns": int(
                getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
            ),
        }
    except FileNotFoundError:
        return {"abs": str(p), "missing": True}


def dir_signature(path: str, *, max_files: int = 2000) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"abs": str(p), "missing": True}
    files = []
    count = 0
    for root, _, fnames in os.walk(p):
        for f in fnames:
            fp = Path(root) / f
            try:
                st = fp.stat()
                files.append(
                    (
                        str(fp.relative_to(p)),
                        int(st.st_size),
                        int(
                            getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
                        ),
                    )
                )
            except FileNotFoundError:
                continue
            count += 1
            if count >= max_files:
                break
        if count >= max_files:
            break
    files.sort()
    h = hashlib.sha1(
        json.dumps(files, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "abs": str(p.resolve()),
        "n_files_seen": len(files),
        "listing_sha1": h,
    }
