from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np
from skactiveml.utils import ExtLabelEncoder

from ..embedder import Embedder
from ._spec import HFDatasetSpec
from ._io import load_datasetdict, merge_train_splits, spec_fingerprint
from ._utils import (
    infer_label_values,
    stack_fixed,
    stack_tabular_columns,
)
from ._cache import sha1_json, npz_load, npz_save


@dataclass
class PipelineConfig:
    """
    Configuration for `HFNumpyFeaturePipeline`.

    Parameters
    ----------
    batch_size : int, default=64
        Mini-batch size used for embedding.
    cache_dir : str or None, default=".hf_embed_cache"
        Directory where cached arrays are stored. If None, caching is disabled
        and the pipeline will raise when asked to write memmaps (because it
        needs a path). If you want "no reuse" but still memmap, provide a
        directory and set `reuse_cache=False`.
    reuse_cache : bool, default=True
        If True and cached files exist for a split, load them instead of
        recomputing. If False, recompute and overwrite the cache files.
    memmap_dtype : numpy dtype, default=np.float16
        Dtype used for the on-disk `X` array. float16 typically halves disk
        usage versus float32 and is often sufficient for embeddings.
    mmap_mode : {"r", "r+", "c"}, default="r"
        Memory-map mode used when opening cached `.npy` files.

    Notes
    -----
    - `X` is always stored as an uncompressed `.npy` file so it can be
      memory-mapped. Compressed `.npz` cannot be memory-mapped in a useful way.
    - `y` and `z` are stored as a small `.npz` file and loaded into memory.
    """

    batch_size: int = 64
    cache_dir: Optional[str] = ".hf_embed_cache"
    reuse_cache: bool = True
    memmap_dtype: Any = np.float16
    mmap_mode: str = "r"


class HFNumpyFeaturePipeline:
    """
    Build NumPy arrays (optionally memory-mapped) from a HuggingFace dataset
    spec.

    This pipeline loads a HuggingFace DatasetDict according to `HFDatasetSpec`,
    embeds the input data using an `Embedder`, and returns arrays:

    - `X_train`, `X_test`: **memory-mapped** arrays backed by `.npy` files
    - `y_train`, `y_test`: in-memory integer arrays
    - optionally `z_train`: in-memory integer array (e.g., annotator labels)

    The key behavior is that embeddings `X` are *streamed* to disk in batches,
    avoiding `np.concatenate` and large RAM peaks. Downstream code may keep
    `X` as a memmap or materialize it later via `np.asarray(X)`.

    Parameters
    ----------
    spec : HFDatasetSpec
        Dataset specification (split names, column keys, etc.).
    embedder : Embedder
        Embedder that converts a batch of inputs to a NumPy array.
    cfg : PipelineConfig or None, default=None
        Pipeline configuration.

    Returns
    -------
    get_arrays() : dict[str, numpy.ndarray]
        Dictionary with keys:
        - `"X_train"`: memmap array
        - `"y_train"`: ndarray
        - `"X_test"`: memmap array
        - `"y_test"`: ndarray
        - optionally `"z_train"`: ndarray

    Notes
    -----
    - Memmap arrays stay on disk. Some operations will implicitly materialize
      them.
    - This pipeline assumes the embedder output shape is **constant** across
      the split. If it changes (e.g., variable-size feature maps), memmap
      writing fails loudly.
    """

    def __init__(
        self,
        spec: HFDatasetSpec,
        embedder: Embedder,
        *,
        cfg: Optional[PipelineConfig] = None,
        x_adapter: Optional[Callable[[Any], Any]] = None,
    ):
        self.spec = spec
        self.embedder = embedder
        self.cfg = cfg or PipelineConfig()
        self.x_adapter = x_adapter
        self.label_encoder_: ExtLabelEncoder | None = None
        self.label_values_: tuple[Any, ...] | None = None

        if self.cfg.cache_dir is None:
            raise ValueError(
                "cache_dir=None is not supported in this memmap-first "
                "pipeline. Provide a cache_dir (even a temp folder) so `.npy` "
                "files have a home."
            )

    def _debug(self, message: str) -> None:
        print(
            "[HFNumpyFeaturePipeline] "
            f"source={self.spec.source!r} source_kind={self.spec.source_kind} "
            f"cache_dir={self.cfg.cache_dir!r} :: {message}",
            flush=True,
        )

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """
        Load the dataset, compute or reuse embeddings, and return NumPy arrays.

        Returns
        -------
        arrays : dict[str, numpy.ndarray]
            See class docstring for keys and types.
        """
        self._debug("get_arrays: loading datasetdict")
        ds = load_datasetdict(self.spec)
        self._debug("get_arrays: datasetdict loaded")
        train_ds = merge_train_splits(ds, self.spec)
        test_ds = ds[self.spec.test_split]
        self._debug(
            f"get_arrays: train split size={len(train_ds)}, "
            f"test split size={len(test_ds)}"
        )
        self._debug("get_arrays: fitting label encoder")
        self._fit_label_encoder(train_ds=train_ds, test_ds=test_ds)
        self._debug("get_arrays: label encoder fitted")
        self._debug("get_arrays: fitting embedder")
        self._fit_embedder(train_ds)
        self._debug("get_arrays: embedder fitted")

        self._debug("get_arrays: resolving train arrays")
        X_train, y_train, z_train = self._compute_or_load_split(
            train_ds, split_name="train"
        )
        self._debug("get_arrays: train arrays ready")
        self._debug("get_arrays: resolving test arrays")
        X_test, y_test, _ = self._compute_or_load_split(
            test_ds, split_name="test"
        )
        self._debug("get_arrays: test arrays ready")

        out = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        if z_train is not None:
            out["z_train"] = z_train
        return out

    def _cache_paths(
        self, *, split_name: str, ds_fingerprint: str
    ) -> Tuple[Path, Path]:
        """
        Compute the cache paths for ``X`` and metadata.

        Returns
        -------
        X_path : pathlib.Path
            Path to `.npy` file storing the embeddings (memmap).
        meta_path : pathlib.Path
            Path to `.npz` file storing labels (``y``) and optional ``z``.
        """
        cache_root = Path(self.cfg.cache_dir) / "embeddings"
        cache_root.mkdir(parents=True, exist_ok=True)

        self._debug(
            f"_cache_paths[{split_name}]: computing spec fingerprint "
            f"(local_signature={self.spec.local_signature})"
        )
        payload = {
            "spec": spec_fingerprint(self.spec),
            "split": split_name,
            "ds_fingerprint": ds_fingerprint,
            "embedder": self.embedder.fingerprint(),
            "batch_size": self.cfg.batch_size,
            "memmap_dtype": str(np.dtype(self.cfg.memmap_dtype)),
        }
        key = sha1_json(payload)
        self._debug(f"_cache_paths[{split_name}]: key={key}")
        return (
            cache_root / f"{key}.X.npy",
            cache_root / f"{key}.meta.npz",
        )

    def _compute_or_load_split(
        self, split_ds, *, split_name: str
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load a split from cache or compute it and write to cache.

        Returns
        -------
        X : numpy.ndarray
            Memmap array backed by a `.npy` file.
        y : numpy.ndarray
            In-memory integer label array.
        z : numpy.ndarray or None
            Optional in-memory array (e.g., annotator labels).
        """
        ds_fpr = getattr(split_ds, "_fingerprint", "unknown")
        self._debug(f"_compute_or_load_split[{split_name}]: resolving cache paths")
        X_path, meta_path = self._cache_paths(
            split_name=split_name, ds_fingerprint=ds_fpr
        )
        self._debug(
            f"_compute_or_load_split[{split_name}]: "
            f"X_path={X_path} meta_path={meta_path}"
        )

        # Reuse cached arrays if allowed and present.
        if self.cfg.reuse_cache and X_path.exists() and meta_path.exists():
            self._debug(f"_compute_or_load_split[{split_name}]: cache hit")
            X = np.load(X_path, mmap_mode=self.cfg.mmap_mode)
            meta = npz_load(meta_path)
            return X, meta["y"], meta.get("z")
        self._debug(f"_compute_or_load_split[{split_name}]: cache miss")

        # Resample audio datasets to embedder SR (e.g., 16k for wav2vec2).
        split_ds = self._maybe_cast_audio_sampling_rate(split_ds)

        # Compute and stream embeddings to memmap.
        self._debug(f"_compute_or_load_split[{split_name}]: embedding split")
        X, y, z = self._embed_split_to_memmap(split_ds, X_path)

        # Save small metadata (labels) separately in a compact `.npz`.
        if z is None:
            npz_save(meta_path, y=y)
        else:
            npz_save(meta_path, y=y, z=z)
        self._debug(f"_compute_or_load_split[{split_name}]: wrote cache files")

        return X, y, z

    def _extract_yz(self, split_ds) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract ``y`` and optional ``z`` from the dataset split.

        Returns
        -------
        y : numpy.ndarray
            Integer labels (in-memory).
        z : numpy.ndarray or None
            Optional integer array (in-memory).
        """
        if self.label_encoder_ is None:
            if len(split_ds) == 0:
                y = np.empty((0,), dtype=np.int64)
            else:
                raise RuntimeError(
                    "Label encoder has not been initialized. Call "
                    "get_arrays() before extracting split labels."
                )
        else:
            y = self.label_encoder_.transform(split_ds[self.spec.y_key])

        z = None
        if (
            self.spec.z_key is not None
            and self.spec.z_key in split_ds.column_names
        ):
            z = stack_fixed(split_ds[self.spec.z_key], dtype=np.int64)

        return y, z

    def _is_hf_audio_column(self, ds) -> bool:
        """Return True if ds.features[x_key] is an Audio feature."""
        if self._uses_multi_column_x():
            return False
        try:
            from datasets import Audio as HFAudio  # newer alias
        except Exception:
            try:
                from datasets.features import (
                    Audio as HFAudio,
                )  # older location
            except Exception:
                return False

        feat = getattr(ds, "features", {}).get(self.spec.x_key, None)
        return isinstance(feat, HFAudio)

    def _maybe_cast_audio_sampling_rate(self, ds):
        """
        If this is an HF Audio column and the embedder has a sampling_rate,
        resample at dataset decode time via cast_column.
        """
        if not self._is_hf_audio_column(ds):
            return ds

        target_sr = getattr(self.embedder, "sampling_rate", None)
        if target_sr is None:
            return ds  # no hint, no resampling

        try:
            from datasets import Audio as HFAudio
        except Exception:
            from datasets.features import Audio as HFAudio

        return ds.cast_column(
            self.spec.x_key, HFAudio(sampling_rate=int(target_sr))
        )

    def _uses_multi_column_x(self) -> bool:
        return not isinstance(self.spec.x_key, str)

    def _x_key_names(self) -> tuple[str, ...]:
        if isinstance(self.spec.x_key, str):
            return (self.spec.x_key,)
        return tuple(self.spec.x_key)

    def _extract_x_values(self, batch_or_split):
        if self._uses_multi_column_x():
            return {
                name: batch_or_split[name] for name in self._x_key_names()
            }
        return batch_or_split[self.spec.x_key]

    def _fit_label_encoder(self, *, train_ds, test_ds) -> None:
        feature = getattr(train_ds, "features", {}).get(self.spec.y_key, None)
        label_values = infer_label_values(
            train_ds[self.spec.y_key],
            test_ds[self.spec.y_key],
            feature=feature,
        )
        self.label_values_ = tuple(label_values)
        if len(self.label_values_) == 0:
            self.label_encoder_ = None
            return

        self.label_encoder_ = ExtLabelEncoder(
            classes=self.label_values_,
            missing_label=None,
        )
        self.label_encoder_.fit(train_ds[self.spec.y_key])

    def _fit_embedder(self, train_ds) -> None:
        fit_fn = getattr(self.embedder, "fit", None)
        if not callable(fit_fn):
            return
        if len(train_ds) == 0:
            return

        train_ds = self._maybe_cast_audio_sampling_rate(train_ds)
        raw_x = self._extract_x_values(train_ds)
        X_fit = self._prepare_x(raw_x)
        fit_fn(X_fit)

    def _prepare_x(self, batch_x):
        """
        Convert HF batch values into the input expected by the embedder.
        - tabular: dict[str, list[number]] -> ndarray (N, D)
        - text: list[str] -> unchanged
        - images: list[PIL/np] -> unchanged
        - audio: list[{"array":..., "sampling_rate":...}] -> list[np.ndarray 1D float32]
        """
        if self.x_adapter is not None:
            return self.x_adapter(batch_x)

        if self._uses_multi_column_x():
            return stack_tabular_columns(
                batch_x,
                column_order=self._x_key_names(),
                dtype=np.float32,
            )

        # numpy object array -> list
        if isinstance(batch_x, np.ndarray) and batch_x.dtype == object:
            batch_x = batch_x.tolist()

        # text
        if isinstance(batch_x, list) and (
            len(batch_x) == 0 or isinstance(batch_x[0], str)
        ):
            return batch_x

        # HF AudioDecoder objects (datasets 4.x): list of decoders
        if (
            isinstance(batch_x, list)
            and len(batch_x) > 0
            and hasattr(batch_x[0], "get_all_samples")
        ):
            out = []
            for dec in batch_x:
                # Old syntax still works in v4.x (nice of them to not break *everything*)
                try:
                    arr = dec["array"]  # numpy (usually)
                    # sr  = dec["sampling_rate"]      # available if you want to assert
                except Exception:
                    samples = dec.get_all_samples()  # torchcodec AudioSamples
                    data = samples.data
                    arr = (
                        data.detach().cpu().numpy()
                        if hasattr(data, "detach")
                        else np.asarray(data)
                    )

                arr = np.asarray(arr, dtype=np.float32)

                # Torchcodec often returns (C, T); downmix to mono
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)

                out.append(arr.reshape(-1))
            return out

        # HF Audio decoded as dicts (older behavior / some setups)
        if (
            isinstance(batch_x, list)
            and len(batch_x) > 0
            and isinstance(batch_x[0], dict)
            and "array" in batch_x[0]
        ):
            return [
                np.asarray(ex["array"], dtype=np.float32).reshape(-1)
                for ex in batch_x
            ]

        if isinstance(batch_x, dict) and "array" in batch_x:
            arrays = batch_x["array"]
            if isinstance(arrays, np.ndarray) and arrays.dtype == object:
                arrays = arrays.tolist()
            return [
                np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays
            ]

        # Images and other modalities: pass through
        return batch_x

    def _embed_split_to_memmap(
        self, split_ds, X_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Embed a dataset split and stream results into a `.npy` memmap.

        Parameters
        ----------
        split_ds : datasets.Dataset
            HuggingFace dataset split (or compatible object supporting
            slicing).
        X_path : pathlib.Path
            Path to the output `.npy` file for embeddings.

        Returns
        -------
        X : numpy.ndarray
            Memmap array backed by `X_path`.
        y : numpy.ndarray
            In-memory labels.
        z : numpy.ndarray or None
            In-memory optional array.

        Raises
        ------
        ValueError
            If the embedder output shape is not constant across batches.
        """
        n = len(split_ds)
        self._debug(f"_embed_split_to_memmap: start n={n} X_path={X_path}")
        y, z = self._extract_yz(split_ds)
        bs = int(self.cfg.batch_size)

        # Handle empty splits by creating an empty `.npy`.
        if n == 0:
            mm = np.lib.format.open_memmap(
                X_path,
                mode="w+",
                dtype=np.dtype(self.cfg.memmap_dtype),
                shape=(0, 0),
            )
            mm.flush()
            return mm, y, z

        # Probe first batch to infer the per-sample embedding shape.
        # This is necessary to preallocate the memmap with the final shape.
        first_batch = split_ds[0 : min(bs, n)]
        X_in0 = self._prepare_x(self._extract_x_values(first_batch))

        X0 = np.asarray(self.embedder.embed(X_in0), dtype=np.float32)
        per_sample_shape = X0.shape[1:]  # (D,) or (T, D) or (C, H, W), etc.
        self._debug(
            "_embed_split_to_memmap: inferred per-sample shape="
            f"{per_sample_shape} from first batch size={len(X0)}"
        )

        # Preallocate disk-backed array: shape = (N, *per_sample_shape).
        mm = np.lib.format.open_memmap(
            X_path,
            mode="w+",
            dtype=np.dtype(self.cfg.memmap_dtype),
            shape=(n, *per_sample_shape),
        )

        # Write the probe batch.
        mm[0 : len(X0)] = X0.astype(mm.dtype, copy=False)

        # Stream the rest in batches.
        for start in range(len(X0), n, bs):
            end = min(start + bs, n)
            batch = split_ds[start:end]
            X_in = self._prepare_x(self._extract_x_values(batch))

            Xb = np.asarray(self.embedder.embed(X_in), dtype=np.float32)

            # Enforce constant embedding shape (required for memmap).
            if Xb.shape[1:] != per_sample_shape:
                raise ValueError(
                    "Embedding output shape is not constant across images, "
                    "cannot memmap. Expected per-sample shape "
                    f"{per_sample_shape}, got {Xb.shape[1:]} for batch "
                    f"{start}:{end}."
                )

            mm[start:end] = Xb.astype(mm.dtype, copy=False)

        mm.flush()
        self._debug("_embed_split_to_memmap: finished writing memmap")
        return mm, y, z
