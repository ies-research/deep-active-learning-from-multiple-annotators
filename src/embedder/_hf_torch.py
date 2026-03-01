from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Union
from ._base import Embedder

ArrayLike1D = Union[np.ndarray, Sequence[float]]


@dataclass
class TorchHFImageEmbedder(Embedder):
    """
    HuggingFace image embedder using a PyTorch backbone.

    Loads a HuggingFace model via ``transformers.AutoModel`` and an associated
    ``AutoImageProcessor``. Given a sequence of images, it returns either pooled
    image embeddings (2D) or the full model feature representation without pooling
    (3D token sequence or 4D spatial map), depending on ``pooling``.

    Pooling:
    - "pooler":
        * If the model provides CLIP-style projected image embeddings via
          ``get_image_features(...)`` or returns ``image_embeds``, those are preferred.
        * Otherwise uses ``pooler_output`` if available.
    - "cls": first token (CLS) from token sequence outputs (3D).
    - "mean": mean over tokens (3D) or global average pool over spatial dims (4D).
    - "none": return full last_hidden_state (3D tokens or 4D map).

    Notes:
    - For CLIPModel, we avoid calling ``forward`` without text. We instead use
      ``get_image_features`` (pooler) or ``vision_model`` (tokens).
    - Outputs are always returned as ``np.float32``.
    """

    model_id: str
    revision: Optional[str] = None
    pooling: Literal["pooler", "cls", "mean", "none"] = "cls"
    include_cls_token: bool = True
    device: Optional[str] = None
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.dtype not in dtype_map:
            raise ValueError(
                f"Unknown dtype={self.dtype!r}. Expected one of {list(dtype_map)}."
            )

        self._torch = torch
        self._torch_dtype = dtype_map[self.dtype]

        self._processor = AutoImageProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            use_fast=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
        )
        self._model.to(self.device).eval()

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "type": "torch_hf_image",
            "model_id": self.model_id,
            "revision": self.revision,
            "pooling": self.pooling,
            "include_cls_token": self.include_cls_token,
            "dtype": self.dtype,
        }

    def _move_and_cast_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        torch = self._torch

        out: Dict[str, Any] = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                out[k] = v.to(self.device)
            else:
                out[k] = v

        # Cast floating tensors on accelerators to requested dtype.
        if self.device != "cpu":
            for k, v in list(out.items()):
                if hasattr(v, "is_floating_point") and v.is_floating_point():
                    out[k] = v.to(dtype=self._torch_dtype)
        else:
            # CPU: keep float32 for stability
            for k, v in list(out.items()):
                if hasattr(v, "is_floating_point") and v.is_floating_point():
                    out[k] = v.to(dtype=torch.float32)

        return out

    def _run_image_forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the model in an image-only safe way and returns a normalized view:
          - h: token sequence (3D) or spatial map (4D), or None
          - pooler: pooled (pre-projection) output if available, or None
          - image_embeds: projected image embedding if available, or None

        This avoids CLIPModel complaining about missing input_ids by using
        get_image_features / vision_model where appropriate.
        """
        m = self._model

        pixel_values = inputs.get("pixel_values", None)
        if pixel_values is None:
            # Fallback: just call the model normally
            out = m(**inputs)
            return self._unwrap_outputs(out)

        # If model exposes CLIP-style get_image_features, prefer that for pooler mode.
        # This avoids needing input_ids.
        if self.pooling == "pooler" and hasattr(m, "get_image_features"):
            img_emb = m.get_image_features(pixel_values=pixel_values)
            return {"h": None, "pooler": None, "image_embeds": img_emb}

        # If model has a dedicated vision submodule, use it to get tokens/pooler.
        # (CLIPModel has vision_model)
        if hasattr(m, "vision_model"):
            vision_out = m.vision_model(
                pixel_values=pixel_values, return_dict=True
            )
            h = getattr(vision_out, "last_hidden_state", None)
            pooler = getattr(vision_out, "pooler_output", None)
            return {"h": h, "pooler": pooler, "image_embeds": None}

        # Otherwise, try regular forward (most pure vision models accept pixel_values only).
        out = m(**inputs)
        return self._unwrap_outputs(out)

    def _unwrap_outputs(self, out: Any) -> Dict[str, Any]:
        """
        Normalize different HF model output structures to:
          - h: torch.Tensor or None (token sequence 3D or spatial map 4D)
          - pooler: torch.Tensor or None
          - image_embeds: torch.Tensor or None (CLIP-like projected image features)
        """
        image_embeds = getattr(out, "image_embeds", None)

        # Some VLMs store vision tokens in vision_model_output.
        vision_out = getattr(out, "vision_model_output", None)
        if vision_out is not None:
            h = getattr(vision_out, "last_hidden_state", None)
            pooler = getattr(vision_out, "pooler_output", None)
        else:
            h = getattr(out, "last_hidden_state", None)
            pooler = getattr(out, "pooler_output", None)

        return {"h": h, "pooler": pooler, "image_embeds": image_embeds}

    def embed(self, x: Sequence[Any]) -> np.ndarray:
        torch = self._torch
        imgs = list(x)

        with torch.no_grad():
            inputs = self._processor(images=imgs, return_tensors="pt")
            inputs = self._move_and_cast_inputs(inputs)

            u = self._run_image_forward(inputs)

            h = u["h"]
            pooler = u["pooler"]
            image_embeds = u["image_embeds"]

            if self.pooling == "pooler":
                if image_embeds is not None:
                    emb = image_embeds
                elif pooler is not None:
                    emb = pooler
                else:
                    raise ValueError(
                        "pooling='pooler' requested but no projected embedding "
                        "('image_embeds' / get_image_features) and no 'pooler_output' available."
                    )

            else:
                if h is None:
                    raise ValueError(
                        "Model output has no usable last_hidden_state for token/spatial pooling. "
                        "For CLIP-like models use pooling='pooler' (projected image embeddings)."
                    )

                if self.pooling == "none":
                    if (not self.include_cls_token) and getattr(
                        h, "ndim", None
                    ) == 3:
                        emb = h[:, 1:, :]
                    else:
                        emb = h

                elif h.ndim == 3:
                    # Token sequence: (B, T, D)
                    if self.pooling == "mean":
                        emb = h.mean(dim=1)
                    elif self.pooling == "cls":
                        emb = h[:, 0, :]
                    else:
                        raise ValueError(
                            f"pooling={self.pooling!r} not supported for token outputs. "
                            "Use 'cls', 'mean', 'none', or 'pooler' (if available)."
                        )

                elif h.ndim == 4:
                    # Spatial map: (B, C, H, W)
                    if self.pooling == "mean":
                        emb = h.mean(dim=(2, 3))
                    elif self.pooling == "none":
                        emb = h
                    else:
                        raise ValueError(
                            f"pooling={self.pooling!r} not supported for spatial outputs. "
                            "Use 'mean', 'none', or 'pooler' (if available)."
                        )
                else:
                    raise ValueError(
                        f"Unsupported last_hidden_state ndim={h.ndim}."
                    )

            return (
                emb.detach()
                .cpu()
                .float()
                .numpy()
                .astype(np.float32, copy=False)
            )


@dataclass
class TorchHFTextEmbedder:
    """
    HuggingFace text embedder using a PyTorch backbone.

    This embedder loads a HuggingFace text model via ``transformers.AutoModel``
    and an associated ``AutoTokenizer``. Given a sequence of texts, it returns
    either pooled text embeddings (2D) or the full token-level representation
    without pooling (3D), depending on ``pooling``.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g.,
        ``"sentence-transformers/all-MiniLM-L6-v2"`` or
        ``"bert-base-uncased"``).
    revision : str or None, default=None
        Optional model revision (branch, tag, or commit hash).
    pooling : {"pooler", "cls", "mean", "none"}, default="cls"
        Pooling strategy to obtain a fixed-size embedding.

        - ``"pooler"``: Use ``out.pooler_output`` if available.
        - ``"cls"``: Use the first token from ``out.last_hidden_state``.
        - ``"mean"``: Attention-mask-aware mean pooling over tokens.
        - ``"none"``: Return the full ``out.last_hidden_state`` without pooling
          (token sequence).
    include_cls_token : bool, default=True
        Only relevant when ``pooling="none"``. If False, the first token is
        dropped (often CLS / <s>), returning only remaining tokens.
    device : str or None, default=None
        Device used for inference. If None, selects ``"cuda"`` if available,
        otherwise ``"cpu"``.
    dtype : {"float32", "float16", "bfloat16"}, default="float32"
        Model weight dtype used on non-CPU devices. Outputs are always returned
        as ``np.float32``.
    max_length : int or None, default=None
        Optional max sequence length for tokenization truncation.
    cache_dir : str or None, default=None
        Optional HuggingFace cache directory.

    Attributes
    ----------
    _tokenizer
        HuggingFace tokenizer created by ``AutoTokenizer``.
    _model
        HuggingFace model created by ``AutoModel`` in evaluation mode.
    _torch
        Imported ``torch`` module (cached for use in methods).

    Notes
    -----
    - Inference is performed under ``torch.no_grad()`` and the model is set to
      evaluation mode.
    - The return type is always a NumPy array of dtype ``np.float32``.
    - Output shapes:
      - pooled modes (``"pooler"``, ``"cls"``, ``"mean"``): typically
        ``(n_texts, hidden_dim)``
      - ``pooling="none"``: ``(n_texts, seq_len, hidden_dim)``

    See Also
    --------
    transformers.AutoTokenizer
    transformers.AutoModel
    """

    model_id: str
    revision: Optional[str] = None
    pooling: Literal["pooler", "cls", "mean", "none"] = "cls"
    include_cls_token: bool = True
    device: Optional[str] = None
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    max_length: Optional[int] = None
    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self._torch = torch
        self._torch_dtype = dtype_map[self.dtype]

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            use_fast=True,
        )

        # Load weights in requested dtype on accelerators.
        model_kwargs: Dict[str, Any] = dict(
            revision=self.revision,
            cache_dir=self.cache_dir,
        )
        if self.device != "cpu" and self.dtype != "float32":
            model_kwargs["torch_dtype"] = self._torch_dtype

        self._model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        self._model.to(self.device).eval()

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "type": "torch_hf_text",
            "model_id": self.model_id,
            "revision": self.revision,
            "pooling": self.pooling,
            "include_cls_token": self.include_cls_token,
            "dtype": self.dtype,
            "max_length": self.max_length,
        }

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch

        with torch.no_grad():
            tok = self._tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}

            out = self._model(**tok)

            # Prefer model-provided pooler_output if requested and available.
            if (
                self.pooling == "pooler"
                and getattr(out, "pooler_output", None) is not None
            ):
                emb = out.pooler_output
            else:
                h = out.last_hidden_state  # (B, T, D)

                if h.ndim != 3:
                    raise ValueError(
                        f"Unsupported last_hidden_state ndim={h.ndim}; "
                        "expected a token sequence (3D)."
                    )

                if self.pooling == "none":
                    emb = h[:, 1:, :] if (not self.include_cls_token) else h

                elif self.pooling == "cls":
                    emb = h[:, 0, :]

                elif self.pooling == "mean":
                    # Attention-mask-aware mean pooling (ignore padding).
                    attn = tok.get("attention_mask", None)
                    if attn is None:
                        # Fallback: naive mean over tokens.
                        emb = h.mean(dim=1)
                    else:
                        mask = attn.unsqueeze(-1).to(h.dtype)  # (B, T, 1)
                        summed = (h * mask).sum(dim=1)  # (B, D)
                        denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
                        emb = summed / denom

                else:
                    raise ValueError(
                        f"pooling={self.pooling!r} not supported. "
                        "Use 'pooler', 'cls', 'mean', or 'none'."
                    )

            return (
                emb.detach()
                .cpu()
                .float()
                .numpy()
                .astype(np.float32, copy=False)
            )


@dataclass
class TorchHFAudioEmbedder:
    """
    HuggingFace audio embedder using a PyTorch backbone.

    This embedder loads a HuggingFace audio model via ``transformers.AutoModel``
    and an associated audio preprocessor via ``transformers.AutoFeatureExtractor``
    (or ``AutoProcessor`` as a fallback). Given a sequence of audio waveforms,
    it returns either pooled embeddings (2D) or the full frame-level
    representation (3D), depending on ``pooling``.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g.,
        ``"facebook/wav2vec2-base"``, ``"facebook/hubert-base-ls960"``,
        ``"microsoft/wavlm-base-plus"``, ``"MIT/ast-finetuned-audioset-10-10-0"``).
    revision : str or None, default=None
        Optional model revision (branch, tag, or commit hash).
    pooling : {"mean", "cls", "none"}, default="mean"
        Pooling strategy to obtain a fixed-size embedding.

        - ``"mean"``: Mask-aware mean pooling over time frames.
        - ``"cls"``: Use first frame / token from ``out.last_hidden_state``.
          (Works reasonably for many encoders, but it's not universally meaningful.)
        - ``"none"``: Return the full ``out.last_hidden_state`` without pooling
          (frame sequence).
    device : str or None, default=None
        Device used for inference. If None, selects ``"cuda"`` if available,
        otherwise ``"cpu"``.
    dtype : {"float32", "float16", "bfloat16"}, default="float32"
        Model weight dtype used on non-CPU devices. Outputs are always returned
        as ``np.float32``.
    sampling_rate : int, default=16000
        Sampling rate of provided waveforms. If your audio is not at this rate,
        resample before calling ``embed``.
    max_length_seconds : float or None, default=None
        Optional maximum audio length in seconds. If provided, inputs are
        truncated to this duration.
    cache_dir : str or None, default=None
        Optional HuggingFace cache directory.

    Attributes
    ----------
    _processor
        HuggingFace audio feature extractor / processor.
    _model
        HuggingFace model created by ``AutoModel`` in evaluation mode.
    _torch
        Imported ``torch`` module (cached for use in methods).

    Notes
    -----
    - Inference is performed under ``torch.no_grad()`` and the model is set to
      evaluation mode.
    - The return type is always a NumPy array of dtype ``np.float32``.
    - Output shapes:
      - pooled modes (``"mean"``, ``"cls"``): typically ``(n_audio, hidden_dim)``
      - ``pooling="none"``: ``(n_audio, n_frames, hidden_dim)``
    """

    model_id: str
    revision: Optional[str] = None
    pooling: Literal["mean", "cls", "none"] = "mean"
    device: Optional[str] = None
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    sampling_rate: int = 16000
    max_length_seconds: Optional[float] = None
    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self._torch = torch
        self._torch_dtype = dtype_map[self.dtype]

        # Try feature extractor first (typical for wav2vec2/huBERT/WavLM/AST/BEATs).
        # Fall back to AutoProcessor if needed.
        from transformers import AutoFeatureExtractor

        self._processor = None
        try:
            self._processor = AutoFeatureExtractor.from_pretrained(
                self.model_id,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        except Exception:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )

        model_kwargs: Dict[str, Any] = dict(
            revision=self.revision,
            cache_dir=self.cache_dir,
        )
        if self.device != "cpu" and self.dtype != "float32":
            model_kwargs["torch_dtype"] = self._torch_dtype

        self._model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        self._model.to(self.device).eval()

        if self.pooling not in {"mean", "cls", "none"}:
            raise ValueError(
                f"pooling={self.pooling!r} not supported. Use 'mean', 'cls', or 'none'."
            )

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "type": "torch_hf_audio",
            "model_id": self.model_id,
            "revision": self.revision,
            "pooling": self.pooling,
            "dtype": self.dtype,
            "sampling_rate": self.sampling_rate,
            "max_length_seconds": self.max_length_seconds,
        }

    def _to_1d_float32(self, x: ArrayLike1D) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(
                f"Each waveform must be 1D (mono). Got shape={arr.shape}."
            )
        if not np.isfinite(arr).all():
            raise ValueError("Waveform contains NaN/Inf values.")
        return arr

    def embed(self, waveforms: Sequence[ArrayLike1D]) -> np.ndarray:
        """
        Compute embeddings for a batch of mono waveforms.

        Parameters
        ----------
        waveforms : sequence of 1D arrays
            Each element is a mono waveform (float array). Must be sampled at
            ``self.sampling_rate``. Resample outside this class if needed.
        """
        torch = self._torch

        xs = [self._to_1d_float32(w) for w in waveforms]

        if self.max_length_seconds is not None:
            max_len = int(round(self.max_length_seconds * self.sampling_rate))
            xs = [x[:max_len] for x in xs]

        # Processor typically expects: raw waveforms + sampling_rate
        with torch.no_grad():
            proc = self._processor(
                xs,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
                truncation=False,  # we do truncation above for consistency
            )

            # Move tensors to device
            proc = {k: v.to(self.device) for k, v in proc.items()}

            out = self._model(**proc)

            h = getattr(out, "last_hidden_state", None)
            if h is None:
                # Some models might expose different fields; try common fallbacks
                h = getattr(out, "extract_features", None)

            if h is None:
                raise ValueError(
                    "Model output has no 'last_hidden_state' (or 'extract_features'). "
                    "This model might not be a plain encoder; consider a different model "
                    "or use AutoModelForAudioClassification."
                )

            if h.ndim != 3:
                raise ValueError(
                    f"Unsupported hidden representation ndim={h.ndim}; expected 3D (B, T, D)."
                )

            if self.pooling == "none":
                emb = h

            elif self.pooling == "cls":
                emb = h[:, 0, :]

            elif self.pooling == "mean":
                attn = proc.get("attention_mask", None)

                if attn is None:
                    emb = h.mean(dim=1)
                else:
                    # h: (B, T_feat, D)
                    B, T_feat, D = h.shape

                    # Case 1: mask already aligned to frames (some processors do this)
                    if attn.shape[1] == T_feat:
                        frame_mask = attn.to(dtype=h.dtype)  # (B, T_feat)
                    else:
                        # Case 2: mask is sample-level (B, T_samples). Convert lengths -> frame lengths.
                        input_lengths = attn.long().sum(dim=-1)  # (B,)

                        # Try to find the helper on the model (different wrappers expose it differently)
                        get_len = getattr(
                            self._model,
                            "_get_feat_extract_output_lengths",
                            None,
                        )
                        if get_len is None:
                            # Some wrappers keep the base model under an attribute (rare with AutoModel, but cheap to try)
                            for attr in (
                                "wav2vec2",
                                "hubert",
                                "wavlm",
                                "model",
                            ):
                                base = getattr(self._model, attr, None)
                                get_len = (
                                    getattr(
                                        base,
                                        "_get_feat_extract_output_lengths",
                                        None,
                                    )
                                    if base is not None
                                    else None
                                )
                                if get_len is not None:
                                    break

                        if get_len is None:
                            raise ValueError(
                                "Got sample-level attention_mask that does not match hidden frame length, "
                                "and the model does not expose _get_feat_extract_output_lengths(). "
                                "Cannot do mask-aware pooling safely."
                            )

                        feat_lengths = get_len(input_lengths).to(
                            device=h.device
                        )  # (B,)

                        # Build frame mask (B, T_feat)
                        t = torch.arange(T_feat, device=h.device)[
                            None, :
                        ]  # (1, T_feat)
                        frame_mask = (t < feat_lengths[:, None]).to(
                            dtype=h.dtype
                        )  # (B, T_feat)

                    # Mask-aware mean over frames
                    mask = frame_mask.unsqueeze(-1)  # (B, T_feat, 1)
                    summed = (h * mask).sum(dim=1)  # (B, D)
                    denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
                    emb = summed / denom
            else:
                raise RuntimeError("Unreachable pooling branch.")

            return (
                emb.detach()
                .cpu()
                .float()
                .numpy()
                .astype(np.float32, copy=False)
            )
