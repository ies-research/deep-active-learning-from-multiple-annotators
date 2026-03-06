from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


BootstrapMaskFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


class BootstrappedKLDivLoss(nn.KLDivLoss):
    """
    KL-divergence loss for classification with optional multi-head predictions
    and bootstrap masking.

    Compared with plain `nn.KLDivLoss`, this criterion is tailored to
    classification-style log-probabilities and sums out the class dimension
    before reduction.

    Supported input shapes
    ----------------------
    - input: (N, C)
    - input: (H, N, C)  # H = number of heads

    Supported target shapes
    -----------------------
    - target: (N,)          hard labels / class indices
    - target: (N, C)        soft labels / probabilities
    - target: (H, N, C)     per-head soft labels

    Bootstrapping
    -------------
    `bootstrap` can be:
    - None:
        No extra masking.
    - float in [0, 1]:
        Fraction of valid items to mask uniformly at random per forward call.
        Example: `bootstrap=0.25` masks 25% of valid items.
    - callable:
        A function
            f(input, soft_target, valid_mask) -> mask
        returning a tensor broadcastable to `input.shape[:-1]`.
        The mask is multiplied with `valid_mask`.

    Reduction semantics
    -------------------
    Let `item_loss` denote the KL divergence summed over the class dimension:
    - reduction="none":      returns item_loss with shape (N,) or (H, N)
    - reduction="sum":       returns masked sum of item losses
    - reduction="mean":      returns masked mean over items
    - reduction="batchmean": alias of "mean" in this classification-oriented
                             implementation

    Notes
    -----
    - `input` must be log-probabilities.
    - If `target` contains class indices, `log_target=True` is not supported.
    - For one-hot targets, this reduces to the usual negative log-likelihood.
    - For soft targets (e.g. MixUp), this minimizes KL(target || prediction).

    Parameters
    ----------
    reduction : {"none", "mean", "sum", "batchmean"}, default="batchmean"
        Reduction applied after summing over the class dimension.
    log_target : bool, default=False
        Whether `target` is already in log-space. Only supported for soft
        targets, not for integer class labels.
    ignore_index : int, default=-100
        Target value to ignore when `target` contains class indices.
    eps : float, default=1e-12
        Small constant to avoid division by zero in masked reductions.
    bootstrap : None or float or callable, default=None
        Bootstrap masking specification used by default in `forward`.
    """

    def __init__(
        self,
        reduction: str = "batchmean",
        log_target: bool = False,
        ignore_index: int = -100,
        eps: float = 1e-12,
        bootstrap: float | BootstrapMaskFn | None = 0.25,
    ):
        super().__init__(reduction="none", log_target=log_target)
        if reduction not in {"none", "mean", "sum", "batchmean"}:
            raise ValueError(
                "`reduction` must be one of "
                "{'none', 'mean', 'sum', 'batchmean'}."
            )
        self._item_reduction = reduction
        self.ignore_index = ignore_index
        self.eps = eps
        self.bootstrap = bootstrap

    def _to_soft_target(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert target to a soft target matching `input` shape.

        Returns
        -------
        soft_target : torch.Tensor
            Probability targets if `log_target=False`, log-probability targets
            if `log_target=True`.
        valid_mask : torch.Tensor
            Float tensor with shape equal to the item shape, i.e.
            `input.shape[:-1]`.
        """
        item_shape = input.shape[:-1]
        n_classes = input.shape[-1]
        device = input.device
        dtype = input.dtype

        # ------------------------------------------------------------
        # 1) SOFT TARGETS
        # ------------------------------------------------------------

        # input: (H, N, C), target: (N, C) -> broadcast to (H, N, C)
        if input.ndim == 3 and target.ndim == 2 and target.shape == input.shape[1:]:
            target = target.unsqueeze(0).expand(input.shape[0], -1, -1)
            target = target.to(device=device, dtype=dtype)
            valid_mask = torch.ones(item_shape, device=device, dtype=dtype)
            return target, valid_mask

        # input: (H, N, C), target: (H, N, C)
        if input.ndim == 3 and target.ndim == 3 and target.shape == input.shape:
            target = target.to(device=device, dtype=dtype)
            valid_mask = torch.ones(item_shape, device=device, dtype=dtype)
            return target, valid_mask

        # input: (N, C), target: (N, C)
        if input.ndim == 2 and target.ndim == 2 and target.shape == input.shape:
            target = target.to(device=device, dtype=dtype)
            valid_mask = torch.ones(item_shape, device=device, dtype=dtype)
            return target, valid_mask

        # ------------------------------------------------------------
        # 2) HARD TARGETS
        # ------------------------------------------------------------

        # input: (H, N, C), target: (N,) -> broadcast to (H, N)
        if input.ndim == 3 and target.ndim == 1 and target.shape == (input.shape[1],):
            target = target.unsqueeze(0).expand(input.shape[0], -1)

        # input: (N, C), target: (N,)
        # input: (H, N, C), target: (H, N)
        if target.ndim == len(item_shape):
            if self.log_target:
                raise ValueError(
                    "Integer / index targets are incompatible with `log_target=True`."
                )
            if target.shape != item_shape:
                raise ValueError(
                    f"Hard target shape {tuple(target.shape)} does not match "
                    f"input item shape {tuple(item_shape)}."
                )

            target = target.to(device=device)
            valid_mask = (target != self.ignore_index).to(dtype=dtype)

            safe_target = target.masked_fill(target == self.ignore_index, 0)

            soft_target = F.one_hot(
                safe_target.long(), num_classes=n_classes
            ).to(dtype=dtype)

            soft_target = soft_target * valid_mask.unsqueeze(-1)
            return soft_target, valid_mask

        raise ValueError(
            f"Unsupported target shape {tuple(target.shape)} for input shape "
            f"{tuple(input.shape)}."
        )

    def _float_bootstrap_mask(
        self,
        valid_mask: torch.Tensor,
        mask_fraction: float,
    ) -> torch.Tensor:
        """
        Randomly mask exactly `round(mask_fraction * n_valid)` valid items.
        """
        if not (0.0 <= mask_fraction <= 1.0):
            raise ValueError(
                f"If `bootstrap` is a float, it must lie in [0, 1], got {mask_fraction}."
            )

        out = valid_mask.clone()
        flat_out = out.reshape(-1)
        flat_valid = valid_mask.reshape(-1) > 0

        valid_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
        n_valid = int(valid_idx.numel())

        if n_valid == 0 or mask_fraction == 0.0:
            return out

        n_mask = int(round(mask_fraction * n_valid))
        if n_mask == 0:
            return out
        if n_mask >= n_valid:
            flat_out[valid_idx] = 0
            return out

        perm = torch.randperm(n_valid, device=valid_mask.device)
        masked_idx = valid_idx[perm[:n_mask]]
        flat_out[masked_idx] = 0
        return out

    def _make_item_mask(
        self,
        input: torch.Tensor,
        soft_target: torch.Tensor,
        valid_mask: torch.Tensor,
        bootstrap: float | BootstrapMaskFn | None,
    ) -> torch.Tensor:
        """
        Build the final item mask of shape `input.shape[:-1]`.
        """
        item_shape = input.shape[:-1]

        if bootstrap is None:
            return valid_mask

        if isinstance(bootstrap, float):
            return self._float_bootstrap_mask(valid_mask, bootstrap)

        if callable(bootstrap):
            mask = bootstrap(input, soft_target, valid_mask)
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, device=input.device, dtype=input.dtype)
            else:
                mask = mask.to(device=input.device, dtype=input.dtype)

            try:
                mask = torch.broadcast_to(mask, item_shape)
            except RuntimeError as e:
                raise ValueError(
                    f"Bootstrap mask with shape {tuple(mask.shape)} is not "
                    f"broadcastable to item shape {tuple(item_shape)}."
                ) from e

            return valid_mask * mask

        raise TypeError(
            "`bootstrap` must be None, a float in [0, 1], or a callable "
            "returning a mask."
        )

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        bootstrap: float | BootstrapMaskFn | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Log-probabilities of shape (N, C) or (H, N, C).
        target : torch.Tensor
            Hard labels (N,), soft labels (N, C), or per-head soft labels
            (H, N, C).
        bootstrap : None or float or callable, default=None
            Overrides `self.bootstrap` for this call only.

            - float: fraction of valid items to mask
            - callable: returns a mask broadcastable to `input.shape[:-1]`

        Returns
        -------
        loss : torch.Tensor
            Scalar loss unless `reduction="none"`, in which case the returned
            shape is (N,) or (H, N).
        """
        if input.ndim not in (2, 3):
            raise ValueError(
                f"`input` must have shape (N, C) or (H, N, C), got {tuple(input.shape)}."
            )

        bootstrap = self.bootstrap if bootstrap is None else bootstrap

        soft_target, valid_mask = self._to_soft_target(input, target)

        pointwise = F.kl_div(
            input,
            soft_target,
            reduction="none",
            log_target=self.log_target,
        )
        item_loss = pointwise.sum(dim=-1)

        item_mask = self._make_item_mask(
            input=input,
            soft_target=soft_target,
            valid_mask=valid_mask,
            bootstrap=bootstrap,
        )

        item_loss = item_loss * item_mask

        if self._item_reduction == "none":
            return item_loss

        if self._item_reduction == "sum":
            return item_loss.sum()

        denom = item_mask.sum().clamp_min(self.eps)
        return item_loss.sum() / denom