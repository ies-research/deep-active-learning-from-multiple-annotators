from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


class BaseRatioScheduler(ABC):
    """Base class for cycle-dependent ratio scheduling.

    This provides a lightweight, sklearn-conform callable interface to resolve
    a ratio value (e.g., assignments-per-sample) as a function of the active
    learning cycle index.

    Parameters
    ----------
    n_cycles : int, optional
        Total number of cycles. If provided, ``__call__`` validates that
        ``cycle`` is within ``[0, n_cycles - 1]``.

    Notes
    -----
    - Subclasses implement ``_value`` and should be *stateless* w.r.t. cycle,
      i.e., calling with the same ``cycle`` should return the same value.
    - Values are returned as ``float`` to allow fractional ratios.

    Examples
    --------
    >>> sched = StepRatioScheduler(
    ...     default=1.0,
    ...     schedule=[{"start": 0, "end": 4, "value": 2.0}],
    ...     n_cycles=10,
    ... )
    >>> sched(0), sched(5)
    (2.0, 1.0)
    """

    def __init__(self, *, n_cycles: Optional[int] = None):
        self.n_cycles = None if n_cycles is None else n_cycles
        if self.n_cycles is not None and self.n_cycles <= 0:
            raise ValueError(f"n_cycles must be > 0, got {self.n_cycles}.")

    def __call__(self, cycle: int) -> float:
        """Return the scheduled ratio for a given cycle.

        Parameters
        ----------
        cycle : int
            Zero-based cycle index.

        Returns
        -------
        ratio : float
            Scheduled ratio for the given cycle.

        Raises
        ------
        ValueError
            If ``n_cycles`` is provided and ``cycle`` is out of range.
        """
        cycle = cycle
        if self.n_cycles is not None and not (0 <= cycle < self.n_cycles):
            raise ValueError(f"cycle={cycle} out of range for n_cycles={self.n_cycles}.")
        return float(self._value(cycle))

    @abstractmethod
    def _value(self, cycle: int) -> float:
        """Subclass hook to compute the ratio for ``cycle``."""
        raise NotImplementedError


