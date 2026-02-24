from __future__ import annotations

from typing import Optional
import numpy as np
from ._base import BaseRatioScheduler

class CosineAnnealingRatioScheduler(BaseRatioScheduler):
    """Cosine annealing ratio scheduler with optional warm restarts.

    This scheduler produces a smooth cosine trajectory between ``start_value``
    (at the beginning of each annealing period) and ``end_value`` (at the end),
    with support for increasing or decreasing schedules depending on
    ``start_value`` and ``end_value``.

    With warm restarts enabled, the schedule restarts every period of length
    ``t_i`` (starting with ``t_0`` and optionally scaled by ``t_mult``), and
    the amplitude can be decayed by ``gamma`` each restart.

    Parameters
    ----------
    start_value : float
        Value at the start of an annealing period.
    end_value : float
        Value at the end of an annealing period.
    n_cycles : int, optional
        Total number of cycles used for validation.
    t_start : int, default=0
        Cycle index at which scheduling starts. For ``cycle < t_start``,
        the scheduler returns ``start_value``.
    t_0 : int, optional
        Initial period length for warm restarts. If ``None``, no restarts
        are used and the schedule runs once from ``t_start`` to ``t_end``.
    t_mult : float, default=1.0
        Period length multiplier after each restart. Must be >= 1.0.
        Only used if ``t_0`` is not ``None``.
    gamma : float, default=1.0
        Amplitude multiplier per restart. Must be in (0, 1] typically.
        ``gamma < 1`` reduces the distance from ``start_value`` to ``end_value``
        after each restart.
    t_end : int, optional
        End cycle (inclusive) for the non-restart case. If omitted, it defaults
        to ``n_cycles - 1`` when ``n_cycles`` is given, otherwise scheduling
        continues indefinitely.

    Returns
    -------
    ratio : float
        Scheduled ratio for a given cycle.

    Notes
    -----
    Core cosine formula (for progress p in [0, 1]):

    ``value(p) = end + 0.5 * (start - end) * (1 + cos(pi * p))``

    This ensures:
    - p=0 -> start
    - p=1 -> end

    Warm restarts:
    - period k has length ``t_k = t_0 * t_mult**k``
    - amplitude k is ``(start_value - end_value) * gamma**k``
      so the period start becomes: ``start_k = end_value + amplitude_k``

    Examples
    --------
    Decreasing (2 -> 1) without restarts:

    >>> sched = CosineAnnealingRatioScheduler(2.0, 1.0, n_cycles=10)
    >>> [round(sched(t), 3) for t in range(3)]
    [2.0, 1.97, 1.883]

    Increasing (1 -> 2) with restarts:

    >>> sched = CosineAnnealingRatioScheduler(
    ...     1.0, 2.0, n_cycles=20, t_0=5, t_mult=2.0, gamma=1.0
    ... )
    >>> float(sched(0)) <= float(sched(4))
    True
    """

    def __init__(
            self,
            start_value: float,
            end_value: float,
            *,
            n_cycles: Optional[int] = None,
            t_start: int = 0,
            t_0: Optional[int] = None,
            t_mult: float = 1.0,
            gamma: float = 1.0,
            t_end: Optional[int] = None,
    ):
        super().__init__(n_cycles=n_cycles)

        self.start_value = start_value
        self.end_value = end_value
        if self.start_value <= 0 or self.end_value <= 0:
            raise ValueError("start_value and end_value must be > 0.")

        self.t_start = t_start
        if self.t_start < 0:
            raise ValueError(f"t_start must be >= 0, got {self.t_start}.")

        self.t_0 = None if t_0 is None else t_0
        if self.t_0 is not None and self.t_0 <= 0:
            raise ValueError(f"t_0 must be > 0, got {self.t_0}.")

        self.t_mult = float(t_mult)
        if self.t_0 is not None and self.t_mult < 1.0:
            raise ValueError(f"t_mult must be >= 1.0, got {self.t_mult}.")

        self.gamma = float(gamma)
        if self.t_0 is not None and not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0,1] for restarts, got {self.gamma}.")

        if t_end is None:
            if self.t_0 is None and self.n_cycles is not None:
                self.t_end = self.n_cycles - 1
            else:
                self.t_end = None
        else:
            self.t_end = t_end
            if self.t_end < self.t_start:
                raise ValueError(f"t_end must be >= t_start, got {self.t_end} < {self.t_start}.")

    @staticmethod
    def _cosine_interp(start: float, end: float, p: float) -> float:
        p = float(np.clip(p, 0.0, 1.0))
        return end + 0.5 * (start - end) * (1.0 + float(np.cos(np.pi * p)))

    def _value(self, cycle: int) -> float:
        # Pre-start: hold start_value
        if cycle < self.t_start:
            return self.start_value

        # No restarts: one pass from t_start to t_end (or indefinitely if t_end None)
        if self.t_0 is None:
            if self.t_end is None:
                # indefinite schedule: treat progress as min(cycle - t_start, 1) with denom 1
                # (i.e., constant at end_value after first step). Better: user should pass t_end.
                # We keep it explicit and unsurprising.
                if cycle == self.t_start:
                    return self.start_value
                return self.end_value

            denom = max(1, self.t_end - self.t_start)
            p = (cycle - self.t_start) / denom
            return self._cosine_interp(self.start_value, self.end_value, p)

        # Warm restarts: determine which period we're in
        t = cycle - self.t_start
        if t < 0:
            return self.start_value

        period_len = float(self.t_0)
        k = 0
        t_in = float(t)

        # Find restart index k and time within period.
        # n_cycles is usually small, so a simple loop is fine.
        while t_in >= period_len:
            t_in -= period_len
            period_len *= self.t_mult
            k += 1

        # Decay amplitude with gamma^k
        amp0 = self.start_value - self.end_value
        amp_k = amp0 * (self.gamma ** k)
        start_k = self.end_value + amp_k

        p = t_in / max(1e-12, period_len)  # avoid divide by zero
        return self._cosine_interp(start_k, self.end_value, p)
