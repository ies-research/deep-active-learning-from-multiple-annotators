from __future__ import annotations

from ._base import BaseRatioScheduler
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class _StepSegment:
    start: int
    end: int
    value: float


class StepRatioScheduler(BaseRatioScheduler):
    """Piecewise-constant ratio scheduler.

    Parameters
    ----------
    default : float, default=1.0
        Default ratio used if no segment matches the cycle.
    schedule : sequence of dict, optional
        Sequence of segments. Each segment is a mapping with keys:
        ``start`` (int, inclusive), ``end`` (int, inclusive), and ``value`` (float).
        Missing ``start`` defaults to 0. Missing ``end`` defaults to
        ``n_cycles - 1`` if ``n_cycles`` is provided, otherwise to a large number.
    n_cycles : int, optional
        Total number of cycles used for validation and default end handling.

    Notes
    -----
    - Segments are matched in the given order; on overlap, the first match wins.
    - Values may be fractional.

    Examples
    --------
    >>> sched = StepRatioScheduler(
    ...     default=1.0,
    ...     schedule=[{"start": 0, "end": 9, "value": 2.0}],
    ...     n_cycles=25,
    ... )
    >>> sched(0), sched(10)
    (2.0, 1.0)
    """

    def __init__(
            self,
            *,
            default: float = 1.0,
            schedule: Optional[Sequence[Dict[str, Any]]] = None,
            n_cycles: Optional[int] = None,
    ):
        super().__init__(n_cycles=n_cycles)
        self.default = default
        if self.default <= 0:
            raise ValueError(f"default must be > 0, got {self.default}.")

        segs: List[_StepSegment] = []
        if schedule is not None:
            for seg in schedule:
                if not isinstance(seg, dict):
                    raise TypeError(f"Each schedule segment must be a dict, got {type(seg)}.")
                start = seg.get("start", 0)
                end_default = (self.n_cycles - 1) if self.n_cycles is not None else 10**9
                end = seg.get("end", end_default)
                value = seg["value"]
                if start < 0:
                    raise ValueError(f"segment start must be >= 0, got {start}.")
                if end < start:
                    raise ValueError(f"segment end must be >= start, got end={end}, start={start}.")
                if value <= 0:
                    raise ValueError(f"segment value must be > 0, got {value}.")
                segs.append(_StepSegment(start=start, end=end, value=value))
        self._segments = tuple(segs)

    def _value(self, cycle: int) -> float:
        for seg in self._segments:
            if seg.start <= cycle <= seg.end:
                return seg.value
        return self.default