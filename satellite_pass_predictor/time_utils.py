"""Time-grid helper shared by propagation and visibility."""

import numpy as np
from skyfield.timelib import Time, Timescale


def build_time_grid(
    ts: Timescale,
    start_time: Time | None = None,
    duration_hours: float = 24,
    step_minutes: float = 1,
) -> Time:
    """Return one vectorized Time sampled every `step_minutes` from `start_time` (default: now).

    The first sample is `start_time`; the last is at or before `start_time + duration_hours`.
    """
    if start_time is None:
        start_time = ts.now()

    n_steps = int(duration_hours * 60 / step_minutes) + 1
    minutes = np.arange(n_steps) * step_minutes
    return start_time + minutes / 1440.0  # Skyfield Time + fractional days
