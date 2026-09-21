"""
Generic time-grid utilities shared across domain modules. Nothing here is
specific to orbit propagation or ground-station visibility -- both just
happen to need "many evenly-spaced instants over the next N hours".
"""

import numpy as np
from skyfield.timelib import Time, Timescale


def build_time_grid(
    ts: Timescale,
    start_time: Time | None = None,
    duration_hours: float = 24,
    step_minutes: float = 1,
) -> Time:
    """
    Build a single vectorized Skyfield time spanning `duration_hours`
    starting at `start_time` (default: now), sampled every `step_minutes`.

    Shared by the ground-track and pass-prediction steps: both need "many
    evenly-spaced instants over the next N hours" and both hand the
    result straight to a function built to vectorize over a time array
    (get_subpoint(), compute_altaz()) rather than looping per-sample in
    Python.
    """
    if start_time is None:
        start_time = ts.now()

    n_steps = int(duration_hours * 60 / step_minutes) + 1
    minutes = np.arange(n_steps) * step_minutes
    return start_time + minutes / 1440.0  # Skyfield Time + fractional days
