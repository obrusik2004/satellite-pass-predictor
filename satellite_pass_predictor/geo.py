"""Antimeridian handling shared by the flat ground-track plot and the 3D globe."""

import numpy as np
from numpy.typing import NDArray


def find_antimeridian_crossings(longitudes: NDArray[np.float64]) -> NDArray[np.intp]:
    """Return indices i where longitudes[i] -> longitudes[i+1] jumps across +-180 degrees.

    A jump of more than 180 degrees between samples counts as a crossing (exactly 180 does not).
    """
    longitudes = np.asarray(longitudes, dtype=float)
    return np.where(np.abs(np.diff(longitudes)) > 180.0)[0]
