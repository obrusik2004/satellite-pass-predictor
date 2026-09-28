"""
Shared geographic helper: antimeridian (+-180 degree longitude) crossing
detection. Used by both visualization.py's flat matplotlib plot (which
breaks the line with an inserted NaN -- a 2D lat/lon rectangle draws +180
and -180 as two different edges, so a crossing needs an explicit break)
and globe.py's 3D globe (which needs actual separate sub-paths, each
ending/starting exactly on the meridian, since pydeck's PathLayer has no
NaN-break equivalent -- see globe.py's own splitting function for that).
The underlying question -- "where does this track cross +-180 degrees"
-- is identical either way, so it's answered once, here.
"""

import numpy as np
from numpy.typing import NDArray


def find_antimeridian_crossings(longitudes: NDArray[np.float64]) -> NDArray[np.intp]:
    """
    Indices i such that consecutive samples longitudes[i] -> longitudes[i+1]
    jump across the +-180 degree antimeridian (a real crossing, not an
    ~360 degree move that happens to coincide with one). `> 180.0`, not
    `>= 180.0`: an exact 180-degree jump is not treated as a crossing --
    not a physically expected case for real subpoint longitudes, but a
    deliberate, pinned-down boundary rather than an accident of the
    comparison operator.
    """
    longitudes = np.asarray(longitudes, dtype=float)
    return np.where(np.abs(np.diff(longitudes)) > 180.0)[0]
