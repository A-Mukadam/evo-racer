"""Track loading and spatial indexing.

Wraps simple_tracks.py. The track format is exactly what
simple_tracks.build_track / simple_tracks.make return:

    walls       : list[((x1, y1), (x2, y2))]   inner loop + outer loop segments, pixels
    checkpoints : list[((ix, iy), (ox, oy))]   ordered, inner point -> outer point, pixels
    centerline  : np.ndarray, shape (N, 2)     closed loop, pixels; spawn position + heading

Adds a uniform spatial grid over the wall segments so each raycast / collision
query only tests nearby segments instead of all of them.
"""

from dataclasses import dataclass

import numpy as np

import simple_tracks

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass
class Track:
    """A loaded track plus its spatial index.

    Attributes:
        name: key into simple_tracks.TRACKS (e.g. "oval").
        walls: list[Segment], pixels. Same order as simple_tracks output.
        checkpoints: list[Segment], pixels, ordered along the racing direction,
            each (inner point, outer point).
        centerline: np.ndarray (N, 2), pixels, closed loop.
        grid: SpatialGrid over `walls`.
    """

    name: str
    walls: list[Segment]
    checkpoints: list[Segment]
    centerline: np.ndarray
    grid: "SpatialGrid"


class SpatialGrid:
    """Uniform grid bucketing wall segments by the cells they overlap."""

    def __init__(self, walls: list[Segment], cell_size: float = 50.0):
        """Build the grid.

        Args:
            walls: list[Segment], pixels.
            cell_size: side length of one square cell, pixels.
                TODO: tune against MAX_RANGE in car.py (smaller cells = fewer
                candidates per cell, more cells traversed per ray).

        Returns:
            None. Stores a mapping (cell_x, cell_y) -> list of wall indices.
        """
        # TODO: compute bounding box, allocate cells, rasterize each segment
        # into every cell it touches (not just its endpoints' cells).
        raise NotImplementedError

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """Map a point to its grid cell.

        Args:
            x, y: point, pixels.

        Returns:
            (cell_x, cell_y) integer cell indices.
        """
        raise NotImplementedError

    def query_ray(self, origin: Point, angle: float, max_range: float) -> list[int]:
        """Candidate walls a ray could hit.

        Args:
            origin: ray start (x, y), pixels.
            angle: absolute ray direction, radians.
            max_range: ray length, pixels.

        Returns:
            Deduplicated list of indices into `walls` for every segment in any
            cell the ray passes through. Superset of true hits; caller does
            exact intersection.
        """
        # TODO: grid traversal (DDA / Amanatides-Woo) along the ray.
        raise NotImplementedError

    def query_radius(self, center: Point, radius: float) -> list[int]:
        """Candidate walls near a point (for car-wall collision).

        Args:
            center: (x, y), pixels.
            radius: search radius, pixels.

        Returns:
            Deduplicated list of indices into `walls` in cells overlapping the
            axis-aligned box of side 2*radius around `center`.
        """
        raise NotImplementedError


def load_track(name: str, half_width: float = 55.0, checkpoint_every: int = 8,
               cell_size: float = 50.0) -> Track:
    """Build a Track from a named simple_tracks generator.

    Args:
        name: key into simple_tracks.TRACKS.
        half_width: track half-width, pixels (passed to simple_tracks.make).
        checkpoint_every: one checkpoint per this many centerline points.
        cell_size: SpatialGrid cell size, pixels.

    Returns:
        Track with walls/checkpoints/centerline exactly as simple_tracks.make
        returns them, plus a SpatialGrid.
    """
    # TODO: call simple_tracks.make(name, half_width, checkpoint_every).
    # TODO: cache by (name, half_width, checkpoint_every) so worker processes
    # don't rebuild the grid every episode.
    raise NotImplementedError


def spawn_pose(track: Track, index: int = 0) -> tuple[float, float, float]:
    """Starting pose on the centerline.

    Args:
        track: loaded Track.
        index: centerline point index to spawn at.

    Returns:
        (x, y, heading): pixels, pixels, radians. Delegates to
        simple_tracks.spawn_pose.
    """
    raise NotImplementedError
