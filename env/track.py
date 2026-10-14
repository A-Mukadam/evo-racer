"""Track loading and spatial indexing.

Wraps simple_tracks.py. The track format is exactly what
simple_tracks.build_track / simple_tracks.make return:

    walls       : list[((x1, y1), (x2, y2))]   inner loop + outer loop segments, pixels
    checkpoints : list[((ix, iy), (ox, oy))]   ordered, inner point -> outer point, pixels
    centerline  : np.ndarray, shape (N, 2)     closed loop, pixels; spawn position + heading

Adds a uniform spatial grid over the wall segments so each raycast / collision
query only tests nearby segments instead of all of them.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import simple_tracks

Point = tuple[float, float]
Segment = tuple[Point, Point]

BOX = (250.0, 200.0, 750.0, 550.0)  # (x0, y0, x1, y1) of the "box" sanity track, pixels.
                                    # From the center: sides 175 px (< MAX_RANGE), ends 250 px.


@dataclass
class Track:
    """A loaded track plus its spatial index.

    Attributes:
        name: key into simple_tracks.TRACKS (e.g. "oval"), or "box".
        walls: list[Segment], pixels. Same order as simple_tracks output.
        checkpoints: list[Segment], pixels, ordered along the racing direction,
            each (inner point, outer point).
        centerline: np.ndarray (N, 2), pixels, closed loop.
        checkpoint_every: checkpoint k sits at centerline index k * checkpoint_every.
        grid: SpatialGrid over `walls`.
    """

    name: str
    walls: list[Segment]
    checkpoints: list[Segment]
    centerline: np.ndarray
    checkpoint_every: int
    grid: "SpatialGrid"


class SpatialGrid:
    """Uniform grid bucketing wall segments by the cells they overlap.

    Also keeps the walls as numpy arrays (`a`, `b`: segment start / end points,
    shape (W, 2)) so car.py can test many segments in one vectorized call.
    """

    def __init__(self, walls: list[Segment], cell_size: float = 50.0):
        """Build the grid.

        Args:
            walls: list[Segment], pixels.
            cell_size: side length of one square cell, pixels. 50 px with
                MAX_RANGE = 200 px means a ray crosses ~4-9 cells.

        Returns:
            None. Stores a mapping (cell_x, cell_y) -> list of wall indices.
        """
        self.cell_size = cell_size
        self.a = np.array([w[0] for w in walls], dtype=float).reshape(-1, 2)
        self.b = np.array([w[1] for w in walls], dtype=float).reshape(-1, 2)
        self.cells: dict[tuple[int, int], list[int]] = {}

        # Register each segment in every cell its bounding box overlaps, padded
        # by `pad` so a hit sitting exactly on a cell border is found from
        # either side. A superset of the cells it truly touches; that's fine,
        # callers do exact intersection.
        pad = 1e-3
        for i, ((x1, y1), (x2, y2)) in enumerate(walls):
            cx0, cy0 = self.cell_of(min(x1, x2) - pad, min(y1, y2) - pad)
            cx1, cy1 = self.cell_of(max(x1, x2) + pad, max(y1, y2) + pad)
            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    self.cells.setdefault((cx, cy), []).append(i)

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """Map a point to its grid cell.

        Args:
            x, y: point, pixels.

        Returns:
            (cell_x, cell_y) integer cell indices.
        """
        return math.floor(x / self.cell_size), math.floor(y / self.cell_size)

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
        # Amanatides-Woo grid traversal: step to whichever cell border
        # (vertical or horizontal) the ray reaches first.
        x, y = origin
        dx, dy = math.cos(angle), math.sin(angle)
        cs = self.cell_size
        cx, cy = self.cell_of(x, y)

        if dx > 0:
            step_x, t_next_x, t_delta_x = 1, ((cx + 1) * cs - x) / dx, cs / dx
        elif dx < 0:
            step_x, t_next_x, t_delta_x = -1, (cx * cs - x) / dx, -cs / dx
        else:
            step_x, t_next_x, t_delta_x = 0, math.inf, math.inf
        if dy > 0:
            step_y, t_next_y, t_delta_y = 1, ((cy + 1) * cs - y) / dy, cs / dy
        elif dy < 0:
            step_y, t_next_y, t_delta_y = -1, (cy * cs - y) / dy, -cs / dy
        else:
            step_y, t_next_y, t_delta_y = 0, math.inf, math.inf

        found: set[int] = set()
        t = 0.0
        while t <= max_range:
            found.update(self.cells.get((cx, cy), ()))
            if t_next_x < t_next_y:
                t, cx = t_next_x, cx + step_x
                t_next_x += t_delta_x
            else:
                t, cy = t_next_y, cy + step_y
                t_next_y += t_delta_y
        return list(found)

    def query_radius(self, center: Point, radius: float) -> list[int]:
        """Candidate walls near a point (for car-wall collision).

        Args:
            center: (x, y), pixels.
            radius: search radius, pixels.

        Returns:
            Deduplicated list of indices into `walls` in cells overlapping the
            axis-aligned box of side 2*radius around `center`.
        """
        x, y = center
        cx0, cy0 = self.cell_of(x - radius, y - radius)
        cx1, cy1 = self.cell_of(x + radius, y + radius)
        found: set[int] = set()
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                found.update(self.cells.get((cx, cy), ()))
        return list(found)


def box_walls(x0: float, y0: float, x1: float, y1: float) -> list[Segment]:
    """Four walls of an axis-aligned rectangle, pixels, same format as simple_tracks walls."""
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


@lru_cache(maxsize=None)
def load_track(name: str, half_width: float = 55.0, checkpoint_every: int = 8,
               cell_size: float = 50.0) -> Track:
    """Build a Track from a named simple_tracks generator.

    Cached, so worker processes build each grid once, not once per episode.
    Callers must not mutate the returned Track.

    Args:
        name: key into simple_tracks.TRACKS, or "box" for the 4-wall sanity
            box (no checkpoints; spawns at the center facing +x).
        half_width: track half-width, pixels (passed to simple_tracks.make).
        checkpoint_every: one checkpoint per this many centerline points.
        cell_size: SpatialGrid cell size, pixels.

    Returns:
        Track with walls/checkpoints/centerline exactly as simple_tracks.make
        returns them, plus a SpatialGrid.
    """
    if name == "box":
        x0, y0, x1, y1 = BOX
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        walls = box_walls(x0, y0, x1, y1)
        # Two-point "centerline": spawn_pose(index=0) -> center, facing +x.
        centerline = np.array([[cx, cy], [cx + 1.0, cy]])
        return Track(name, walls, [], centerline, checkpoint_every,
                     SpatialGrid(walls, cell_size))

    walls, checkpoints, centerline = simple_tracks.make(name, half_width, checkpoint_every)
    return Track(name, walls, checkpoints, centerline, checkpoint_every,
                 SpatialGrid(walls, cell_size))


def spawn_pose(track: Track, index: int = 0) -> tuple[float, float, float]:
    """Starting pose on the centerline.

    Args:
        track: loaded Track.
        index: centerline point index to spawn at.

    Returns:
        (x, y, heading): pixels, pixels, radians. Delegates to
        simple_tracks.spawn_pose.
    """
    return simple_tracks.spawn_pose(track.centerline, index)
