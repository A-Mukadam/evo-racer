"""Arcade car physics and raycast distance sensors.

Physics is arcade, not simulation: speed, friction, and a turn rate scaled by
speed. No drift, tire slip, or gears. Fixed timestep of 1 frame.
"""

from dataclasses import dataclass

import numpy as np

from env.track import Point, Segment, SpatialGrid

# Sensor layout: fixed by design, see CLAUDE.md.
SENSOR_ANGLES_DEG = (-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0)  # relative to heading
N_SENSORS = len(SENSOR_ANGLES_DEG)

# TODO: tune these in the W1 manual-drive loop.
MAX_RANGE = None      # ray length, pixels; sensor readings are divided by this
MAX_SPEED = None      # pixels / frame; speed input is divided by this
ACCEL = None          # pixels / frame^2 at throttle = +1
BRAKE = None          # pixels / frame^2 at throttle = -1
FRICTION = None       # fractional speed loss per frame, unitless in [0, 1)
MAX_TURN_RATE = None  # radians / frame at full steer and MAX_SPEED
CAR_RADIUS = None     # collision radius, pixels


@dataclass
class CarState:
    """Kinematic state of the car.

    Attributes:
        x, y: position, pixels.
        heading: radians.
        speed: pixels / frame, >= 0.
    """

    x: float
    y: float
    heading: float
    speed: float = 0.0


def step_physics(state: CarState, steer: float, throttle: float) -> CarState:
    """Advance the car one frame.

    Args:
        state: current CarState.
        steer: in [-1, 1]; negative = turn left (toward decreasing heading).
        throttle: in [-1, 1]; positive accelerates, negative brakes.

    Returns:
        New CarState after one frame. Turn rate scales with speed / MAX_SPEED.
        Speed clamped to [0, MAX_SPEED].
    """
    # TODO: decide whether to mutate in place or return a new state
    # (in place is faster in the eval loop).
    raise NotImplementedError


def ray_segment_distance(origin: Point, angle: float, seg: Segment,
                         max_range: float) -> float | None:
    """Exact distance from a ray origin to a wall segment.

    Args:
        origin: (x, y), pixels.
        angle: absolute ray direction, radians.
        seg: ((x1, y1), (x2, y2)), pixels.
        max_range: ray length, pixels.

    Returns:
        Distance in pixels along the ray to the intersection, or None if the
        ray misses the segment or the hit is beyond max_range.
    """
    raise NotImplementedError


def cast_rays(state: CarState, walls: list[Segment], grid: SpatialGrid) -> np.ndarray:
    """Read all 7 distance sensors.

    Args:
        state: current CarState.
        walls: list[Segment], pixels (the grid stores indices into this).
        grid: SpatialGrid over `walls`.

    Returns:
        np.ndarray shape (N_SENSORS,), float, each in [0, 1]:
        nearest hit distance / MAX_RANGE, or 1.0 if nothing within MAX_RANGE.
        Order matches SENSOR_ANGLES_DEG.
    """
    raise NotImplementedError


def cast_rays_raw(state: CarState, walls: list[Segment],
                  grid: SpatialGrid) -> list[tuple[float, Point | None]]:
    """Unnormalized sensor readings for the debug overlay and tests.

    Args:
        state: current CarState.
        walls: list[Segment], pixels.
        grid: SpatialGrid over `walls`.

    Returns:
        One (distance_pixels, hit_point_or_None) per sensor, in
        SENSOR_ANGLES_DEG order. distance = MAX_RANGE when no hit.
    """
    raise NotImplementedError


def collides(state: CarState, walls: list[Segment], grid: SpatialGrid) -> bool:
    """Whether the car touches any wall.

    Args:
        state: current CarState.
        walls: list[Segment], pixels.
        grid: SpatialGrid over `walls`.

    Returns:
        True if any wall segment is within CAR_RADIUS pixels of (x, y).
    """
    # TODO: point-vs-circle is enough for W1; also consider testing the
    # swept segment from the previous position so fast cars can't tunnel.
    raise NotImplementedError


def observation(state: CarState, walls: list[Segment], grid: SpatialGrid) -> np.ndarray:
    """Network input vector.

    Args:
        state: current CarState.
        walls: list[Segment], pixels.
        grid: SpatialGrid over `walls`.

    Returns:
        np.ndarray shape (8,), float in [0, 1]:
        7 normalized sensor readings followed by speed / MAX_SPEED.
    """
    raise NotImplementedError
