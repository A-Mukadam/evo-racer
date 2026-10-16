"""Arcade car physics and raycast distance sensors.

Physics is arcade, not simulation: speed, friction, and a turn rate scaled by
speed. No drift, tire slip, or gears. Fixed timestep of 1 frame.
"""

import math
from dataclasses import dataclass

import numpy as np

from env.track import Point, Segment, SpatialGrid

# Sensor layout: fixed by design, see CLAUDE.md.
SENSOR_ANGLES_DEG = (-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0)  # relative to heading
N_SENSORS = len(SENSOR_ANGLES_DEG)
SENSOR_ANGLES = tuple(math.radians(a) for a in SENSOR_ANGLES_DEG)

# First-pass values; tune in the W1 manual-drive loop.
MAX_RANGE = 200.0     # ray length, pixels; sensor readings are divided by this
MAX_SPEED = 10.0      # pixels / frame; speed input is divided by this. Raised from 6 in V2:
                      # at 6, full throttle took every corner, so random drivers lapped.
ACCEL = 0.3           # pixels / frame^2 at throttle = +1 (friction alone caps speed at
                      # ACCEL * (1 - FRICTION) / FRICTION = 14.7, above MAX_SPEED)
BRAKE = 0.5           # pixels / frame^2 at throttle = -1
FRICTION = 0.02       # fractional speed loss per frame, unitless in [0, 1)
MAX_TURN_RATE = 0.08  # radians / frame at full steer and MAX_SPEED
TURN_SPEED = 2.0      # pixels / frame; turn rate ramps up linearly to MAX_TURN_RATE
                      # at this speed, then stays flat. So turning radius at full
                      # steer = speed / MAX_TURN_RATE (125 px at MAX_SPEED): faster
                      # cars turn wider and tight corners need braking.
                      # Set TURN_SPEED = MAX_SPEED for a purely linear ramp.
CAR_RADIUS = 8.0      # collision radius, pixels

# No tunneling: a car moving < 2 * CAR_RADIUS per frame can't jump over a
# zero-thickness wall without some frame ending within CAR_RADIUS of it.
assert MAX_SPEED < 2 * CAR_RADIUS


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
    """Advance the car one frame, mutating `state` in place.

    Args:
        state: current CarState.
        steer: in [-1, 1]; negative = turn left (toward decreasing heading).
        throttle: in [-1, 1]; positive accelerates, negative brakes.

    Returns:
        The same CarState object, after one frame. Turn rate scales with
        min(speed / TURN_SPEED, 1). Speed clamped to [0, MAX_SPEED].
    """
    steer = min(max(steer, -1.0), 1.0)
    throttle = min(max(throttle, -1.0), 1.0)

    speed = state.speed + throttle * (ACCEL if throttle >= 0 else BRAKE)
    speed *= 1.0 - FRICTION
    state.speed = min(max(speed, 0.0), MAX_SPEED)

    state.heading += steer * MAX_TURN_RATE * min(state.speed / TURN_SPEED, 1.0)
    state.x += state.speed * math.cos(state.heading)
    state.y += state.speed * math.sin(state.heading)
    return state


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
        ray misses the segment, is parallel to it, or the hit is beyond max_range.
    """
    # Solve origin + t*d = a + u*e for t (distance along ray) and u (0..1 along segment).
    (ox, oy), ((ax, ay), (bx, by)) = origin, seg
    dx, dy = math.cos(angle), math.sin(angle)
    ex, ey = bx - ax, by - ay
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return None
    aox, aoy = ax - ox, ay - oy
    t = (aox * ey - aoy * ex) / denom
    u = (aox * dy - aoy * dx) / denom
    if 0.0 <= t <= max_range and 0.0 <= u <= 1.0:
        return t
    return None


def _nearest_hits(state: CarState, walls: list[Segment],
                  grid: SpatialGrid | None) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ray_segment_distance for all sensors against candidate walls.

    Returns:
        (dist, hit): dist shape (N_SENSORS,) pixels, MAX_RANGE where no hit;
        hit shape (N_SENSORS,) bool.
    """
    angles = state.heading + np.array(SENSOR_ANGLES)
    if grid is None:   # brute force over every wall (reference for tests)
        a = np.array([w[0] for w in walls], dtype=float)
        b = np.array([w[1] for w in walls], dtype=float)
    else:
        idx: set[int] = set()
        for ang in angles:
            idx.update(grid.query_ray((state.x, state.y), ang, MAX_RANGE))
        if not idx:
            return np.full(N_SENSORS, MAX_RANGE), np.zeros(N_SENSORS, dtype=bool)
        idx_arr = np.fromiter(idx, dtype=int, count=len(idx))
        a, b = grid.a[idx_arr], grid.b[idx_arr]

    dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]   # (R, 1)
    ex, ey = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]                # (K,)
    aox, aoy = a[:, 0] - state.x, a[:, 1] - state.y               # (K,)
    denom = dx * ey - dy * ex                                     # (R, K)
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)        # parallel -> no hit
    t = (aox * ey - aoy * ex) / denom
    u = (aox * dy - aoy * dx) / denom
    valid = (t >= 0.0) & (t <= MAX_RANGE) & (u >= 0.0) & (u <= 1.0)
    dist = np.where(valid, t, np.inf).min(axis=1)
    hit = np.isfinite(dist)
    return np.where(hit, dist, MAX_RANGE), hit


def cast_rays(state: CarState, walls: list[Segment], grid: SpatialGrid | None) -> np.ndarray:
    """Read all 7 distance sensors.

    Args:
        state: current CarState.
        walls: list[Segment], pixels (the grid stores indices into this).
        grid: SpatialGrid over `walls`, or None to brute-force every wall.

    Returns:
        np.ndarray shape (N_SENSORS,), float, each in [0, 1]:
        nearest hit distance / MAX_RANGE, or 1.0 if nothing within MAX_RANGE.
        Order matches SENSOR_ANGLES_DEG.
    """
    dist, _ = _nearest_hits(state, walls, grid)
    return dist / MAX_RANGE


def cast_rays_raw(state: CarState, walls: list[Segment],
                  grid: SpatialGrid | None) -> list[tuple[float, Point | None]]:
    """Unnormalized sensor readings for the debug overlay and tests.

    Args:
        state: current CarState.
        walls: list[Segment], pixels.
        grid: SpatialGrid over `walls`, or None to brute-force every wall.

    Returns:
        One (distance_pixels, hit_point_or_None) per sensor, in
        SENSOR_ANGLES_DEG order. distance = MAX_RANGE when no hit.
    """
    dist, hit = _nearest_hits(state, walls, grid)
    out = []
    for rel, d, h in zip(SENSOR_ANGLES, dist, hit):
        ang = state.heading + rel
        point = (state.x + d * math.cos(ang), state.y + d * math.sin(ang)) if h else None
        out.append((float(d), point))
    return out


def collides(state: CarState, walls: list[Segment], grid: SpatialGrid | None) -> bool:
    """Whether the car touches any wall.

    Args:
        state: current CarState.
        walls: list[Segment], pixels.
        grid: SpatialGrid over `walls`, or None to brute-force every wall.

    Returns:
        True if any wall segment is within CAR_RADIUS pixels of (x, y).
        Tunneling is ruled out by MAX_SPEED < 2 * CAR_RADIUS (asserted above).
    """
    if grid is None:
        a = np.array([w[0] for w in walls], dtype=float)
        b = np.array([w[1] for w in walls], dtype=float)
    else:
        idx = grid.query_radius((state.x, state.y), CAR_RADIUS)
        if not idx:
            return False
        a, b = grid.a[idx], grid.b[idx]

    # Distance from the car center to the closest point on each segment.
    e = b - a
    p = np.array([state.x, state.y])
    u = np.clip(np.einsum("ij,ij->i", p - a, e) / np.einsum("ij,ij->i", e, e), 0.0, 1.0)
    closest = a + u[:, None] * e
    d2 = np.einsum("ij,ij->i", p - closest, p - closest)
    return bool((d2 <= CAR_RADIUS * CAR_RADIUS).any())


def observation(state: CarState, walls: list[Segment], grid: SpatialGrid | None) -> np.ndarray:
    """Network input vector.

    Args:
        state: current CarState.
        walls: list[Segment], pixels.
        grid: SpatialGrid over `walls`.

    Returns:
        np.ndarray shape (8,), float in [0, 1]:
        7 normalized sensor readings followed by speed / MAX_SPEED.
    """
    obs = np.empty(N_SENSORS + 1)
    obs[:N_SENSORS] = cast_rays(state, walls, grid)
    obs[N_SENSORS] = state.speed / MAX_SPEED
    return obs
