"""Geometry tests. Must pass before any EA code is trusted.

Wrong raycast distances look like GA failure; these catch them first.
"""

import math

import numpy as np
import pytest

import simple_tracks
from env.car import (CAR_RADIUS, MAX_RANGE, N_SENSORS, SENSOR_ANGLES, CarState,
                     cast_rays, cast_rays_raw, collides, ray_segment_distance)
from env.game import RaceEnv, crossed
from env.track import SpatialGrid, load_track

ALL_TRACKS = ["box"] + list(simple_tracks.TRACKS)


def box_walls(size: float = 400.0):
    """Four-wall square box for sanity tests.

    Args:
        size: side length, pixels. Box spans (0, 0) to (size, size).

    Returns:
        list of 4 wall segments ((x1, y1), (x2, y2)), pixels, same format as
        simple_tracks walls.
    """
    return [((0.0, 0.0), (size, 0.0)), ((size, 0.0), (size, size)),
            ((size, size), (0.0, size)), ((0.0, size), (0.0, 0.0))]


def box_distance(x: float, y: float, angle: float, size: float) -> float:
    """Analytic distance from (x, y) inside the box to its wall along `angle`, capped at MAX_RANGE."""
    dx, dy = math.cos(angle), math.sin(angle)
    ts = []
    if abs(dx) > 1e-12:
        ts.append((size - x) / dx if dx > 0 else -x / dx)
    if abs(dy) > 1e-12:
        ts.append((size - y) / dy if dy > 0 else -y / dy)
    return min(min(ts), MAX_RANGE)


def test_ray_hits_box_walls_at_known_distance():
    """Car at box center, heading 0: each ray's distance matches the
    analytic distance to the box wall (within 1e-6 pixels, or MAX_RANGE if farther).
    Also checks off-center, rotated poses."""
    size = 300.0
    walls = box_walls(size)
    grid = SpatialGrid(walls)
    for x, y, heading in [(150, 150, 0.0), (60, 220, 0.3), (250, 40, -2.0), (100, 100, math.pi)]:
        state = CarState(x, y, heading)
        for rel, (dist, hit) in zip(SENSOR_ANGLES, cast_rays_raw(state, walls, grid)):
            expected = box_distance(x, y, heading + rel, size)
            assert dist == pytest.approx(expected, abs=1e-6)
            if hit is not None:   # hit point lies on the box boundary
                assert min(abs(hit[0]), abs(hit[0] - size),
                           abs(hit[1]), abs(hit[1] - size)) == pytest.approx(0, abs=1e-6)


def test_ray_misses_parallel_segment():
    """A ray parallel to a segment returns None."""
    assert ray_segment_distance((0, 0), 0.0, ((10, 5), (20, 5)), 100) is None
    assert ray_segment_distance((0, 0), 0.0, ((10, 0), (20, 0)), 100) is None  # collinear


def test_ray_ignores_hits_behind_origin():
    """A segment behind the ray origin is not hit."""
    seg = ((-10, -5), (-10, 5))
    assert ray_segment_distance((0, 0), 0.0, seg, 100) is None
    assert ray_segment_distance((0, 0), math.pi, seg, 100) == pytest.approx(10)
    assert ray_segment_distance((0, 0), math.pi, seg, 5) is None   # beyond max_range


def test_sensor_angles_relative_to_heading():
    """Rotating the car rotates the rays: readings in a box are permuted as expected."""
    size = 300.0
    walls = box_walls(size)
    grid = SpatialGrid(walls)
    x, y = 100.0, 170.0   # off-center so every direction reads differently

    def read(heading):
        return [d for d, _ in cast_rays_raw(CarState(x, y, heading), walls, grid)]

    r0, r90, r180 = read(0.0), read(math.pi / 2), read(math.pi)
    # SENSOR_ANGLES_DEG = (-90, -45, -20, 0, 20, 45, 90) -> indices 0..6
    assert r0[0] == pytest.approx(y)          # -90 = screen up = car's left
    assert r0[6] == pytest.approx(size - y)   # +90 = screen down = car's right
    assert r90[0] == pytest.approx(r0[3])     # heading 90: -90 points where 0 did
    assert r90[1] == pytest.approx(r0[5])     #             -45 points where +45 did
    assert r90[3] == pytest.approx(r0[6])     #               0 points where +90 did
    assert r180[0] == pytest.approx(r0[6])    # heading 180 swaps left and right
    assert r180[6] == pytest.approx(r0[0])


def test_sensor_readings_normalized():
    """cast_rays returns shape (7,) with all values in [0, 1]."""
    env = RaceEnv()
    for name in ALL_TRACKS:
        track = load_track(name)
        for seed in [None, 0, 1, 2]:
            env.reset(track, seed)
            r = cast_rays(env.state, track.walls, track.grid)
            assert r.shape == (N_SENSORS,)
            assert np.all((r >= 0.0) & (r <= 1.0))


def test_spatial_grid_matches_brute_force():
    """On every simple_tracks track, grid-accelerated cast_rays equals
    brute-force over all walls for many random poses."""
    rng = np.random.default_rng(0)
    for name in ALL_TRACKS:
        track = load_track(name)
        poses = [(x, y, h) for x, y, h in zip(rng.uniform(0, 1000, 300),
                                              rng.uniform(0, 700, 300),
                                              rng.uniform(-math.pi, math.pi, 300))]
        # Poses within a few pixels of a wall stress collision and short rays.
        for i in rng.integers(len(track.walls), size=200):
            (x1, y1), (x2, y2) = track.walls[i]
            u = rng.uniform()
            poses.append((x1 + u * (x2 - x1) + rng.normal(0, 6),
                          y1 + u * (y2 - y1) + rng.normal(0, 6),
                          rng.uniform(-math.pi, math.pi)))
        # Rays running exactly along grid-cell borders and through cell corners.
        for x, y in [(150.0, 150.0), (200.0, 250.0), (500.0, 350.0)]:
            for h in np.arange(8) * math.pi / 4:
                poses.append((x, y, h))

        for x, y, h in poses:
            s = CarState(x, y, h)
            grid_d = [d for d, _ in cast_rays_raw(s, track.walls, track.grid)]
            brute_d = [d for d, _ in cast_rays_raw(s, track.walls, None)]
            assert grid_d == pytest.approx(brute_d, abs=1e-9), (name, x, y, h)
            assert collides(s, track.walls, track.grid) == collides(s, track.walls, None)


def test_collision_at_wall():
    """collides() is True within CAR_RADIUS of a wall and False at box center."""
    walls = box_walls(400.0)
    grid = SpatialGrid(walls)
    assert not collides(CarState(200, 200, 0), walls, grid)
    assert collides(CarState(CAR_RADIUS * 0.5, 200, 0), walls, grid)
    assert collides(CarState(CAR_RADIUS, 200, 0), walls, grid)
    assert not collides(CarState(CAR_RADIUS * 1.5, 200, 0), walls, grid)
    assert collides(CarState(400 - CAR_RADIUS * 0.5, 400 - CAR_RADIUS * 0.5, 0), walls, grid)


def test_checkpoint_crossing():
    """crossed() detects motion across a checkpoint and ignores motion that stops short."""
    seg = ((10.0, 20.0), (10.0, 0.0))   # (inner, outer); racing direction is +x
    assert crossed((0, 10), (15, 10), seg)
    assert crossed((5, 10), (10, 10), seg)        # ending exactly on the line counts
    assert not crossed((0, 10), (9, 10), seg)     # stops short
    assert not crossed((15, 10), (0, 10), seg)    # backwards
    assert not crossed((0, 30), (15, 30), seg)    # passes beyond the segment's end
    assert not crossed((0, 10), (0, 10), seg)     # no motion


def test_track_matches_simple_tracks_interface():
    """load_track output walls/checkpoints/centerline equal simple_tracks.make output
    exactly (types, lengths, values)."""
    for name in simple_tracks.TRACKS:
        track = load_track(name)
        walls, checkpoints, centerline = simple_tracks.make(name)
        assert isinstance(track.walls, list) and isinstance(track.checkpoints, list)
        assert track.walls == walls
        assert track.checkpoints == checkpoints
        assert isinstance(track.centerline, np.ndarray)
        assert np.array_equal(track.centerline, centerline)


def test_spawn_is_inside_track():
    """spawn_pose on every track is not colliding and faces the next checkpoint,
    for the canonical start and for seeded spawns."""
    env = RaceEnv()
    for name in simple_tracks.TRACKS:
        track = load_track(name)
        for seed in [None] + list(range(10)):
            env.reset(track, seed)
            s = env.state
            assert not collides(s, track.walls, track.grid), (name, seed)
            (ix, iy), (ox, oy) = track.checkpoints[env.next_checkpoint]
            to_next = ((ix + ox) / 2 - s.x, (iy + oy) / 2 - s.y)
            assert to_next[0] * math.cos(s.heading) + to_next[1] * math.sin(s.heading) > 0
