"""Geometry tests. Must pass before any EA code is trusted.

Wrong raycast distances look like GA failure; these catch them first.
"""


def box_walls(size: float = 400.0):
    """Four-wall square box for sanity tests.

    Args:
        size: side length, pixels. Box spans (0, 0) to (size, size).

    Returns:
        list of 4 wall segments ((x1, y1), (x2, y2)), pixels, same format as
        simple_tracks walls.
    """
    raise NotImplementedError


def test_ray_hits_box_walls_at_known_distance():
    """Car at box center, heading 0: each ray's distance matches the
    analytic distance to the box wall (within 1e-6 pixels, or MAX_RANGE if farther)."""
    raise NotImplementedError


def test_ray_misses_parallel_segment():
    """A ray parallel to a segment returns None."""
    raise NotImplementedError


def test_ray_ignores_hits_behind_origin():
    """A segment behind the ray origin is not hit."""
    raise NotImplementedError


def test_sensor_angles_relative_to_heading():
    """Rotating the car rotates the rays: readings in a box are permuted as expected."""
    raise NotImplementedError


def test_sensor_readings_normalized():
    """cast_rays returns shape (7,) with all values in [0, 1]."""
    raise NotImplementedError


def test_spatial_grid_matches_brute_force():
    """On every simple_tracks track, grid-accelerated cast_rays equals
    brute-force over all walls for many random poses."""
    raise NotImplementedError


def test_collision_at_wall():
    """collides() is True within CAR_RADIUS of a wall and False at box center."""
    raise NotImplementedError


def test_checkpoint_crossing():
    """crossed() detects motion across a checkpoint and ignores motion that stops short."""
    raise NotImplementedError


def test_track_matches_simple_tracks_interface():
    """load_track output walls/checkpoints/centerline equal simple_tracks.make output
    exactly (types, lengths, values)."""
    raise NotImplementedError


def test_spawn_is_inside_track():
    """spawn_pose on every track is not colliding and faces checkpoint 0 or 1."""
    raise NotImplementedError
