"""Physics and episode-loop tests for env.car / env.game."""

import math

import numpy as np

import simple_tracks
from env.car import CAR_RADIUS, MAX_SPEED, MAX_TURN_RATE, CarState, step_physics
from env.game import MAX_FRAMES, STALL_FRAMES, RaceEnv, compute_fitness, run_episode
from env.track import BOX, load_track


def follow_gaps(obs):
    """Steer toward the side with more free space; brake when the front ray is short."""
    s = obs[:7]
    steer = 3.0 * ((s[4] + s[5] + s[6]) - (s[0] + s[1] + s[2]))
    return float(np.clip(steer, -1, 1)), (1.0 if s[3] > 0.45 else -0.3)


def test_fitness_formula():
    assert compute_fitness(10, 200) == 10 * 100 + (10 / 200) * 500
    assert compute_fitness(0, 0) == 0


def test_speed_clamped_and_never_negative():
    s = CarState(0, 0, 0)
    for _ in range(200):
        step_physics(s, 0.0, 1.0)
    assert s.speed == MAX_SPEED
    for _ in range(200):
        step_physics(s, 0.0, -1.0)
    assert s.speed == 0.0


def test_steer_sign_and_speed_scaling():
    """Negative steer decreases heading (left); a stopped car can't turn."""
    s = CarState(0, 0, 0, speed=MAX_SPEED)
    step_physics(s, -1.0, 1.0)
    assert s.heading == -MAX_TURN_RATE
    s = CarState(0, 0, 0, speed=MAX_SPEED)
    step_physics(s, 1.0, 1.0)
    assert s.heading == MAX_TURN_RATE
    s = CarState(0, 0, 0)
    step_physics(s, 1.0, 0.0)
    assert s.heading == 0.0


def test_wall_ends_episode():
    """Driving straight in the box hits the right wall."""
    track = load_track("box")
    env = RaceEnv(stall_frames=None)
    res = run_episode(lambda obs: (0.0, 1.0), track, None, env)
    assert res["reason"] == "wall"
    assert BOX[2] - env.state.x <= CAR_RADIUS


def test_stall_ends_episode():
    res = run_episode(lambda obs: (0.0, 0.0), load_track("oval"), None)
    assert res["reason"] == "stall"
    assert res["frames_alive"] == STALL_FRAMES
    assert res["fitness"] == 0


def test_hand_controller_laps_every_track():
    """Physics constants leave every track drivable: a simple reactive
    controller survives to the frame cap and completes more than one lap."""
    for name in simple_tracks.TRACKS:
        track = load_track(name)
        res = run_episode(follow_gaps, track, 0)
        assert res["reason"] == "max_frames", name
        assert res["frames_alive"] == MAX_FRAMES
        assert res["checkpoints_passed"] > len(track.checkpoints), name


def test_lap_times():
    """One lap per full set of checkpoints; lap times add up to at most the episode."""
    env = RaceEnv()
    for name, seed in [("oval", None), ("wavy", 4)]:
        track = load_track(name)
        res = run_episode(follow_gaps, track, seed, env)
        n_laps = res["checkpoints_passed"] // len(track.checkpoints)
        assert n_laps >= 1, name
        assert len(env.lap_frames) == n_laps
        assert all(f > 100 for f in env.lap_frames)
        assert sum(env.lap_frames) + env.current_lap_frames == res["frames_alive"]
    env.reset(load_track("oval"), None)
    assert env.lap_frames == [] and env.current_lap_frames == 0


def test_episode_deterministic_given_seed():
    track = load_track("wavy")
    assert run_episode(follow_gaps, track, 3) == run_episode(follow_gaps, track, 3)


def test_seed_varies_spawn():
    track = load_track("oval")
    env = RaceEnv()
    spawns = set()
    for seed in range(20):
        env.reset(track, seed)
        spawns.add((round(env.state.x, 6), round(env.state.y, 6)))
    assert len(spawns) > 5
    env.reset(track, None)
    x, y, heading = simple_tracks.spawn_pose(track.centerline, 0)
    assert (env.state.x, env.state.y, env.state.heading) == (x, y, heading)
    assert env.next_checkpoint == 1
