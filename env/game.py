"""RaceEnv: headless episode loop, checkpoint tracking, fitness.

Fitness = checkpoints_passed * 100 + (checkpoints_passed / frames_alive) * 500
Episode ends on wall contact, on no new checkpoint for STALL_FRAMES frames, or
at MAX_FRAMES. Raw distance traveled is never rewarded.
"""

import numpy as np

from env.car import CarState
from env.track import Segment, Track

STALL_FRAMES = 150   # frames without a new checkpoint before termination
MAX_FRAMES = None    # hard episode cap, frames. TODO: set from lap time on the longest track.


def compute_fitness(checkpoints_passed: int, frames_alive: int) -> float:
    """Episode fitness.

    Args:
        checkpoints_passed: count of checkpoints crossed in order (can exceed
            len(checkpoints) if the car laps).
        frames_alive: frames survived, >= 1.

    Returns:
        checkpoints_passed * 100 + (checkpoints_passed / frames_alive) * 500.
    """
    raise NotImplementedError


def crossed(p0: tuple[float, float], p1: tuple[float, float], seg: Segment) -> bool:
    """Whether the car's motion this frame crossed a checkpoint line.

    Args:
        p0: car position at the start of the frame, pixels.
        p1: car position at the end of the frame, pixels.
        seg: checkpoint ((ix, iy), (ox, oy)), pixels.

    Returns:
        True if segment p0->p1 intersects `seg`.
    """
    # TODO: decide whether crossing direction matters (prevents back-and-forth exploit).
    raise NotImplementedError


class RaceEnv:
    """Headless racing episode. No pygame dependency."""

    def __init__(self, max_frames: int | None = MAX_FRAMES, stall_frames: int = STALL_FRAMES):
        """Configure termination limits.

        Args:
            max_frames: hard episode cap, frames.
            stall_frames: frames allowed without a new checkpoint.

        Returns:
            None. Call reset() before step().
        """
        raise NotImplementedError

    def reset(self, track: Track, seed: int) -> np.ndarray:
        """Start a new episode.

        Args:
            track: loaded Track.
            seed: episode seed. All genomes in a generation must receive the
                same seed so they face identical conditions.

        Returns:
            Initial observation, np.ndarray shape (8,), float in [0, 1]
            (see car.observation).
        """
        # TODO: decide what the seed controls (spawn index? small heading
        # jitter?). Must be deterministic given (track, seed).
        raise NotImplementedError

    def step(self, steer: float, throttle: float) -> tuple[np.ndarray, bool, dict]:
        """Advance one frame.

        Args:
            steer: in [-1, 1].
            throttle: in [-1, 1].

        Returns:
            (obs, done, info):
                obs: np.ndarray shape (8,), float in [0, 1].
                done: True if the episode terminated this frame.
                info: {"checkpoints_passed": int, "frames_alive": int,
                       "reason": str | None  # "wall" | "stall" | "max_frames"}
        """
        raise NotImplementedError

    @property
    def state(self) -> CarState:
        """Current CarState (pixels, radians). For rendering only."""
        raise NotImplementedError

    @property
    def next_checkpoint(self) -> int:
        """Index into track.checkpoints of the next checkpoint to cross."""
        raise NotImplementedError

    def fitness(self) -> float:
        """Fitness of the episode so far.

        Returns:
            compute_fitness(checkpoints_passed, frames_alive).
        """
        raise NotImplementedError


def run_episode(policy, track: Track, seed: int, env: RaceEnv | None = None) -> dict:
    """Run one full headless episode.

    Args:
        policy: callable obs (np.ndarray (8,)) -> (steer, throttle), each in [-1, 1].
        track: loaded Track.
        seed: episode seed.
        env: reusable RaceEnv, or None to construct one.

    Returns:
        {"fitness": float, "checkpoints_passed": int, "frames_alive": int,
         "reason": str}
    """
    raise NotImplementedError
