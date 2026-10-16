"""RaceEnv: headless episode loop, checkpoint tracking, fitness.

Fitness = checkpoints_passed * 100 + (checkpoints_passed / frames_alive) * 500
Episode ends on wall contact, on no new checkpoint for STALL_FRAMES frames, or
at MAX_FRAMES. Raw distance traveled is never rewarded.
"""

import math

import numpy as np

from env.car import CarState, collides, observation, step_physics
from env.track import Segment, Track, spawn_pose

STALL_FRAMES = 150      # frames without a new checkpoint before termination
MAX_FRAMES = 600        # hard episode cap, frames: ~2.8 laps of the longest track
                        # (rounded_rect, ~2155 px) at MAX_SPEED (10 px/frame)
SPAWN_JITTER_DEG = 5.0  # seeded spawns get heading jitter in +/- this, degrees


def compute_fitness(checkpoints_passed: int, frames_alive: int) -> float:
    """Episode fitness.

    Args:
        checkpoints_passed: count of checkpoints crossed in order (can exceed
            len(checkpoints) if the car laps).
        frames_alive: frames survived, >= 1 (0 is treated as 1).

    Returns:
        checkpoints_passed * 100 + (checkpoints_passed / frames_alive) * 500.
    """
    return checkpoints_passed * 100 + (checkpoints_passed / max(frames_alive, 1)) * 500


def crossed(p0: tuple[float, float], p1: tuple[float, float], seg: Segment) -> bool:
    """Whether the car's motion this frame crossed a checkpoint line, forwards.

    Args:
        p0: car position at the start of the frame, pixels.
        p1: car position at the end of the frame, pixels.
        seg: checkpoint ((ix, iy), (ox, oy)), pixels.

    Returns:
        True if segment p0->p1 intersects `seg` while moving in the racing
        direction. Backward crossings return False, so driving back and forth
        over a line can't farm checkpoints.
    """
    (ix, iy), (ox, oy) = seg
    mx, my = p1[0] - p0[0], p1[1] - p0[1]
    cx, cy = ix - ox, iy - oy   # outer -> inner, perpendicular to the racing direction
    # cross(motion, outer->inner) > 0 exactly when motion points forward
    # (simple_tracks puts inner at centerline + left normal, in y-up math terms).
    denom = mx * cy - my * cx
    if denom <= 0.0:
        return False
    # Solve p0 + t*motion = outer + u*(outer->inner).
    qx, qy = ox - p0[0], oy - p0[1]
    t = (qx * cy - qy * cx) / denom
    u = (qx * my - qy * mx) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


class RaceEnv:
    """Headless racing episode. No pygame dependency."""

    def __init__(self, max_frames: int | None = MAX_FRAMES,
                 stall_frames: int | None = STALL_FRAMES):
        """Configure termination limits.

        Args:
            max_frames: hard episode cap, frames. None = no cap (manual drive only).
            stall_frames: frames allowed without a new checkpoint. None = no limit.

        Returns:
            None. Call reset() before step().
        """
        self.max_frames = max_frames
        self.stall_frames = stall_frames
        self.track: Track | None = None
        self._state: CarState | None = None
        self._next = 0
        self.checkpoints_passed = 0
        self.frames_alive = 0
        self._last_checkpoint_frame = 0
        self.lap_frames: list[int] = []   # frames per completed lap, in order
        self._lap_start_frame = 0
        self.reason: str | None = None

    def reset(self, track: Track, seed: int | None) -> np.ndarray:
        """Start a new episode.

        Args:
            track: loaded Track.
            seed: episode seed. Picks the spawn checkpoint and a heading jitter
                in +/- SPAWN_JITTER_DEG, deterministically given (track, seed).
                All genomes in a generation must receive the same seed so they
                face identical conditions. None = canonical start (checkpoint 0,
                no jitter), for manual driving and videos.

        Returns:
            Initial observation, np.ndarray shape (8,), float in [0, 1]
            (see car.observation).
        """
        n = len(track.checkpoints)
        if seed is None or n == 0:
            k, jitter = 0, 0.0
        else:
            rng = np.random.default_rng(seed)
            k = int(rng.integers(n))
            jitter = math.radians(rng.uniform(-SPAWN_JITTER_DEG, SPAWN_JITTER_DEG))

        x, y, heading = spawn_pose(track, k * track.checkpoint_every)
        self.track = track
        self._state = CarState(x, y, heading + jitter)
        self._next = (k + 1) % n if n else 0   # spawn sits on checkpoint k
        self.checkpoints_passed = 0
        self.frames_alive = 0
        self._last_checkpoint_frame = 0
        self.lap_frames = []
        self._lap_start_frame = 0
        self.reason = None
        return observation(self._state, track.walls, track.grid)

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
        if self.reason is not None:
            raise RuntimeError("episode is over; call reset()")
        track, s = self.track, self._state
        p0 = (s.x, s.y)
        step_physics(s, steer, throttle)
        self.frames_alive += 1

        if collides(s, track.walls, track.grid):
            self.reason = "wall"   # a checkpoint crossed on the crash frame doesn't count
        else:
            if track.checkpoints and crossed(p0, (s.x, s.y), track.checkpoints[self._next]):
                self.checkpoints_passed += 1
                self._next = (self._next + 1) % len(track.checkpoints)
                self._last_checkpoint_frame = self.frames_alive
                # A lap = every checkpoint once, back to the spawn checkpoint.
                if self.checkpoints_passed % len(track.checkpoints) == 0:
                    self.lap_frames.append(self.frames_alive - self._lap_start_frame)
                    self._lap_start_frame = self.frames_alive
            if (self.stall_frames is not None
                    and self.frames_alive - self._last_checkpoint_frame >= self.stall_frames):
                self.reason = "stall"
            elif self.max_frames is not None and self.frames_alive >= self.max_frames:
                self.reason = "max_frames"

        info = {"checkpoints_passed": self.checkpoints_passed,
                "frames_alive": self.frames_alive,
                "reason": self.reason}
        return observation(s, track.walls, track.grid), self.reason is not None, info

    @property
    def state(self) -> CarState:
        """Current CarState (pixels, radians). For rendering only."""
        return self._state

    @property
    def current_lap_frames(self) -> int:
        """Frames since the current lap started (spawn or last lap completion)."""
        return self.frames_alive - self._lap_start_frame

    @property
    def next_checkpoint(self) -> int:
        """Index into track.checkpoints of the next checkpoint to cross."""
        return self._next

    def fitness(self) -> float:
        """Fitness of the episode so far.

        Returns:
            compute_fitness(checkpoints_passed, frames_alive).
        """
        return compute_fitness(self.checkpoints_passed, self.frames_alive)


def run_episode(policy, track: Track, seed: int | None, env: RaceEnv | None = None) -> dict:
    """Run one full headless episode.

    Args:
        policy: callable obs (np.ndarray (8,)) -> (steer, throttle), each in [-1, 1].
        track: loaded Track.
        seed: episode seed.
        env: reusable RaceEnv, or None to construct one. Needs max_frames or
            stall_frames set, or the episode may never end.

    Returns:
        {"fitness": float, "checkpoints_passed": int, "frames_alive": int,
         "reason": str}
    """
    env = env or RaceEnv()
    obs = env.reset(track, seed)
    done = False
    while not done:
        steer, throttle = policy(obs)
        obs, done, _ = env.step(steer, throttle)
    return {"fitness": env.fitness(),
            "checkpoints_passed": env.checkpoints_passed,
            "frames_alive": env.frames_alive,
            "reason": env.reason}
