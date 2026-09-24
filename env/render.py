"""Pygame drawing and the raycast debug overlay.

Never imported by the evolution loop: no render() or clock.tick() during
evolution. Used for W1 manual driving, debugging, and champion videos.
"""

import pygame

from env.car import CarState
from env.track import Point, Track

SCREEN_SIZE = (1000, 700)  # pixels; matches the simple_tracks canvas assumption
FPS = 60                   # frames / second, display only


def init_display(size: tuple[int, int] = SCREEN_SIZE) -> pygame.Surface:
    """Open the pygame window.

    Args:
        size: (width, height), pixels.

    Returns:
        The display Surface.
    """
    raise NotImplementedError


def draw_track(surface: pygame.Surface, track: Track, show_checkpoints: bool = True,
               next_checkpoint: int | None = None) -> None:
    """Draw walls, and optionally checkpoints.

    Args:
        surface: target Surface.
        track: loaded Track (pixels).
        show_checkpoints: draw checkpoint lines if True.
        next_checkpoint: index of the checkpoint the car must cross next;
            highlighted if given.

    Returns:
        None.
    """
    raise NotImplementedError


def draw_car(surface: pygame.Surface, state: CarState) -> None:
    """Draw the car body and heading indicator.

    Args:
        surface: target Surface.
        state: CarState (pixels, radians).

    Returns:
        None.
    """
    raise NotImplementedError


def draw_rays(surface: pygame.Surface, state: CarState,
              readings: list[tuple[float, Point | None]]) -> None:
    """Debug overlay: draw every sensor ray and its hit point.

    Args:
        surface: target Surface.
        state: CarState (pixels, radians).
        readings: output of car.cast_rays_raw, one (distance_pixels,
            hit_point_or_None) per sensor.

    Returns:
        None.
    """
    # TODO: label each ray with its distance so wrong values are obvious.
    raise NotImplementedError


def draw_hud(surface: pygame.Surface, lines: list[str]) -> None:
    """Draw text lines (speed, checkpoints, frame, fitness) in a corner.

    Args:
        surface: target Surface.
        lines: text to draw, top to bottom.

    Returns:
        None.
    """
    raise NotImplementedError


def manual_drive(track_name: str = "oval") -> None:
    """W1 gate: drive the car with the keyboard, rays drawn on screen.

    Args:
        track_name: key into simple_tracks.TRACKS, or "box" for the 4-wall
            sanity box.

    Returns:
        None. Runs until the window is closed.
    """
    # TODO: arrow keys -> (steer, throttle) -> RaceEnv.step; draw every frame.
    raise NotImplementedError


def replay(policy, track_name: str, seed: int, out_path: str | None = None) -> None:
    """Render a policy driving one episode; optionally save frames for video.

    Args:
        policy: callable obs (np.ndarray (8,)) -> (steer, throttle), each in [-1, 1].
        track_name: key into simple_tracks.TRACKS.
        seed: episode seed passed to RaceEnv.reset.
        out_path: directory for PNG frames, or None to only display.

    Returns:
        None.
    """
    raise NotImplementedError


if __name__ == "__main__":
    manual_drive()
