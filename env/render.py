"""Pygame drawing and the raycast debug overlay.

Never imported by the evolution loop: no render() or clock.tick() during
evolution. Used for W1 manual driving, debugging, and champion videos.

Manual drive:  python -m env.render [track]     (track: box, oval, rounded_rect, wavy, peanut)
"""

import math
import os
import sys
from functools import lru_cache

import pygame

import simple_tracks
from env.car import MAX_SPEED, SENSOR_ANGLES, CarState, cast_rays_raw
from env.game import RaceEnv
from env.track import Point, Track, load_track

SCREEN_SIZE = (1000, 700)  # pixels; matches the simple_tracks canvas assumption
FPS = 60                   # frames / second, display only
TRACK_NAMES = ["box"] + list(simple_tracks.TRACKS)

BG = (20, 20, 24)
WALL = (220, 220, 220)
CHECKPOINT = (110, 45, 45)
NEXT_CHECKPOINT = (60, 210, 90)
RAY = (80, 160, 255)
HIT = (255, 80, 80)
TEXT = (235, 235, 235)
CAR_SPRITE_PX = 28         # car sprite side, pixels: ~25 px nose to tail, ~14 px wide


def make_icon() -> pygame.Surface:
    """Window / Dock icon: a top-down race car, nose up, drawn on a 128x128 transparent canvas."""
    icon = pygame.Surface((128, 128), pygame.SRCALPHA)
    body, dark, rim, white = (215, 35, 45), (35, 35, 40), (150, 150, 160), (240, 240, 240)
    for x, y, w, h in [(36, 18, 12, 22), (80, 18, 12, 22),     # front tires
                       (32, 76, 16, 30), (80, 76, 16, 30)]:    # rear tires, bigger
        pygame.draw.rect(icon, dark, (x, y, w, h), border_radius=4)
        pygame.draw.rect(icon, rim, (x, y, w, h), 2, border_radius=4)   # visible on dark backgrounds
    pygame.draw.rect(icon, body, (38, 6, 52, 7), border_radius=2)          # front wing, thin
    pygame.draw.polygon(icon, body, [(60, 8), (68, 8), (73, 42), (55, 42)])  # nose
    pygame.draw.rect(icon, body, (48, 38, 32, 68), border_radius=10)       # sidepods
    pygame.draw.rect(icon, body, (36, 106, 56, 14), border_radius=3)       # rear wing, thick
    pygame.draw.rect(icon, white, (62, 13, 4, 93))                         # stripe, wing to wing
    pygame.draw.ellipse(icon, dark, (56, 48, 16, 26))                      # cockpit
    return icon


@lru_cache(maxsize=1)
def _car_sprite() -> pygame.Surface:
    """The icon shrunk to CAR_SPRITE_PX, built once."""
    return pygame.transform.smoothscale(make_icon(), (CAR_SPRITE_PX, CAR_SPRITE_PX))


def init_display(size: tuple[int, int] = SCREEN_SIZE) -> pygame.Surface:
    """Open the pygame window.

    Args:
        size: (width, height), pixels.

    Returns:
        The display Surface.
    """
    pygame.init()
    pygame.display.set_caption("evo-racer")
    pygame.display.set_icon(make_icon())   # before set_mode: some platforms ignore later changes
    return pygame.display.set_mode(size)


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
    if show_checkpoints:
        for i, (a, b) in enumerate(track.checkpoints):
            if i != next_checkpoint:
                pygame.draw.line(surface, CHECKPOINT, a, b, 1)
    if next_checkpoint is not None and track.checkpoints:
        a, b = track.checkpoints[next_checkpoint]
        pygame.draw.line(surface, NEXT_CHECKPOINT, a, b, 3)
    for a, b in track.walls:
        pygame.draw.line(surface, WALL, a, b, 2)


def draw_car(surface: pygame.Surface, state: CarState) -> None:
    """Draw the car as a top-down race car pointing along its heading.

    Cosmetic only: collisions still use the CAR_RADIUS circle at (x, y).

    Args:
        surface: target Surface.
        state: CarState (pixels, radians).

    Returns:
        None.
    """
    # make_icon() draws the nose pointing up (heading -90 deg); rotozoom turns counterclockwise.
    sprite = pygame.transform.rotozoom(_car_sprite(), -math.degrees(state.heading) - 90, 1.0)
    surface.blit(sprite, sprite.get_rect(center=(state.x, state.y)))


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
    font = pygame.font.Font(None, 18)
    for rel, (dist, hit) in zip(SENSOR_ANGLES, readings):
        ang = state.heading + rel
        end = hit or (state.x + dist * math.cos(ang), state.y + dist * math.sin(ang))
        pygame.draw.line(surface, RAY, (state.x, state.y), end, 1)
        if hit:
            pygame.draw.circle(surface, HIT, hit, 4)
        label = font.render(f"{dist:.0f}", True, HIT if hit else RAY)
        surface.blit(label, (end[0] + 4, end[1] - 6))


def draw_hud(surface: pygame.Surface, lines: list[str]) -> None:
    """Draw text lines (speed, checkpoints, frame, fitness) in a corner.

    Args:
        surface: target Surface.
        lines: text to draw, top to bottom.

    Returns:
        None.
    """
    font = pygame.font.Font(None, 22)
    for i, line in enumerate(lines):
        surface.blit(font.render(line, True, TEXT), (10, 10 + 20 * i))


def draw_hud_bottom(surface: pygame.Surface, line: str) -> None:
    """Draw one text line in the bottom-left corner (controls hint)."""
    font = pygame.font.Font(None, 22)
    surface.blit(font.render(line, True, TEXT), (10, surface.get_height() - 24))


def _lap_line(env: RaceEnv) -> str:
    """HUD line: laps completed, current lap time, average and fastest lap, in seconds at FPS."""
    if not env.track.checkpoints:
        return "laps --"

    def secs(frames: float) -> str:
        return f"{frames / FPS:.2f} s"

    laps = env.lap_frames
    avg = secs(sum(laps) / len(laps)) if laps else "--"
    best = secs(min(laps)) if laps else "--"
    return f"laps {len(laps)}   this lap {secs(env.current_lap_frames)}   avg {avg}   fastest {best}"


def _draw_frame(surface: pygame.Surface, env: RaceEnv, show_checkpoints: bool,
                hud: list[str]) -> None:
    track, state = env.track, env.state
    surface.fill(BG)
    draw_track(surface, track, show_checkpoints,
               env.next_checkpoint if track.checkpoints else None)
    draw_rays(surface, state, cast_rays_raw(state, track.walls, track.grid))
    draw_car(surface, state)
    draw_hud(surface, hud)


def manual_drive(track_name: str = "oval") -> None:
    """W1 gate: drive the car with the keyboard, rays drawn on screen.

    Keys: arrows / WASD drive, R restart, N random spawn, TAB next track,
    C toggle checkpoints, ESC quit. No stall or frame limit, so you can sit
    still and read the ray distances. On a crash the car freezes so you can
    inspect the rays at the moment of contact.

    Args:
        track_name: key into simple_tracks.TRACKS, or "box" for the 4-wall
            sanity box.

    Returns:
        None. Runs until the window is closed.
    """
    surface = init_display()
    clock = pygame.time.Clock()
    env = RaceEnv(max_frames=None, stall_frames=None)
    track_idx = TRACK_NAMES.index(track_name)
    seed = None
    show_checkpoints = True
    env.reset(load_track(track_name), seed)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_c:
                show_checkpoints = not show_checkpoints
            elif event.key in (pygame.K_r, pygame.K_n, pygame.K_TAB):
                if event.key == pygame.K_TAB:
                    track_idx = (track_idx + 1) % len(TRACK_NAMES)
                    seed = None
                elif event.key == pygame.K_n:
                    seed = 0 if seed is None else seed + 1
                env.reset(load_track(TRACK_NAMES[track_idx]), seed)

        keys = pygame.key.get_pressed()
        steer = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
        throttle = (keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s])
        if env.reason is None:
            env.step(float(steer), float(throttle))

        s = env.state
        status = "driving" if env.reason is None else f"ENDED: {env.reason}  (R to restart)"
        _draw_frame(surface, env, show_checkpoints, [
            f"track {env.track.name}   spawn seed {seed}",
            f"speed {s.speed:.2f} / {MAX_SPEED} px/frame   heading {math.degrees(s.heading) % 360:.0f} deg",
            f"checkpoints {env.checkpoints_passed}   next {env.next_checkpoint}   frame {env.frames_alive}",
            _lap_line(env),
            f"fitness {env.fitness():.1f}",
            status,
        ])
        draw_hud_bottom(surface, "arrows/WASD drive   R restart   N random spawn   "
                                 "TAB next track   C checkpoints   ESC quit")
        pygame.display.flip()
        clock.tick(FPS)


def replay(policy, track_name: str, seed: int | None, out_path: str | None = None) -> None:
    """Render a policy driving one episode; optionally save frames for video.

    Args:
        policy: callable obs (np.ndarray (8,)) -> (steer, throttle), each in [-1, 1].
        track_name: key into simple_tracks.TRACKS.
        seed: episode seed passed to RaceEnv.reset.
        out_path: directory for PNG frames, or None to only display.

    Returns:
        None.
    """
    surface = init_display()
    clock = pygame.time.Clock()
    env = RaceEnv()
    obs = env.reset(load_track(track_name), seed)
    if out_path:
        os.makedirs(out_path, exist_ok=True)

    done = False
    while not done:
        if any(e.type == pygame.QUIT for e in pygame.event.get()):
            break
        obs, done, _ = env.step(*policy(obs))
        _draw_frame(surface, env, True, [
            f"track {track_name}   seed {seed}   frame {env.frames_alive}",
            f"checkpoints {env.checkpoints_passed}   fitness {env.fitness():.1f}",
            _lap_line(env),
            f"ended: {env.reason}" if done else "",
        ])
        pygame.display.flip()
        if out_path:
            pygame.image.save(surface, os.path.join(out_path, f"frame_{env.frames_alive:05d}.png"))
        else:
            clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    manual_drive(sys.argv[1] if len(sys.argv) > 1 else "oval")
