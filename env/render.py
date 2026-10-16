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
from env.game import RaceEnv, run_episode
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
GHOST_BODY = (175, 175, 182)  # race(): body colour of the non-leading cars
GHOST_ALPHA = 150             # race(): 0-255 opacity of the non-leading cars
QUIET_TEXT = (130, 130, 138)


def make_icon(body: tuple[int, int, int] = (215, 35, 45)) -> pygame.Surface:
    """Window / Dock icon: a top-down race car, nose up, drawn on a 128x128 transparent canvas."""
    icon = pygame.Surface((128, 128), pygame.SRCALPHA)
    dark, rim, white = (35, 35, 40), (150, 150, 160), (240, 240, 240)
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


@lru_cache(maxsize=2)
def _car_sprite(ghost: bool = False) -> pygame.Surface:
    """The icon shrunk to CAR_SPRITE_PX, built once. ghost = gray and see-through."""
    sprite = pygame.transform.smoothscale(make_icon(GHOST_BODY if ghost else (215, 35, 45)),
                                          (CAR_SPRITE_PX, CAR_SPRITE_PX))
    if ghost:
        sprite.fill((255, 255, 255, GHOST_ALPHA), special_flags=pygame.BLEND_RGBA_MULT)
    return sprite


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


def draw_car(surface: pygame.Surface, state: CarState, ghost: bool = False) -> None:
    """Draw the car as a top-down race car pointing along its heading.

    Cosmetic only: collisions still use the CAR_RADIUS circle at (x, y).

    Args:
        surface: target Surface.
        state: CarState (pixels, radians).
        ghost: draw it gray and see-through (race() uses this for non-leaders).

    Returns:
        None.
    """
    # make_icon() draws the nose pointing up (heading -90 deg); rotozoom turns counterclockwise.
    sprite = pygame.transform.rotozoom(_car_sprite(ghost), -math.degrees(state.heading) - 90, 1.0)
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


def draw_hud(surface: pygame.Surface, lines: list[str],
             colors: list[tuple[int, int, int]] | None = None) -> None:
    """Draw text lines (speed, checkpoints, frame, fitness) in a corner.

    Args:
        surface: target Surface.
        lines: text to draw, top to bottom.
        colors: one RGB per line, or None for TEXT on every line.

    Returns:
        None.
    """
    font = pygame.font.Font(None, 22)
    for i, line in enumerate(lines):
        surface.blit(font.render(line, True, colors[i] if colors else TEXT), (10, 10 + 20 * i))


def draw_hud_bottom(surface: pygame.Surface, line: str) -> None:
    """Draw one text line in the bottom-left corner (controls hint)."""
    font = pygame.font.Font(None, 22)
    surface.blit(font.render(line, True, TEXT), (10, surface.get_height() - 24))


def draw_table(surface: pygame.Surface, rows: list[list[str]],
               colors: list[tuple[int, int, int]], top: int = 10) -> None:
    """Text rows in aligned columns on a dark see-through panel, top-left.

    Args:
        surface: target Surface.
        rows: cells per row; every row has the same number of cells.
        colors: one RGB per row.
        top: y of the panel's top edge, pixels.
    """
    font = pygame.font.Font(None, 22)
    gap, pad, line = 18, 8, 20
    widths = [max(font.size(r[c])[0] for r in rows) for c in range(len(rows[0]))]
    panel = pygame.Surface((sum(widths) + gap * (len(widths) - 1) + 2 * pad,
                            line * len(rows) + 2 * pad - 4), pygame.SRCALPHA)
    panel.fill((*BG, 210))
    surface.blit(panel, (10 - pad, top))
    for r, (cells, color) in enumerate(zip(rows, colors)):
        x = 10
        for cell, w in zip(cells, widths):
            surface.blit(font.render(cell, True, color), (x, top + pad + line * r))
            x += w + gap


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


def race(policies: list, labels: list[str], track_name: str, seed: int | None,
         out_path: str | None = None) -> None:
    """Replay several drivers at once from the same start: the fastest in colour
    with its rays, every other one as a gray see-through ghost.

    Fastest = highest episode fitness (most track covered, then speed), found by
    one quick headless run per driver first, so the highlight is fixed from
    frame 1. Cars don't collide with each other. The window stays open on the
    final frame until closed (ESC); with out_path it saves frames and exits.

    Args:
        policies: callables obs (np.ndarray (8,)) -> (steer, throttle).
        labels: one name per policy, shown in the legend.
        track_name: key into simple_tracks.TRACKS.
        seed: episode seed shared by every car (None = canonical start).
        out_path: directory for PNG frames, or None to only display.

    Returns:
        None.
    """
    track = load_track(track_name)
    scores = [run_episode(p, track, seed)["fitness"] for p in policies]
    lead = max(range(len(policies)), key=scores.__getitem__)
    order = sorted(range(len(policies)), key=scores.__getitem__, reverse=True)

    surface = init_display()
    clock = pygame.time.Clock()
    envs = [RaceEnv() for _ in policies]
    obs = [env.reset(track, seed) for env in envs]
    if out_path:
        os.makedirs(out_path, exist_ok=True)

    frame = 0
    while True:
        if any(e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
               for e in pygame.event.get()):
            break
        running = [i for i, env in enumerate(envs) if env.reason is None]
        for i in running:
            obs[i], _, _ = envs[i].step(*policies[i](obs[i]))
        if running:
            frame += 1

        leader = envs[lead]
        surface.fill(BG)
        draw_track(surface, track, True, leader.next_checkpoint if track.checkpoints else None)
        for i, env in enumerate(envs):
            if i != lead:
                draw_car(surface, env.state, ghost=True)
        draw_rays(surface, leader.state, cast_rays_raw(leader.state, track.walls, track.grid))
        draw_car(surface, leader.state)

        draw_hud(surface, [f"track {track_name}   seed {seed}   frame {frame}"
                           + ("" if running else "   -   all cars done, ESC to close")])
        rows = [["", "driver", "laps", "best lap", "status"]]
        colors = [QUIET_TEXT]
        for rank, i in enumerate(order, 1):
            env = envs[i]
            status = {"wall": "crashed", None: "running"}.get(env.reason, "finished")
            rows.append([f"{rank}.", labels[i] + ("  (fastest)" if i == lead else ""), str(len(env.lap_frames)),
                         f"{min(env.lap_frames) / FPS:.2f} s" if env.lap_frames else "--", status])
            colors.append(TEXT if i == lead else QUIET_TEXT)
        draw_table(surface, rows, colors, top=36)
        pygame.display.flip()

        if out_path:
            if not running:
                break
            pygame.image.save(surface, os.path.join(out_path, f"frame_{frame:05d}.png"))
        else:
            clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    manual_drive(sys.argv[1] if len(sys.argv) > 1 else "oval")
