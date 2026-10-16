"""
Simple parametric tracks for the neuroevolution racing project.

Emits the same format as the real-circuit CSV loader, so training code
never needs to know which kind of track it got:

    walls       : list[((x1,y1),(x2,y2))]   inner loop + outer loop segments
    checkpoints : list[((ix,iy),(ox,oy))]   ordered, inner point -> outer point
    centerline  : np.ndarray (N,2)          for spawn position + heading

Run directly to preview all tracks:  python simple_tracks.py
"""

import math
from functools import partial

import numpy as np

# ---------------------------------------------------------------- centerlines
# Each returns an (N,2) array of points forming a CLOSED loop.
# Canvas assumption: roughly 1000 x 700 pixels, origin top-left.

CX, CY = 500.0, 350.0   # canvas center


def oval(n=200, rx=380.0, ry=240.0):
    """The sanity track. If the GA can't solve this, the bug is yours."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([CX + rx * np.cos(t), CY + ry * np.sin(t)], axis=1)


def rounded_rect(n=240, hw=380.0, hh=230.0, r=120.0):
    """Four straights, four corners. Tests braking-ish behavior."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # superellipse: exponent controls corner sharpness
    p = 4.0
    x = np.abs(np.cos(t)) ** (2 / p) * np.sign(np.cos(t))
    y = np.abs(np.sin(t)) ** (2 / p) * np.sign(np.sin(t))
    return np.stack([CX + hw * x, CY + hh * y], axis=1)


def wavy(n=260, rx=360.0, ry=220.0, lobes=3, amp=0.18):
    """Gentle S-curves. First track that actually needs steering skill."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1.0 + amp * np.sin(lobes * t)
    return np.stack([CX + rx * r * np.cos(t), CY + ry * r * np.sin(t)], axis=1)


def peanut(n=280, rx=330.0, ry=200.0, a=0.3, b=0.25):
    """One tight end, one open end, pinched waist. Generalization hold-out.

    `a` makes the ends unequal (tight end on the left), `b` pinches the waist
    so the sides bend the other way. Tightest turn ~66 px radius, tighter than
    any training track (wavy ~75 px).
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1.0 + a * np.cos(t) + b * np.cos(2 * t)
    pts = np.stack([CX + rx * r * np.cos(t), CY + ry * r * np.sin(t)], axis=1)
    pts[:, 0] += CX - (pts[:, 0].min() + pts[:, 0].max()) / 2   # `a` shifts it right; recenter
    return pts


# ---------------------------------------------------------- F1-inspired courses
# Built from pieces like a real circuit map: ("S", length) is a straight,
# ("R", radius, degrees) / ("L", radius, degrees) a right / left arc (R turns
# clockwise on screen). Exactly two straights have length None: their lengths
# are solved so the loop closes. Arcs are >= 65 px radius so the walls
# (55 px half-width) never fold over. Run validate() after editing a layout.

F1_PIECES = {
    # Monza-inspired: long straights broken by three chicanes, big sweepers.
    "f1_chicanes": [("S", None), ("R", 70, 50), ("L", 70, 100), ("R", 70, 50), ("S", 60),
                    ("R", 150, 90), ("S", 30), ("L", 70, 40), ("R", 70, 80), ("L", 70, 40),
                    ("R", 90, 90), ("S", 380), ("L", 80, 40), ("R", 80, 80), ("L", 80, 40),
                    ("R", 100, 90), ("S", None), ("R", 100, 90)],
    # Monaco-inspired: two 65 px hairpins at the ends of narrow fingers, many 90 deg corners.
    "f1_hairpins": [("S", None), ("R", 70, 90), ("S", 120), ("L", 65, 180), ("S", 120), ("R", 70, 90),
                    ("S", 380), ("R", 70, 90), ("S", None), ("R", 70, 90), ("S", 60),
                    ("R", 70, 90), ("S", 120), ("L", 65, 180), ("S", 120), ("R", 70, 90),
                    ("S", 380), ("R", 70, 90), ("S", 350), ("R", 70, 90)],
    # Suzuka-inspired: deep esses (too wavy to drive straight through), a
    # double-apex Spoon, a final chicane.
    "f1_esses": [("S", None), ("R", 70, 75), ("L", 70, 150), ("R", 70, 150), ("L", 70, 150),
                 ("R", 70, 75), ("S", 60), ("R", 90, 90), ("S", None), ("R", 90, 90), ("S", 615),
                 ("R", 70, 45), ("S", 40), ("R", 70, 45), ("S", 40), ("L", 70, 40), ("R", 70, 80),
                 ("L", 70, 40), ("S", 20), ("R", 100, 90)],
    # Interlagos-inspired, COUNTER-clockwise (every other track runs clockwise):
    # Senna S, long left sweeper, stepped infield.
    "f1_reverse": [("S", 180), ("L", 70, 90), ("S", None), ("L", 75, 40), ("R", 75, 80), ("L", 75, 40),
                   ("L", 120, 90), ("S", None), ("L", 80, 90), ("S", 40), ("L", 70, 90), ("S", 60),
                   ("R", 70, 90), ("S", 30), ("L", 70, 90), ("S", 40), ("R", 70, 90), ("S", 30),
                   ("L", 70, 90)],
}


def _trace(pieces, auto_lengths, step=1.0):
    """Walk the pieces from (0, 0) heading 0. Returns (points, auto-straight unit directions, final heading)."""
    x = y = h = 0.0
    pts, dirs, k = [(x, y)], [], 0
    for piece in pieces:
        if piece[0] == "S":
            length = piece[1]
            if length is None:
                length = auto_lengths[k]
                dirs.append((math.cos(h), math.sin(h)))
                k += 1
            n = max(1, int(length / step))
            for _ in range(n):
                x += length / n * math.cos(h)
                y += length / n * math.sin(h)
                pts.append((x, y))
        else:
            kind, radius, degrees = piece
            n = max(2, int(radius * math.radians(degrees) / step))
            dh = (1 if kind == "R" else -1) * math.radians(degrees) / n
            for _ in range(n):           # midpoint rule: exact radius to << 1 px
                h += dh / 2
                x += radius * abs(dh) * math.cos(h)
                y += radius * abs(dh) * math.sin(h)
                h += dh / 2
                pts.append((x, y))
    return np.array(pts), dirs, h


def _resample_loop(pts, spacing):
    """Evenly spaced points along a closed polyline (so checkpoints are evenly spaced too)."""
    closed = np.vstack([pts, pts[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    n = int(round(s[-1] / spacing))
    t = np.arange(n) * s[-1] / n
    return np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], axis=1)


def piece_loop(pieces, spacing=8.0):
    """Closed centerline from F1_PIECES-style pieces, centred on the canvas,
    one point every `spacing` px (checkpoint_every=8 -> a checkpoint every 64 px)."""
    _, dirs, heading = _trace(pieces, [0.0, 0.0])
    if len(dirs) != 2:
        raise ValueError("need exactly two straights with length None")
    if abs(abs(math.degrees(heading)) - 360) > 1e-6:
        raise ValueError(f"pieces turn {math.degrees(heading):.1f} deg in total, need +/-360")
    open_end = _trace(pieces, [0.0, 0.0])[0][-1]
    lengths = np.linalg.solve(np.array(dirs).T, -open_end)   # close the gap with the two auto straights
    if (lengths < 0).any():
        raise ValueError(f"auto straights would be negative: {lengths.round(1)}")
    pts = _trace(pieces, list(lengths))[0][:-1]
    pts += np.array([CX, CY]) - (pts.min(0) + pts.max(0)) / 2
    return _resample_loop(pts, spacing)


# ------------------------------------------------------------------ conversion

def build_track(centerline, half_width=55.0, checkpoint_every=8):
    """Offset a closed centerline into inner/outer walls + ordered checkpoints.

    half_width : pixels. Wider = easier. Start at 55-70 for week 1.
    """
    pts = np.asarray(centerline, dtype=float)

    # tangent via central difference, wrapping at the ends (closed loop)
    nxt = np.roll(pts, -1, axis=0)
    prv = np.roll(pts, 1, axis=0)
    tang = nxt - prv
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)

    # left normal (rotate tangent +90 degrees)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)

    inner = pts + normal * half_width
    outer = pts - normal * half_width

    def loop_segments(loop):
        return [(tuple(loop[i]), tuple(loop[(i + 1) % len(loop)]))
                for i in range(len(loop))]

    walls = loop_segments(inner) + loop_segments(outer)

    checkpoints = [(tuple(inner[i]), tuple(outer[i]))
                   for i in range(0, len(pts), checkpoint_every)]

    return walls, checkpoints, pts


def spawn_pose(centerline, index=0):
    """Starting (x, y, heading_radians) on the centerline."""
    pts = np.asarray(centerline, dtype=float)
    p = pts[index]
    d = pts[(index + 1) % len(pts)] - p
    return float(p[0]), float(p[1]), float(np.arctan2(d[1], d[0]))


TRACKS = {
    "oval": oval,
    "rounded_rect": rounded_rect,
    "wavy": wavy,
    "peanut": peanut,
    **{name: partial(piece_loop, pieces) for name, pieces in F1_PIECES.items()},
}
F1_TRACKS = tuple(F1_PIECES)


def validate(name, half_width=55.0, canvas=(1000, 700), margin=15):
    """Layout checks for a new track. Returns a list of problems ([] = OK):
    walls crossing each other, a turn too tight for the walls, or walls off the canvas."""
    walls, _, c = make(name, half_width)
    problems = []

    # tightest turn: radius of the circle through points 3 apart (smooths the 8 px steps)
    a, b, d = np.roll(c, 3, 0), c, np.roll(c, -3, 0)
    cross = np.abs((b - a)[:, 0] * (d - a)[:, 1] - (b - a)[:, 1] * (d - a)[:, 0])
    radius = (np.linalg.norm(b - a, axis=1) * np.linalg.norm(d - b, axis=1)
              * np.linalg.norm(d - a, axis=1) / (2 * cross + 1e-12))
    if radius.min() < 1.1 * half_width:
        problems.append(f"tightest turn {radius.min():.0f} px < 1.1 x half-width")

    pts = np.array([p for w in walls for p in w])
    if (pts.min(0) < margin).any() or (pts.max(0) > np.array(canvas) - margin).any():
        problems.append(f"walls leave the canvas: {pts.min(0).round()}..{pts.max(0).round()}")

    A = np.array([w[0] for w in walls])
    E = np.array([w[1] for w in walls]) - A
    for i in range(len(A)):   # proper crossings only; touching neighbours don't count
        den = E[i, 0] * E[:, 1] - E[i, 1] * E[:, 0]
        q = A - A[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (q[:, 0] * E[:, 1] - q[:, 1] * E[:, 0]) / den
            u = (q[:, 0] * E[i, 1] - q[:, 1] * E[i, 0]) / den
        if ((np.abs(den) > 1e-12) & (t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)).any():
            problems.append("walls cross")
            break
    return problems


def make(name, half_width=55.0, checkpoint_every=8):
    return build_track(TRACKS[name](), half_width, checkpoint_every)


# --------------------------------------------------------------------- preview

def _preview():
    import matplotlib.pyplot as plt

    rows = (len(TRACKS) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4.5 * rows))
    for ax, (name, fn) in zip(axes.ravel(), TRACKS.items()):
        walls, cps, mid = build_track(fn())
        for (a, b) in walls:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=1)
        for (a, b) in cps:
            ax.plot([a[0], b[0]], [a[1], b[1]],
                    color="red", linewidth=0.8, alpha=0.6)
        ax.plot(mid[:, 0], mid[:, 1], "--",
                color="blue", linewidth=0.8, alpha=0.5)
        sx, sy, sa = spawn_pose(fn())
        ax.plot([sx], [sy], "o", color="green", markersize=8)
        ax.set_title(f"{name}  ({len(walls)} walls, {len(cps)} checkpoints)")
        ax.set_aspect("equal")
        ax.invert_yaxis()          # screen coords: y grows downward
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("tracks_preview.png", dpi=110)
    print("wrote tracks_preview.png")


if __name__ == "__main__":
    _preview()