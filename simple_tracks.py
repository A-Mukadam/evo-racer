"""
Simple parametric tracks for the neuroevolution racing project.

Emits the same format as the real-circuit CSV loader, so training code
never needs to know which kind of track it got:

    walls       : list[((x1,y1),(x2,y2))]   inner loop + outer loop segments
    checkpoints : list[((ix,iy),(ox,oy))]   ordered, inner point -> outer point
    centerline  : np.ndarray (N,2)          for spawn position + heading

Run directly to preview all tracks:  python simple_tracks.py
"""

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


def peanut(n=280, rx=380.0, ry=230.0):
    """One tight end, one open end. Good generalization hold-out."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1.0 - 0.28 * np.cos(t) ** 2
    return np.stack([CX + rx * r * np.cos(t), CY + ry * r * np.sin(t)], axis=1)


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
}


def make(name, half_width=55.0, checkpoint_every=8):
    return build_track(TRACKS[name](), half_width, checkpoint_every)


# --------------------------------------------------------------------- preview

def _preview():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
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