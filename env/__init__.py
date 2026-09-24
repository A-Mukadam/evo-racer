"""Racing environment: tracks, car physics, raycast sensors, rendering.

Coordinate conventions (shared by every module in this package):
    position : pixels, origin top-left, y grows downward (screen coords)
    heading  : radians, 0 = +x, measured with atan2 in screen coords
    time     : frames (fixed timestep, no wall-clock dependence)
    speed    : pixels / frame
"""
