"""One-way multipath eigenpaths in a flat waveguide (method of images).

The water column is bounded by the surface (z = 0) and a flat seabed
(z = -depth). Sound travels in straight lines at the depth-averaged sound
speed, so every eigenpath with n boundary reflections is a straight line to
an image of the end point, and all its reflections share one grazing angle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SURFACE = "surface"
BOTTOM = "bottom"


def _reflect(point: np.ndarray, boundary: str, depth: float) -> np.ndarray:
    p = np.array(point, dtype=float)
    p[2] = -p[2] if boundary == SURFACE else -2.0 * depth - p[2]
    return p


def _sequences(max_bounces: int) -> list[tuple[str, ...]]:
    """Boundary sequences up to max_bounces; reflections must alternate."""
    seqs = [()]
    for n in range(1, max_bounces + 1):
        for first, second in ((SURFACE, BOTTOM), (BOTTOM, SURFACE)):
            seqs.append(tuple(first if i % 2 == 0 else second for i in range(n)))
    return seqs


@dataclass(frozen=True)
class Path:
    """A one-way eigenpath from a start point to an end point.

    ``departure`` is the unit vector leaving the start point; ``arrival`` is
    the unit vector at the end point pointing back along the incoming ray.
    """

    boundaries: tuple[str, ...]
    length: float
    grazing: float
    departure: np.ndarray
    arrival: np.ndarray

    @property
    def name(self) -> str:
        return "-".join(self.boundaries) if self.boundaries else "direct"

    @property
    def n_surface(self) -> int:
        return self.boundaries.count(SURFACE)

    @property
    def n_bottom(self) -> int:
        return self.boundaries.count(BOTTOM)

    def delay(self, sound_speed: float) -> float:
        return self.length / sound_speed

    def amplitude(self, frequency_hz, environment) -> np.ndarray:
        """Complex pressure ratio along the path, excluding the travel-time phase.

        Spherical spreading 1/L, Francois-Garrison absorption, Eckart surface
        and Rayleigh bottom reflection coefficients.
        """
        f = np.asarray(frequency_hz, dtype=float)
        alpha = environment.absorption_db_per_km(f)
        amp = (10.0 ** (-alpha * self.length / 20000.0) / self.length).astype(complex)
        if self.n_surface:
            amp *= environment.surface_reflection(f, self.grazing) ** self.n_surface
        if self.n_bottom:
            amp *= environment.bottom_reflection(self.grazing) ** self.n_bottom
        return amp


def eigenpaths(start, end, depth: float, max_bounces: int = 1) -> list[Path]:
    """All eigenpaths between two points with at most ``max_bounces`` reflections."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    for name, p in (("start", start), ("end", end)):
        if not -depth < p[2] < 0.0:
            raise ValueError(f"{name} point z={p[2]:.2f} m is outside the water column")

    paths = []
    for seq in _sequences(max_bounces):
        end_image = end
        for b in reversed(seq):
            end_image = _reflect(end_image, b, depth)
        start_image = start
        for b in seq:
            start_image = _reflect(start_image, b, depth)
        d = end_image - start
        length = float(np.linalg.norm(d))
        grazing = float(np.arcsin(abs(d[2]) / length))
        back = start_image - end
        paths.append(Path(seq, length, grazing, d / length, back / np.linalg.norm(back)))
    return paths
