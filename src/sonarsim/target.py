"""Triangular-mesh targets and their Kirchhoff (physical-optics) scattering.

SonarSim uses the exp(+i 2 pi f t) time convention throughout: a delay tau
multiplies a spectrum by exp(-i 2 pi f tau). In that convention, for an
incident plane wave travelling along unit vector i and scattering towards
unit vector s, the far-field scattering amplitude of a body is

    f(i, s) = (i k / 2 pi) * sum_n g_n * Integral_{T_n} exp(-i k (i - s) . r) dS

over the illuminated facets T_n, where g_n = s . n_n for a rigid surface and
g_n = i . n_n for a pressure-release surface (n_n is the outward normal). The
integral over each triangle has a closed form, so facets need not be small
compared with the wavelength. Target strength is 20 log10 |f| (dB re 1 m).

Multiple scattering, edge diffraction, creeping waves, elastic responses and
self-shadowing between parts of the body are not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path as FilePath

import numpy as np

_AXIS = {"x": 0, "y": 1, "z": 2}


def _axis_vector(spec: str) -> np.ndarray:
    spec = spec.strip().lower()
    sign = -1.0 if spec.startswith("-") else 1.0
    v = np.zeros(3)
    v[_AXIS[spec.lstrip("+-")]] = sign
    return v


def _read_obj(path: FilePath) -> tuple[np.ndarray, np.ndarray]:
    """Minimal Wavefront OBJ reader: vertices and fan-triangulated faces."""
    vertices, faces = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
                faces.extend([idx[0], idx[j], idx[j + 1]] for j in range(1, len(idx) - 1))
    if not vertices or not faces:
        raise ValueError(f"No triangles found in {path}")
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


@dataclass
class Mesh:
    """Triangle mesh in its local frame: front along +x, up along +z.

    Use :meth:`from_file` to load OBJ (built-in) or STL/GLTF/PLY (needs
    trimesh). The mesh is centred on its bounding box, which becomes the
    reference point that a :class:`Pose` positions in the world.
    """

    vertices: np.ndarray
    faces: np.ndarray
    name: str = "mesh"

    @classmethod
    def from_file(cls, path, scale: float = 1.0, up_axis: str = "+y",
                  front_axis: str = "+z") -> Mesh:
        """Load a mesh file.

        ``up_axis`` and ``front_axis`` name the file's own axes (the defaults
        match a Blender OBJ export: Y up, the model's front towards +Z).
        """
        path = FilePath(path)
        if path.suffix.lower() == ".obj":
            v, f = _read_obj(path)
        else:
            try:
                import trimesh
            except ImportError as exc:
                raise ImportError("Install trimesh (pip install sonarsim[mesh]) to load "
                                  f"{path.suffix} files; OBJ needs no extra package.") from exc
            loaded = trimesh.load(str(path), force="mesh")
            v, f = np.asarray(loaded.vertices, float), np.asarray(loaded.faces, np.int64)
        return cls.from_arrays(v * scale, f, name=path.stem, up_axis=up_axis,
                               front_axis=front_axis)

    @classmethod
    def from_arrays(cls, vertices, faces, name: str = "mesh", up_axis: str = "+z",
                    front_axis: str = "+x") -> Mesh:
        up, front = _axis_vector(up_axis), _axis_vector(front_axis)
        if abs(up @ front) > 1e-9:
            raise ValueError("up_axis and front_axis must be perpendicular")
        rot = np.vstack([front, np.cross(up, front), up])      # rows: new x, y, z
        v = np.asarray(vertices, dtype=float) @ rot.T
        v -= 0.5 * (v.min(axis=0) + v.max(axis=0))
        mesh = cls(v, np.asarray(faces, dtype=np.int64), name)
        mesh._orient_outward()
        return mesh

    @classmethod
    def sphere(cls, radius: float = 1.0, n_lat: int = 90, n_lon: int = 180) -> Mesh:
        """UV sphere with outward winding, for validation against analytic results."""
        theta = np.linspace(0.0, np.pi, n_lat + 1)[1:-1]
        phi = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)
        tt, pp = np.meshgrid(theta, phi, indexing="ij")
        ring = np.column_stack([np.sin(tt).ravel() * np.cos(pp).ravel(),
                                np.sin(tt).ravel() * np.sin(pp).ravel(), np.cos(tt).ravel()])
        v = radius * np.vstack([[0, 0, 1], ring, [0, 0, -1]])
        bottom = len(v) - 1

        def idx(i, j):
            return 1 + i * n_lon + j % n_lon

        faces = [[0, idx(0, j), idx(0, j + 1)] for j in range(n_lon)]
        for i in range(n_lat - 2):
            for j in range(n_lon):
                faces += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)],
                          [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
        faces += [[bottom, idx(n_lat - 2, j + 1), idx(n_lat - 2, j)] for j in range(n_lon)]
        mesh = cls(v, np.asarray(faces, dtype=np.int64), f"sphere_r{radius:g}")
        mesh._orient_outward()
        return mesh

    def _orient_outward(self) -> None:
        """Flip the winding if the mesh's signed volume is negative."""
        tri = self.vertices[self.faces]
        signed_volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()
        if signed_volume < 0:
            self.faces = self.faces[:, ::-1].copy()

    @property
    def n_facets(self) -> int:
        return len(self.faces)

    @cached_property
    def extent(self) -> np.ndarray:
        return np.ptp(self.vertices, axis=0)

    @cached_property
    def radius(self) -> float:
        """Largest distance from the reference point to a vertex."""
        return float(np.linalg.norm(self.vertices, axis=1).max())

    @cached_property
    def areas(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    def sample_surface(self, n: int, rng=None) -> np.ndarray:
        """Area-weighted random points on the surface (for point-cloud views)."""
        rng = np.random.default_rng(rng)
        tri = self.vertices[self.faces]
        pick = rng.choice(self.n_facets, size=n, p=self.areas / self.areas.sum())
        u, v = rng.random((2, n))
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        t = tri[pick]
        return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])

    def info(self) -> dict:
        return {"name": self.name, "n_facets": self.n_facets,
                "extent_m": self.extent.round(4).tolist(), "radius_m": round(self.radius, 4)}


@dataclass
class Pose:
    """Target position (m, world frame) and orientation (deg).

    Rotation is R = Rz(yaw) Ry(pitch) Rx(roll) applied to the mesh's local
    frame (front +x, up +z). Yaw is measured counterclockwise from east.
    """

    position: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)

    @classmethod
    def facing(cls, position, aspect_deg: float = 0.0, sonar_position=(0.0, 0.0, 0.0),
               pitch: float = 0.0, roll: float = 0.0) -> Pose:
        """Pose with a given aspect: 0 = front towards the sonar, 90 = right side."""
        p = np.asarray(position, dtype=float)
        to_sonar = np.asarray(sonar_position, dtype=float) - p
        yaw = np.degrees(np.arctan2(to_sonar[1], to_sonar[0])) + aspect_deg
        return cls(p, yaw % 360.0, pitch, roll)

    @property
    def rotation(self) -> np.ndarray:
        y, p, r = np.radians([self.yaw, self.pitch, self.roll])
        rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        return rz @ ry @ rx

    def to_dict(self) -> dict:
        return {"position": self.position.round(4).tolist(), "yaw_deg": float(self.yaw),
                "pitch_deg": float(self.pitch), "roll_deg": float(self.roll)}


def _expm1_over(x: np.ndarray) -> np.ndarray:
    """E(x) = (exp(i x) - 1) / x, with E(0) = i."""
    out = np.full(x.shape, 1j, dtype=complex)
    nz = x != 0
    out[nz] = np.expm1(1j * x[nz]) / x[nz]
    return out


def _expm1_over_derivative(x: np.ndarray) -> np.ndarray:
    """dE/dx, using its Taylor series near zero."""
    out = np.empty(x.shape, dtype=complex)
    small = np.abs(x) < 1e-3
    xs = x[small]
    out[small] = -0.5 - 1j * xs / 3 + xs**2 / 8 + 1j * xs**3 / 30
    xl = x[~small]
    out[~small] = (1j * xl * np.exp(1j * xl) - np.expm1(1j * xl)) / xl**2
    return out


def triangle_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """J(a, b) = Integral over the unit right triangle of exp(i(a u + b v)) du dv.

    Closed form J = (E(b) - E(a)) / (a - b) with E(x) = (exp(ix) - 1) / x,
    which equals (a e^{ib} - b e^{ia} + b - a) / (a b (a - b)); J(0, 0) = 1/2.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    near = np.abs(diff) < 1e-4
    out = np.empty(np.broadcast(a, b).shape, dtype=complex)
    far = ~near
    out[far] = (_expm1_over(b[far]) - _expm1_over(a[far])) / diff[far]
    out[near] = -_expm1_over_derivative(0.5 * (a[near] + b[near]))
    return out


class Target:
    """A mesh target with a boundary condition, scattering by Kirchhoff facets."""

    def __init__(self, mesh: Mesh, boundary: str = "rigid"):
        if boundary not in ("rigid", "pressure_release"):
            raise ValueError("boundary must be 'rigid' or 'pressure_release'")
        self.mesh = mesh
        self.boundary = boundary

    @property
    def name(self) -> str:
        return self.mesh.name

    def world_vertices(self, pose: Pose) -> np.ndarray:
        return self.mesh.vertices @ pose.rotation.T + pose.position

    def scattering_amplitude(self, frequencies, incident, scattered, pose: Pose | None = None,
                             sound_speed: float = 1500.0, max_elements: int = 4_000_000):
        """Far-field scattering amplitude f (m) for propagation directions i -> s.

        ``incident`` is the propagation direction of the incoming wave and
        ``scattered`` the direction of the outgoing wave (backscatter: s = -i).
        The phase is referenced to the pose position (the mesh centre).
        """
        f = np.atleast_1d(np.asarray(frequencies, dtype=float))
        i_hat = np.asarray(incident, float) / np.linalg.norm(incident)
        s_hat = np.asarray(scattered, float) / np.linalg.norm(scattered)
        rot = np.eye(3) if pose is None else pose.rotation
        # Work in the mesh frame: rotate the directions instead of the mesh.
        i_loc, s_loc = rot.T @ i_hat, rot.T @ s_hat

        tri = self.mesh.vertices[self.mesh.faces]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        cross = np.cross(e1, e2)
        area2 = np.linalg.norm(cross, axis=1)
        normals = cross / np.where(area2 > 0, area2, 1.0)[:, None]

        lit = (normals @ i_loc < 0) & (normals @ s_loc > 0) & (area2 > 0)
        g = (normals @ (s_loc if self.boundary == "rigid" else i_loc))[lit]
        weight = g * area2[lit]
        q = s_loc - i_loc                 # exp(-i k (i - s) . r) = exp(i k q . r)
        ua, ub, u0 = e1[lit] @ q, e2[lit] @ q, tri[lit, 0] @ q

        k = 2 * np.pi * f / sound_speed
        total = np.zeros(len(f), dtype=complex)
        chunk = max(1, max_elements // len(f))
        for s in range(0, len(weight), chunk):
            sl = slice(s, s + chunk)
            kernel = triangle_kernel(np.outer(k, ua[sl]), np.outer(k, ub[sl]))
            total += (np.exp(1j * np.outer(k, u0[sl])) * kernel) @ weight[sl]
        return 1j * k / (2 * np.pi) * total

    def backscatter(self, frequencies, incident, pose: Pose | None = None,
                    sound_speed: float = 1500.0):
        incident = np.asarray(incident, dtype=float)
        return self.scattering_amplitude(frequencies, incident, -incident, pose, sound_speed)

    def target_strength(self, frequencies, incident, pose: Pose | None = None,
                        sound_speed: float = 1500.0):
        """Monostatic target strength (dB re 1 m)."""
        fbs = self.backscatter(frequencies, incident, pose, sound_speed)
        return 20 * np.log10(np.maximum(np.abs(fbs), 1e-30))

    def info(self) -> dict:
        return {**self.mesh.info(), "boundary": self.boundary}
