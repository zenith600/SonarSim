"""AutoPlacer: randomised, physically valid target placements (domain randomisation).

Each sample draws a slant range, bearing, target depth and orientation, plus
a seabed type and sea state. A placement is rejected and redrawn when the
target's rotated bounding box comes closer than ``clearance`` to the surface
or seabed, or when the geometry is impossible (slant range shorter than the
depth offset). Every sample carries its own seed so it can be regenerated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from sonarsim.environment import SEDIMENTS, Environment
from sonarsim.sonar import SonarConfig
from sonarsim.target import Pose, Target


@dataclass
class PlacementBounds:
    slant_range: tuple[float, float] | None = None   # default: blanking + 10 to max range - 10
    bearing_deg: tuple[float, float] | None = None   # default: inside the beamformed sector
    clearance: float = 1.0                           # m from surface and seabed
    yaw_deg: tuple[float, float] = (0.0, 360.0)
    pitch_deg: tuple[float, float] = (0.0, 0.0)
    roll_deg: tuple[float, float] = (0.0, 0.0)
    seabeds: tuple[str, ...] = tuple(SEDIMENTS)
    sea_states: tuple[int, ...] = (1, 2, 3, 4)

    def resolved(self, sonar: SonarConfig) -> PlacementBounds:
        rng = self.slant_range or (sonar.blanking_range + 10.0, sonar.max_range - 10.0)
        half = sonar.sector_deg - 2.0
        brg = self.bearing_deg or (-half, half)
        return replace(self, slant_range=tuple(rng), bearing_deg=tuple(brg))


@dataclass
class Scenario:
    index: int
    seed: int                       # seed for the ping's random components
    environment: Environment
    pose: Pose
    slant_range: float              # direct-path range sonar -> target centre (m)
    horizontal_range: float
    bearing_deg: float
    depth: float                    # target centre depth below surface (m)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"index": self.index, "seed": self.seed, "slant_range_m": round(self.slant_range, 4),
                "horizontal_range_m": round(self.horizontal_range, 4),
                "bearing_deg": round(self.bearing_deg, 4), "depth_m": round(self.depth, 4),
                **self.pose.to_dict()}


class AutoPlacer:
    def __init__(self, sonar: SonarConfig, environment: Environment, target: Target,
                 bounds: PlacementBounds | None = None, seed: int | None = None,
                 max_tries: int = 1000):
        self.sonar = sonar
        self.environment = environment
        self.target = target
        self.bounds = (bounds or PlacementBounds()).resolved(sonar)
        self.max_tries = max_tries
        self._seeds = np.random.SeedSequence(seed)
        self._count = 0

    def _half_height(self, pose: Pose) -> float:
        z = (self.target.mesh.vertices @ pose.rotation.T)[:, 2]
        return float(max(z.max(), -z.min()))

    def sample(self) -> Scenario:
        rng = np.random.default_rng(self._seeds.spawn(1)[0])
        b = self.bounds
        for _ in range(self.max_tries):
            env = replace(self.environment, seabed=str(rng.choice(b.seabeds)),
                          sea_state=int(rng.choice(b.sea_states)))
            yaw, pitch, roll = (rng.uniform(*r) for r in (b.yaw_deg, b.pitch_deg, b.roll_deg))
            pose = Pose(np.zeros(3), yaw % 360.0, pitch, roll)
            half = self._half_height(pose)
            d_min, d_max = b.clearance + half, env.depth - b.clearance - half
            if d_min >= d_max:
                raise ValueError("target is too tall for the water depth and clearance")
            depth = rng.uniform(d_min, d_max)
            slant = rng.uniform(*b.slant_range)
            dz = depth - self.sonar.depth
            if slant <= abs(dz) + 0.5:
                continue
            horizontal = float(np.sqrt(slant**2 - dz**2))
            bearing = rng.uniform(*b.bearing_deg)
            phi = np.radians(bearing)
            pose.position = np.array([horizontal * np.cos(phi), horizontal * np.sin(phi), -depth])
            scenario = Scenario(self._count, int(rng.integers(2**31 - 1)), env, pose,
                                float(slant), horizontal, float(bearing), float(depth))
            self._count += 1
            return scenario
        raise RuntimeError("AutoPlacer could not find a valid placement; widen the bounds")

    def samples(self, n: int) -> list[Scenario]:
        return [self.sample() for _ in range(n)]
