"""Active sonar configuration: waveform, hydrophone array and beam geometry.

The sonar sits at (0, 0, -depth) looking east (+x). Its receiving array is a
horizontal uniform line array along y, so bearing 0 is broadside and positive
bearings are towards north. The transmitter covers the beamformed sector
horizontally and has a Gaussian vertical beam tilted down by ``tilt_deg``;
the same vertical pattern is applied on receive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from sonarsim.waveforms import Waveform

NOMINAL_SOUND_SPEED = 1500.0


@dataclass
class SonarConfig:
    name: str
    waveform: Waveform
    source_level_db: float               # dB re 1 uPa rms at 1 m
    n_elements: int
    element_spacing: float | None = None  # m; default half a wavelength at f_high
    sector_deg: float = 30.0             # half-width of the beamformed sector
    n_beams: int = 101
    sample_rate: float | None = None     # complex baseband Hz; default 2.5 x bandwidth
    depth: float = 7.0                   # m below the surface
    max_range: float = 200.0             # m
    blanking_range: float = 30.0         # m
    vertical_beamwidth_deg: float = 60.0  # -3 dB full width
    tilt_deg: float = 0.0                # depression of the vertical beam axis
    max_bounces: int = 1                 # boundary reflections per one-way target path
    shading: str = "hann"

    @property
    def center_frequency(self) -> float:
        return self.waveform.center_frequency

    @property
    def bandwidth(self) -> float:
        return self.waveform.bandwidth

    @property
    def spacing(self) -> float:
        if self.element_spacing is not None:
            return self.element_spacing
        return NOMINAL_SOUND_SPEED / self.waveform.f_high / 2.0

    @property
    def fs(self) -> float:
        if self.sample_rate is not None:
            return self.sample_rate
        return 2.5 * max(self.bandwidth, 1.0 / self.waveform.duration)

    @property
    def position(self) -> np.ndarray:
        return np.array([0.0, 0.0, -self.depth])

    @property
    def element_offsets(self) -> np.ndarray:
        """Element y-offsets (m) about the array centre."""
        return (np.arange(self.n_elements) - (self.n_elements - 1) / 2) * self.spacing

    @property
    def beam_angles_deg(self) -> np.ndarray:
        return np.linspace(-self.sector_deg, self.sector_deg, self.n_beams)

    @property
    def shading_weights(self) -> np.ndarray:
        if self.shading == "hann":
            return np.hanning(self.n_elements + 2)[1:-1]
        if self.shading == "uniform":
            return np.ones(self.n_elements)
        raise ValueError("shading must be 'hann' or 'uniform'")

    def range_resolution(self, sound_speed: float = NOMINAL_SOUND_SPEED) -> float:
        return sound_speed / (2.0 * self.bandwidth)

    def _vertical_pattern(self, elevation_rad):
        half = np.radians(self.vertical_beamwidth_deg) / 2.0
        x = (np.asarray(elevation_rad) + np.radians(self.tilt_deg)) / half
        return np.exp(-0.5 * np.log(2.0) * x**2)

    def transmit_pattern(self, direction) -> np.ndarray:
        """Transmit amplitude pattern for unit direction vector(s) (..., 3)."""
        d = np.asarray(direction, dtype=float)
        bearing = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
        elevation = np.arcsin(np.clip(d[..., 2], -1.0, 1.0))
        excess = np.clip(np.abs(bearing) - (self.sector_deg + 5.0), 0.0, 10.0)
        horizontal = 0.5 * (1.0 + np.cos(np.pi * excess / 10.0))
        horizontal = np.where(d[..., 0] > 0, horizontal, 0.0)
        return horizontal * self._vertical_pattern(elevation)

    def receive_pattern(self, direction) -> np.ndarray:
        """Receive amplitude pattern of each element (vertical aperture only)."""
        d = np.asarray(direction, dtype=float)
        elevation = np.arcsin(np.clip(d[..., 2], -1.0, 1.0))
        return np.where(d[..., 0] > 0, self._vertical_pattern(elevation), 0.0)

    def with_overrides(self, **changes) -> SonarConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(element_spacing=self.spacing, sample_rate=self.fs,
                 center_frequency=self.center_frequency, bandwidth=self.bandwidth)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SonarConfig:
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        fields["waveform"] = Waveform(**d["waveform"])
        return cls(**fields)


def A_HF_imaging() -> SonarConfig:
    """Configuration A: high-frequency imaging (80-120 kHz LFM, 48 elements)."""
    return SonarConfig(
        name="A_HF_imaging", waveform=Waveform.lfm(80e3, 120e3, 5e-3),
        source_level_db=210.0, n_elements=48, sector_deg=25.0, n_beams=101,
        depth=7.0, max_range=180.0, blanking_range=30.0)


def B_LF_search() -> SonarConfig:
    """Configuration B: low-frequency search (20-40 kHz LFM, 24 elements)."""
    return SonarConfig(
        name="B_LF_search", waveform=Waveform.lfm(20e3, 40e3, 10e-3),
        source_level_db=215.0, n_elements=24, sector_deg=40.0, n_beams=81,
        depth=7.0, max_range=410.0, blanking_range=30.0)


PRESETS = {"A_HF_imaging": A_HF_imaging, "B_LF_search": B_LF_search}


def preset(name: str) -> SonarConfig:
    try:
        return PRESETS[name]()
    except KeyError:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}") from None
