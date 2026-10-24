"""Ocean environment: sound speed, absorption, boundaries and sea state.

Coordinates used throughout SonarSim: x points east, y north and z up. The
sea surface is the plane z = 0 and the seabed is z = -depth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Representative wind speeds (m/s, 10 m height) for each sea state, taken as
# the mid-point of the Beaufort class usually paired with it.
SEA_STATE_WIND_SPEED = {0: 0.0, 1: 1.8, 2: 4.4, 3: 6.7, 4: 9.4, 5: 12.3, 6: 15.5}

# Illustrative Lambert surface backscatter strengths (dB) per sea state. There
# is no widely accepted empirical model at 20-120 kHz, so these are adjustable
# defaults; override them with Environment.surface_mu_db.
SEA_STATE_SURFACE_MU_DB = {0: -60.0, 1: -55.0, 2: -50.0, 3: -45.0, 4: -40.0,
                           5: -36.0, 6: -32.0}


@dataclass(frozen=True)
class Sediment:
    """Fluid-sediment geoacoustic parameters and Lambert backscatter strength."""

    name: str
    density_ratio: float           # rho_sediment / rho_water
    speed_ratio: float             # c_sediment / c_water
    attenuation_db_per_wavelength: float
    lambert_mu_db: float           # bottom backscatter: S_b = mu + 10 log10(sin^2 theta)

    @property
    def loss_parameter(self) -> float:
        """delta in k_b = (omega / c_b)(1 + i delta); 54.58 = 40 pi log10(e)."""
        return self.attenuation_db_per_wavelength / 54.575


# Representative values for the five seabed types AutoPlacer draws from.
# Replace them with site data where available.
SEDIMENTS = {
    "clay": Sediment("clay", 1.5 / 1.025, 0.99, 0.2, -35.0),
    "silt": Sediment("silt", 1.7 / 1.025, 1.05, 1.0, -32.0),
    "sand": Sediment("sand", 1.9 / 1.025, 1.10, 0.8, -27.0),
    "gravel": Sediment("gravel", 2.0 / 1.025, 1.20, 0.6, -22.0),
    "rock": Sediment("rock", 2.5 / 1.025, 2.50, 0.1, -17.0),
}


def mackenzie_sound_speed(temperature, salinity, depth):
    """Sound speed (m/s) from Mackenzie's nine-term equation (1981).

    Valid for 2-30 degC, 25-40 ppt and 0-8000 m.
    """
    t, s, d = (np.asarray(x, dtype=float) for x in (temperature, salinity, depth))
    return (1448.96 + 4.591 * t - 5.304e-2 * t**2 + 2.374e-4 * t**3
            + 1.340 * (s - 35.0) + 1.630e-2 * d + 1.675e-7 * d**2
            - 1.025e-2 * t * (s - 35.0) - 7.139e-13 * t * d**3)


def francois_garrison_absorption(frequency_hz, temperature, salinity, depth, ph=8.0):
    """Seawater absorption (dB/km) from Francois and Garrison (1982).

    Sum of boric-acid, magnesium-sulphate and pure-water relaxation terms.
    ``depth`` is in metres and ``frequency_hz`` may be an array.
    """
    f = np.asarray(frequency_hz, dtype=float) / 1e3          # kHz
    t, s, d = float(temperature), float(salinity), float(depth)
    theta = 273.0 + t
    c = 1412.0 + 3.21 * t + 1.19 * s + 0.0167 * d

    a1 = 8.86 / c * 10.0 ** (0.78 * ph - 5.0)
    f1 = 2.8 * np.sqrt(s / 35.0) * 10.0 ** (4.0 - 1245.0 / theta)

    a2 = 21.44 * s / c * (1.0 + 0.025 * t)
    p2 = 1.0 - 1.37e-4 * d + 6.2e-9 * d**2
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / theta) / (1.0 + 0.0018 * (s - 35.0))

    p3 = 1.0 - 3.83e-5 * d + 4.9e-10 * d**2
    if t <= 20.0:
        a3 = 4.937e-4 - 2.59e-5 * t + 9.11e-7 * t**2 - 1.50e-8 * t**3
    else:
        a3 = 3.964e-4 - 1.146e-5 * t + 1.45e-7 * t**2 - 6.5e-10 * t**3

    return (a1 * f1 * f**2 / (f1**2 + f**2)
            + a2 * p2 * f2 * f**2 / (f2**2 + f**2)
            + a3 * p3 * f**2)


def rms_wave_height(wind_speed):
    """RMS surface elevation (m) of a fully developed Pierson-Moskowitz sea.

    Significant wave height H_s = 0.21 U^2 / g and sigma = H_s / 4.
    """
    return 0.21 * np.asarray(wind_speed, dtype=float) ** 2 / 9.81 / 4.0


def eckart_reflection(frequency_hz, grazing_rad, wave_height_rms, sound_speed):
    """Coherent sea-surface reflection coefficient (Eckart 1953).

    R_s = -exp(-2 k^2 sigma^2 sin^2 theta) for a pressure-release rough surface.
    """
    k = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float) / sound_speed
    return -np.exp(-2.0 * (k * wave_height_rms * np.sin(grazing_rad)) ** 2)


def rayleigh_reflection(grazing_rad, sediment: Sediment):
    """Plane-wave reflection coefficient of a fluid seabed (Rayleigh).

    R = (m sin(theta) - sqrt(n^2 - cos^2 theta)) / (m sin(theta) + sqrt(...)),
    with m = rho_b / rho_w and n = (c_w / c_b)(1 + i delta). Shear is ignored.
    """
    theta = np.asarray(grazing_rad, dtype=float)
    m = sediment.density_ratio
    n = (1.0 + 1j * sediment.loss_parameter) / sediment.speed_ratio
    root = np.sqrt(n**2 - np.cos(theta) ** 2 + 0j)
    root = np.where(root.imag < 0, -root, root)  # decaying wave in the seabed
    return (m * np.sin(theta) - root) / (m * np.sin(theta) + root)


@dataclass
class Environment:
    """Range-independent water column with a flat surface and seabed.

    Temperature varies linearly from ``temperature_surface`` to
    ``temperature_bottom``; salinity is uniform. Sound speed follows
    Mackenzie's equation, but propagation uses straight rays with the
    depth-averaged speed (refraction is neglected).
    """

    depth: float = 40.0
    temperature_surface: float = 18.0
    temperature_bottom: float = 12.0
    salinity: float = 35.0
    ph: float = 8.0
    seabed: str = "sand"
    sea_state: int = 2
    wind_speed: float | None = None        # m/s; default from sea_state
    shipping: float = 0.5                  # Wenz shipping activity, 0-1
    surface_mu_db: float | None = None     # default from sea_state

    def __post_init__(self):
        if self.seabed not in SEDIMENTS:
            raise ValueError(f"seabed must be one of {sorted(SEDIMENTS)}, got {self.seabed!r}")
        if self.sea_state not in SEA_STATE_WIND_SPEED:
            raise ValueError(f"sea_state must be 0-6, got {self.sea_state!r}")

    @property
    def sediment(self) -> Sediment:
        return SEDIMENTS[self.seabed]

    @property
    def wind(self) -> float:
        return SEA_STATE_WIND_SPEED[self.sea_state] if self.wind_speed is None else self.wind_speed

    @property
    def surface_mu(self) -> float:
        if self.surface_mu_db is not None:
            return self.surface_mu_db
        return SEA_STATE_SURFACE_MU_DB[self.sea_state]

    @property
    def wave_height_rms(self) -> float:
        return float(rms_wave_height(self.wind))

    def temperature_at(self, depth_m):
        frac = np.clip(np.asarray(depth_m, dtype=float) / self.depth, 0.0, 1.0)
        return self.temperature_surface + (self.temperature_bottom - self.temperature_surface) * frac

    def sound_speed_profile(self, n: int = 101):
        """Return (depths, sound speeds) sampled from the surface to the seabed."""
        depths = np.linspace(0.0, self.depth, n)
        return depths, mackenzie_sound_speed(self.temperature_at(depths), self.salinity, depths)

    @property
    def mean_sound_speed(self) -> float:
        """Depth-averaged sound speed, used for travel times and range axes."""
        depths, c = self.sound_speed_profile()
        return float(np.sum(0.5 * (c[1:] + c[:-1]) * np.diff(depths)) / self.depth)

    def absorption_db_per_km(self, frequency_hz):
        """Francois-Garrison absorption evaluated at mid-water-column conditions."""
        mid = self.depth / 2.0
        return francois_garrison_absorption(frequency_hz, float(self.temperature_at(mid)),
                                            self.salinity, mid, self.ph)

    def surface_reflection(self, frequency_hz, grazing_rad):
        return eckart_reflection(frequency_hz, grazing_rad, self.wave_height_rms,
                                 self.mean_sound_speed)

    def bottom_reflection(self, grazing_rad):
        return rayleigh_reflection(grazing_rad, self.sediment)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(wind_speed=self.wind, surface_mu_db=self.surface_mu,
                 mean_sound_speed=self.mean_sound_speed,
                 sediment=asdict(self.sediment))
        return d
