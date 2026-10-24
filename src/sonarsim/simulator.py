"""Ping simulation: element signals from target echoes, reverberation and noise.

Signals are complex baseband pressures (uPa) on each hydrophone, sampled at
``sonar.fs`` from the start of transmission. The target echo is synthesised
in the frequency domain for every pair of outgoing and returning eigenpaths:

    E_pq(f) = sqrt(2) 10^(SL/20) X(f) A_p(f) A_q(f) b_T b_R f_pq(f) exp(-i 2 pi f (tau_p + tau_q)),

where X is the pulse spectrum, A the path amplitudes, b_T and b_R the beam
patterns and f_pq the Kirchhoff scattering amplitude for incidence along p
and scattering back along q. Element m adds the phase of the arrival
direction of path q across the line array.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import next_fast_len

from sonarsim.environment import Environment
from sonarsim.noise import ambient_noise_spectrum
from sonarsim.propagation import eigenpaths
from sonarsim.reverberation import reverberation_spectrum
from sonarsim.sonar import SonarConfig
from sonarsim.target import Pose, Target

COMPONENTS = ("target", "reverberation", "noise")


@dataclass
class TargetArrival:
    """One out-and-back multipath arrival of the target echo."""

    outgoing: str
    returning: str
    range_m: float          # c * two-way delay / 2
    delay_s: float
    level_db: float         # echo level at fc, dB re 1 uPa (rms), before beam patterns
    bearing_deg: float      # direction of arrival at the array
    elevation_deg: float

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Ping:
    sonar: SonarConfig
    environment: Environment
    sample_rate: float
    sound_speed: float
    data: np.ndarray                         # (n_elements, n_samples) complex uPa
    target: Target | None = None
    pose: Pose | None = None
    arrivals: list[TargetArrival] = field(default_factory=list)
    components: dict[str, np.ndarray] | None = None
    seed: int | None = None

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def ranges(self) -> np.ndarray:
        """Slant range (m) of each sample for a two-way travel time."""
        return self.sound_speed * np.arange(self.n_samples) / self.sample_rate / 2.0


INTERP_TAPS = 20
PAIR_FLOOR_DB = 100.0


def _band_limited_resample(coarse_f, values, fine_f, support):
    """Interpolate the spectrum of a time-limited response onto new frequencies.

    The response must lie within |t| <= support / 2 and ``coarse_f`` must be
    uniform with spacing df <= 1 / (2 support). Sampling at df periodises the
    response with period W = 1 / df; a smooth time window that is 1 over the
    support and 0 over the periodic images recovers it exactly. In frequency
    that window is a sinc times a Gaussian, so each new frequency needs only
    the INTERP_TAPS nearest samples on either side (relative error ~1e-6).
    """
    df = coarse_f[1] - coarse_f[0]
    period = 1.0 / df
    if period < 2.0 * support:
        raise ValueError("coarse grid too sparse for the response support")
    sigma = (period - support) / 10.0               # window edges 5 sigma clear
    pos = (np.asarray(fine_f) - coarse_f[0]) / df
    base = np.floor(pos).astype(np.int64)
    taps = base[:, None] + np.arange(-INTERP_TAPS + 1, INTERP_TAPS + 1)[None, :]
    if taps.min() < 0 or taps.max() >= len(coarse_f):
        raise ValueError("coarse grid needs INTERP_TAPS samples beyond the fine band")
    x = pos[:, None] - taps                          # (f - f_j) / df
    kernel = np.sinc(x) * np.exp(-2.0 * (np.pi * sigma * df * x) ** 2)
    return np.sum(values[taps] * kernel, axis=1)


def _target_spectrum(sonar, env, target, pose, f_pass, band, pulse_spectrum, c):
    """Element spectra of the target echo and the list of arrivals."""
    paths = eigenpaths(sonar.position, pose.position, env.depth, sonar.max_bounces)
    fb = f_pass[band]

    # Coarse frequency grid for the Kirchhoff sum. Relative to the mesh
    # centre the response lasts at most 4 R / c, so a spacing of half its
    # inverse lets it be interpolated exactly onto the fine grid.
    support = 4.0 * target.mesh.radius / c
    df = 1.0 / (2.0 * support + 4.0 / sonar.bandwidth)
    margin = (INTERP_TAPS + 1) * df
    n_coarse = int(np.ceil((fb.max() - fb.min() + 2 * margin) / df)) + 1
    coarse_f = fb.min() - margin + df * np.arange(n_coarse)

    amp = {p.name: p.amplitude(fb, env) for p in paths}
    tx = {p.name: float(sonar.transmit_pattern(p.departure)) for p in paths}
    rx = {p.name: float(sonar.receive_pattern(p.departure)) for p in paths}
    y = sonar.element_offsets
    scale = np.sqrt(2.0) * 10.0 ** (sonar.source_level_db / 20.0)
    ic = np.argmin(np.abs(fb - sonar.center_frequency))

    # Pairs whose path losses put them more than PAIR_FLOOR_DB below the
    # strongest pair cannot affect the signal; they are listed but not summed.
    strength = {(p.name, q.name): tx[p.name] * rx[q.name]
                * np.max(np.abs(amp[p.name] * amp[q.name])) for p in paths for q in paths}
    floor = max(strength.values()) * 10.0 ** (-PAIR_FLOOR_DB / 20.0)

    spectrum = np.zeros((sonar.n_elements, len(fb)), dtype=complex)
    arrivals = []
    for p in paths:
        for q in paths:
            delay = (p.length + q.length) / c
            u = q.departure                     # direction of arrival at the array
            if strength[(p.name, q.name)] >= floor:
                f_coarse = target.scattering_amplitude(coarse_f, -p.arrival, q.arrival, pose, c)
                f_pq = _band_limited_resample(coarse_f, f_coarse, fb, support)
                echo = scale * amp[p.name] * amp[q.name] * f_pq * np.exp(-2j * np.pi * fb * delay)
                steer = np.exp(2j * np.pi * np.outer(y, fb) * u[1] / c)
                spectrum += tx[p.name] * rx[q.name] * echo * steer
                echo_fc = echo[ic]
            else:
                f_fc = target.scattering_amplitude([fb[ic]], -p.arrival, q.arrival, pose, c)[0]
                echo_fc = scale * amp[p.name][ic] * amp[q.name][ic] * f_fc
            arrivals.append(TargetArrival(
                outgoing=p.name, returning=q.name, range_m=c * delay / 2.0, delay_s=delay,
                level_db=float(20 * np.log10(abs(echo_fc) / np.sqrt(2.0) + 1e-30)),
                bearing_deg=float(np.degrees(np.arctan2(u[1], u[0]))),
                elevation_deg=float(np.degrees(np.arcsin(u[2])))))

    full = np.zeros((sonar.n_elements, len(f_pass)), dtype=complex)
    full[:, band] = spectrum * pulse_spectrum[band]
    return full, sorted(arrivals, key=lambda a: a.delay_s)


def simulate_ping(sonar: SonarConfig, environment: Environment, target: Target | None = None,
                  pose: Pose | None = None, *, seed: int | None = None,
                  include=COMPONENTS, keep_components: bool = False) -> Ping:
    """Simulate one ping and return the element signals with ground truth."""
    unknown = set(include) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown components {sorted(unknown)}; choose from {COMPONENTS}")
    if "target" in include and (target is None or pose is None):
        raise ValueError("a target and pose are needed to simulate the target echo")

    rng = np.random.default_rng(seed)
    c = environment.mean_sound_speed
    fs, fc = sonar.fs, sonar.center_frequency
    pulse = sonar.waveform.baseband(fs)
    n = next_fast_len(int(np.ceil((2.0 * sonar.max_range / c + sonar.waveform.duration) * fs)) + 1)

    f_bb = np.fft.fftfreq(n, 1.0 / fs)
    f_pass = fc + f_bb
    pulse_spectrum = np.fft.fft(pulse, n)
    guard = max(4.0 / sonar.waveform.duration, 0.05 * sonar.bandwidth)
    band = np.abs(f_bb) <= sonar.bandwidth / 2.0 + guard

    parts: dict[str, np.ndarray] = {}
    arrivals: list[TargetArrival] = []
    if "target" in include:
        parts["target"], arrivals = _target_spectrum(sonar, environment, target, pose, f_pass,
                                                     band, pulse_spectrum, c)
    if "reverberation" in include:
        parts["reverberation"] = reverberation_spectrum(rng, sonar, environment, n,
                                                        pulse_spectrum)
    if "noise" in include:
        parts["noise"] = ambient_noise_spectrum(rng, sonar.n_elements, n, fs, fc, environment)

    total = sum(parts.values()) if parts else np.zeros((sonar.n_elements, n), dtype=complex)
    data = np.fft.ifft(total, axis=1)
    components = ({k: np.fft.ifft(v, axis=1) for k, v in parts.items()}
                  if keep_components else None)
    return Ping(sonar=sonar, environment=environment, sample_rate=fs, sound_speed=c,
                data=data, target=target, pose=pose, arrivals=arrivals,
                components=components, seed=seed)
