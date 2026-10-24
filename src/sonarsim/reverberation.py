"""Boundary reverberation from the sea surface and seabed.

Each boundary inside the transmit sector is divided into patches, one per
time sample (slant-range step c / 2 fs) and azimuth step. A patch at slant
range r and grazing angle theta returns a delayed copy of the pulse with a
complex Gaussian amplitude of mean-square

    2 * 10^((SL - 2 TL + S) / 10) * dA * b_T^2 * b_R^2,

where TL = 20 log10 r + alpha r / 1000 (direct path only), dA is the patch
area and S = mu + 10 log10(sin^2 theta) is Lambert's law with mu set by the
seabed type (bottom) or sea state (surface). Volume reverberation and
reverberation via multipath are not modelled.
"""
from __future__ import annotations

import numpy as np


def reverberation_spectrum(rng: np.random.Generator, sonar, environment, n_samples: int,
                           pulse_spectrum: np.ndarray, azimuth_step_deg: float | None = None,
                           oversample: int = 4) -> np.ndarray:
    """Element spectra (M x N) of boundary reverberation, including the pulse."""
    c = environment.mean_sound_speed
    fs, fc = sonar.fs, sonar.center_frequency
    m, spacing = sonar.n_elements, sonar.spacing
    r = c * np.arange(n_samples) / fs / 2.0
    alpha = float(environment.absorption_db_per_km(fc))

    if azimuth_step_deg is None:
        azimuth_step_deg = np.degrees(c / fc / (m * spacing)) / 3.0
    half = sonar.sector_deg + 15.0
    az = np.radians(np.arange(-half, half + 1e-9, azimuth_step_deg))
    d_az = np.radians(azimuth_step_deg)

    # Group patches by the array's spatial frequency s = cos(elev) sin(az):
    # a line array cannot tell apart arrivals that share s.
    s_step = c / sonar.waveform.f_high / (m * spacing) / oversample
    s_max = np.sin(np.radians(half))
    n_bins = int(np.ceil(2 * s_max / s_step)) + 1
    s_grid = -s_max + s_step * np.arange(n_bins)
    field = np.zeros(n_bins * n_samples, dtype=complex)

    boundaries = (
        (sonar.depth, +1.0, environment.surface_mu),
        (environment.depth - sonar.depth, -1.0, environment.sediment.lambert_mu_db),
    )
    for height, sign, mu in boundaries:
        idx = np.nonzero(r > height * 1.001)[0]
        if idx.size == 0:
            continue
        rv = r[idx]
        sin_g = height / rv
        cos_g = np.sqrt(1.0 - sin_g**2)
        elevation = sign * np.arcsin(sin_g)
        d_rho = (c / (2.0 * fs)) / np.maximum(cos_g, 0.05)
        loss_db = 2 * (20 * np.log10(rv) + alpha * rv / 1000.0)
        strength_db = mu + 10 * np.log10(sin_g**2)
        base = (10.0 ** ((sonar.source_level_db - loss_db + strength_db) / 10.0)
                * rv * cos_g * d_az * d_rho)

        dirs = np.stack(np.broadcast_arrays(
            (np.cos(elevation)[:, None] * np.cos(az)[None, :]),
            (np.cos(elevation)[:, None] * np.sin(az)[None, :]),
            np.sin(elevation)[:, None]), axis=-1)
        gain = (sonar.transmit_pattern(dirs) * sonar.receive_pattern(dirs)) ** 2
        intensity = 2.0 * base[:, None] * gain
        amp = np.sqrt(intensity) * (rng.standard_normal(intensity.shape)
                                    + 1j * rng.standard_normal(intensity.shape)) / np.sqrt(2.0)

        s = dirs[..., 1]
        bins = np.clip(np.rint((s + s_max) / s_step).astype(np.int64), 0, n_bins - 1)
        flat = (bins * n_samples + idx[:, None]).ravel()
        field += (np.bincount(flat, amp.real.ravel(), n_bins * n_samples)
                  + 1j * np.bincount(flat, amp.imag.ravel(), n_bins * n_samples))

    field = field.reshape(n_bins, n_samples)
    active = np.nonzero(np.any(field != 0, axis=1))[0]
    spectra = np.fft.fft(field[active], axis=1)

    f = fc + np.fft.fftfreq(n_samples, 1.0 / fs)
    y0 = sonar.element_offsets[0]
    out = np.zeros((m, n_samples), dtype=complex)
    for spec, s in zip(spectra, s_grid[active]):
        k = 2j * np.pi * f * s / c
        phase = np.exp(k * y0)
        step = np.exp(k * spacing)
        for e in range(m):
            out[e] += spec * phase
            phase *= step
    return out * pulse_spectrum
