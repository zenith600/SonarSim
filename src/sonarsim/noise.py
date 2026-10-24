"""Ambient noise from Wenz-curve spectra.

The spectrum level is the power sum of the turbulence, shipping, wind and
thermal components in the parametric form of Coates (1990), as used by
Stojanovic (2007). Levels are in dB re 1 uPa^2/Hz with frequency in kHz.
"""
from __future__ import annotations

import numpy as np


def wenz_spectrum_level(frequency_hz, wind_speed: float, shipping: float = 0.5):
    """Ambient noise spectrum level (dB re 1 uPa^2/Hz)."""
    f = np.maximum(np.asarray(frequency_hz, dtype=float) / 1e3, 1e-3)
    turbulence = 17.0 - 30.0 * np.log10(f)
    ships = 40.0 + 20.0 * (shipping - 0.5) + 26.0 * np.log10(f) - 60.0 * np.log10(f + 0.03)
    wind = 50.0 + 7.5 * np.sqrt(wind_speed) + 20.0 * np.log10(f) - 40.0 * np.log10(f + 0.4)
    thermal = -15.0 + 20.0 * np.log10(f)
    parts = np.stack([turbulence, ships, wind, thermal])
    return 10.0 * np.log10(np.sum(10.0 ** (parts / 10.0), axis=0))


def ambient_noise_spectrum(rng: np.random.Generator, n_elements: int, n_samples: int,
                           sample_rate: float, center_frequency: float, environment):
    """Spectra (FFT along time) of independent complex-baseband noise per element.

    Scaled so that E|n|^2 = 2 * integral of the Wenz spectrum over the
    simulated band, matching the convention that a unit envelope has an rms
    pressure of 1/sqrt(2).
    """
    f = center_frequency + np.fft.fftfreq(n_samples, 1.0 / sample_rate)
    psd = 10.0 ** (wenz_spectrum_level(f, environment.wind, environment.shipping) / 10.0)
    white = (rng.standard_normal((n_elements, n_samples))
             + 1j * rng.standard_normal((n_elements, n_samples))) / np.sqrt(2.0)
    return np.fft.fft(white, axis=1) * np.sqrt(2.0 * psd * sample_rate)
