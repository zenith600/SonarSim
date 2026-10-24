"""Receiver processing: matched filtering, beamforming and display products."""
from __future__ import annotations

import numpy as np


def matched_filter(data: np.ndarray, waveform, sample_rate: float) -> np.ndarray:
    """Correlate each row with the transmitted replica (FFT, same length).

    Output is normalised by the replica energy, so an echo of envelope
    amplitude A produces a peak of magnitude A at its two-way delay.
    """
    data = np.atleast_2d(data)
    replica = waveform.baseband(sample_rate)
    n = data.shape[-1]
    spectrum = np.fft.fft(replica, n)
    energy = float(np.sum(np.abs(replica) ** 2))
    return np.fft.ifft(np.fft.fft(data, axis=-1) * np.conj(spectrum), axis=-1) / energy


def beamform(data: np.ndarray, sonar, sound_speed: float, angles_deg=None) -> np.ndarray:
    """Delay-and-sum (true time delay) beams for the horizontal line array.

    Implemented per frequency bin, so it is exact for wideband signals.
    Returns an (n_beams, n_samples) array normalised by the shading sum.
    """
    angles = sonar.beam_angles_deg if angles_deg is None else np.asarray(angles_deg, float)
    m, n = data.shape
    spectra = np.fft.fft(data, axis=1)
    f = sonar.center_frequency + np.fft.fftfreq(n, 1.0 / sonar.fs)
    w = sonar.shading_weights
    y0, d = sonar.element_offsets[0], sonar.spacing
    beams = np.empty((len(angles), n), dtype=complex)
    for b, s in enumerate(np.sin(np.radians(angles))):
        k = -2j * np.pi * f * s / sound_speed
        phase = np.exp(k * y0)
        step = np.exp(k * d)
        acc = np.zeros(n, dtype=complex)
        for e in range(m):
            acc += w[e] * spectra[e] * phase
            phase *= step
        beams[b] = acc
    return np.fft.ifft(beams, axis=1) / w.sum()


def to_db(x, floor: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def sector_image(beams: np.ndarray, ranges: np.ndarray, blanking_range: float = 0.0,
                 normalize: str = "range_median", smooth_m: float = 1.0):
    """Beamformed image in dB.

    ``normalize="range_median"`` divides each range sample by the median
    across beams (smoothed over ``smooth_m``), giving contrast above the
    reverberation background; ``"none"`` returns absolute dB re 1 uPa.
    Samples inside the blanking range are set to NaN.
    """
    env = np.abs(beams)
    if normalize == "range_median":
        base = np.median(env, axis=0)
        width = max(1, int(round(smooth_m / max(ranges[1] - ranges[0], 1e-9))))
        base = np.convolve(base, np.ones(width) / width, mode="same")
        image = to_db(env) - to_db(base)[None, :]
    elif normalize == "none":
        image = to_db(env)
    else:
        raise ValueError("normalize must be 'range_median' or 'none'")
    image[:, ranges < blanking_range] = np.nan
    return image


def pool_ranges(values: np.ndarray, ranges: np.ndarray, n_bins: int):
    """Max-pool the last axis into at most ``n_bins`` range bins (for storage).

    NaNs (blanked samples) are ignored unless a whole bin is blanked.
    """
    factor = int(np.ceil(values.shape[-1] / n_bins))
    if factor <= 1:
        return values, ranges
    n = values.shape[-1] // factor * factor
    shaped = values[..., :n].reshape(*values.shape[:-1], -1, factor)
    blank = np.all(np.isnan(shaped), axis=-1)
    pooled = np.max(np.where(np.isnan(shaped), -np.inf, shaped), axis=-1)
    pooled[blank] = np.nan
    return pooled, ranges[:n].reshape(-1, factor).mean(axis=1)


def range_profile(mf_data: np.ndarray, kind: str = "array_average") -> np.ndarray:
    """Matched-filter envelope (dB re 1 uPa) averaged over elements or beams."""
    if kind != "array_average":
        raise ValueError("kind must be 'array_average'")
    return to_db(np.mean(np.abs(np.atleast_2d(mf_data)), axis=0))


def find_peak(profile_db: np.ndarray, ranges: np.ndarray, r_min: float, r_max: float):
    """Range and level of the strongest sample within [r_min, r_max]."""
    sel = np.nonzero((ranges >= r_min) & (ranges <= r_max))[0]
    i = sel[np.argmax(profile_db[sel])]
    return float(ranges[i]), float(profile_db[i])


def local_floor(profile_db: np.ndarray, ranges: np.ndarray, center: float, exclude: float,
                width: float = 10.0) -> float:
    """Median level in windows of ``width`` m either side of a target region."""
    sel = (np.abs(ranges - center) > exclude) & (np.abs(ranges - center) <= exclude + width)
    return float(np.median(profile_db[sel])) if np.any(sel) else float("nan")
