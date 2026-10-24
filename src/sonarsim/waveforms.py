"""Transmit waveforms as complex baseband signals.

A passband pressure p(t) is represented by its complex envelope x(t) about
the centre frequency fc: p(t) = Re{x(t) exp(i 2 pi fc t)}. A unit-amplitude
envelope therefore has an rms pressure of 1/sqrt(2).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal.windows import tukey


@dataclass(frozen=True)
class Waveform:
    kind: str            # "lfm" or "cw"
    f_low: float         # Hz
    f_high: float        # Hz
    duration: float      # s
    taper: float = 0.1   # Tukey window fraction

    def __post_init__(self):
        if self.kind not in ("lfm", "cw"):
            raise ValueError("kind must be 'lfm' or 'cw'")
        if self.f_high < self.f_low or self.duration <= 0:
            raise ValueError("need f_high >= f_low and duration > 0")

    @classmethod
    def lfm(cls, f_low: float, f_high: float, duration: float, taper: float = 0.1) -> Waveform:
        return cls("lfm", f_low, f_high, duration, taper)

    @classmethod
    def cw(cls, frequency: float, duration: float, taper: float = 0.1) -> Waveform:
        return cls("cw", frequency, frequency, duration, taper)

    @property
    def center_frequency(self) -> float:
        return 0.5 * (self.f_low + self.f_high)

    @property
    def bandwidth(self) -> float:
        """Swept bandwidth for LFM; about 1/T for CW."""
        return self.f_high - self.f_low if self.kind == "lfm" else 1.0 / self.duration

    def baseband(self, sample_rate: float) -> np.ndarray:
        """Complex envelope sampled at ``sample_rate`` (starts at t = 0)."""
        n = max(int(round(self.duration * sample_rate)), 2)
        t = np.arange(n) / sample_rate
        window = tukey(n, self.taper)
        if self.kind == "cw":
            return window.astype(complex)
        sweep = self.f_high - self.f_low
        phase = 2 * np.pi * (-0.5 * sweep * t + 0.5 * sweep / self.duration * t**2)
        return window * np.exp(1j * phase)

    def to_dict(self) -> dict:
        return asdict(self)
