"""
Noise Module
============
Underwater acoustic noise generation and injection.

Supported noise models
----------------------
'white'    : White Gaussian noise – flat power spectrum.
'thermal'  : Johnson-Nyquist thermal noise – amplitude rises with √frequency
             above ~1 kHz (electronic/thermal floor of a hydrophone).
'ambient'  : Wenz-model ambient ocean noise – dominant at low frequencies,
             decays roughly as 1/f above the deep-noise minimum (~300 Hz).
'shipping' : Low-frequency shipping/industrial noise – steep negative slope
             below 200 Hz.

Usage (signal)
--------------
    from acousticsim.noise import NoiseModel
    nm = NoiseModel('ambient', snr_db=12.0, seed=42)
    t, sig = hydrophone.get_multipath_signal(noise_model=nm)

Usage (arrivals injection)
--------------------------
    nm.inject_arrivals(hydrophone, n_noise_arrivals=8)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate pink (1/f) noise via spectral shaping of white noise."""
    white = rng.standard_normal(n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1.0  # avoid /0
    fft /= np.sqrt(freqs)
    signal = np.fft.irfft(fft, n=n_samples)
    return signal / (np.std(signal) + 1e-20)


def _shaped_noise(
    n_samples: int,
    sample_rate: float,
    noise_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return unit-std noise shaped for the requested model."""

    # FFT shaping requires at least a few samples to be meaningful.
    if n_samples <= 4:
        sig = rng.standard_normal(max(n_samples, 1))
        std = float(np.std(sig))
        return sig / (std if std > 0 else 1.0)

    if noise_type == 'white':
        sig = rng.standard_normal(n_samples)

    elif noise_type == 'thermal':
        # Rise above ~1 kHz: shape white noise so |H(f)| ∝ max(f/1000, 1)^0.5
        white = rng.standard_normal(n_samples)
        f = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        f[0] = 1.0
        gain = np.sqrt(np.maximum(f / 1000.0, 1.0))
        fft = np.fft.rfft(white) * gain
        sig = np.fft.irfft(fft, n=n_samples)

    elif noise_type == 'ambient':
        # Wenz model: dominant below 1 kHz, slope ≈ −6 dB/octave above.
        # Approximate: |H(f)| ∝ 1 / max(f/300, 1)
        white = rng.standard_normal(n_samples)
        f = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        f[0] = 1.0
        gain = 1.0 / np.maximum(f / 300.0, 1.0)
        fft = np.fft.rfft(white) * gain
        sig = np.fft.irfft(fft, n=n_samples)

    elif noise_type == 'shipping':
        # Heavy below 200 Hz, steep roll-off above.
        white = rng.standard_normal(n_samples)
        f = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        f[0] = 1.0
        gain = 1.0 / (1.0 + (f / 200.0) ** 2)
        fft = np.fft.rfft(white) * gain
        sig = np.fft.irfft(fft, n=n_samples)

    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            "Choose: 'white', 'thermal', 'ambient', 'shipping'."
        )

    std = float(np.std(sig))
    return sig / (std if std > 0 else 1.0)


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------

def add_noise_to_signal(
    signal: np.ndarray,
    noise_model: 'NoiseModel',
    sample_rate: float = 44100.0,
) -> np.ndarray:
    """
    Add noise from *noise_model* to an existing signal array.

    Args:
        signal:       1-D float array (the clean signal).
        noise_model:  A :class:`NoiseModel` instance.
        sample_rate:  Sample rate in Hz (used for spectral shaping).

    Returns:
        Noisy signal array, same shape as *signal*.
    """
    noise = noise_model.generate(len(signal), sample_rate=sample_rate)
    return signal + noise


# ---------------------------------------------------------------------------
# NoiseModel
# ---------------------------------------------------------------------------

class NoiseModel:
    """
    Configurable underwater acoustic noise generator.

    Parameters
    ----------
    noise_type : str
        One of ``'white'``, ``'thermal'``, ``'ambient'``, ``'shipping'``.
    snr_db : float
        Target signal-to-noise ratio in dB when injecting noise into a
        signal with ``add_noise_to_signal`` or
        ``Hydrophone.get_multipath_signal(noise_model=...)``.
        Set to ``None`` to use a fixed absolute ``noise_amplitude``.
    noise_amplitude : float or None
        Absolute noise RMS amplitude used when *snr_db* is ``None``.
        Default: ``1e-6``.
    seed : int or None
        Seed for the internal random number generator (reproducibility).
    """

    NOISE_TYPES = ('white', 'thermal', 'ambient', 'shipping')

    def __init__(
        self,
        noise_type: str = 'white',
        snr_db: float = 20.0,
        noise_amplitude: float = 1e-6,
        seed: int | None = None,
    ):
        if noise_type not in self.NOISE_TYPES:
            raise ValueError(
                f"noise_type must be one of {self.NOISE_TYPES}, got '{noise_type}'."
            )
        self.noise_type = noise_type
        self.snr_db = snr_db
        self.noise_amplitude = noise_amplitude
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, n_samples: int, sample_rate: float = 44100.0,
                 reference_signal: np.ndarray | None = None) -> np.ndarray:
        """
        Generate a noise array of length *n_samples*.

        If *reference_signal* is provided **and** ``snr_db`` is not ``None``,
        the noise amplitude is scaled so that SNR = ``snr_db`` dB relative to
        the RMS of *reference_signal*.

        Args:
            n_samples:        Number of samples to generate.
            sample_rate:      Sample rate in Hz (for spectral shaping).
            reference_signal: Optional clean signal for SNR computation.

        Returns:
            1-D float64 noise array.
        """
        raw = _shaped_noise(n_samples, sample_rate, self.noise_type, self._rng)

        if self.snr_db is not None and reference_signal is not None:
            sig_rms = np.sqrt(np.mean(reference_signal ** 2)) + 1e-30
            desired_noise_rms = sig_rms / (10 ** (self.snr_db / 20.0))
            return raw * desired_noise_rms
        else:
            return raw * self.noise_amplitude

    # ------------------------------------------------------------------
    # Arrival injection
    # ------------------------------------------------------------------

    def inject_arrivals(
        self,
        hydrophone,
        n_noise_arrivals: int = 5,
        frequency_khz: float = 25.0,
        time_range: tuple[float, float] = (0.001, 0.5),
    ) -> None:
        """
        Inject randomised synthetic noise arrivals into a hydrophone.

        Noise arrivals mimic spurious detections (biological noise,
        reverberation tails, electronic glitches) and will appear in the
        feature vector produced by :func:`~acousticsim.feature_extractor.extract_features`.

        Args:
            hydrophone:       :class:`~acousticsim.hydrophone.Hydrophone` instance.
            n_noise_arrivals: Number of fake arrivals to inject.
            frequency_khz:    Frequency label attached to each noise arrival.
            time_range:       ``(t_min, t_max)`` in seconds for random arrival times.
        """
        t_min, t_max = time_range
        for _ in range(n_noise_arrivals):
            arr_time = float(self._rng.uniform(t_min, t_max))
            amplitude = float(
                np.abs(self._rng.normal(loc=0.0, scale=self.noise_amplitude))
            )
            phase = float(self._rng.uniform(0.0, 2 * np.pi))

            hydrophone.arrivals.append({
                'time': arr_time,
                'amplitude': amplitude,
                'phase': phase,
                'path_length': arr_time * 1500.0,
                'bounces': 0,
                'frequency_khz': frequency_khz,
                'wave_type': 'noise',
                'frequency_range': (frequency_khz * 0.9, frequency_khz * 1.1),
                'position': hydrophone.position.copy(),
                'perpendicular_distance': 0.0,
                '_is_noise': True,
            })

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NoiseModel(type='{self.noise_type}', snr_db={self.snr_db}, "
            f"noise_amplitude={self.noise_amplitude})"
        )
