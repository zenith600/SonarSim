import numpy as np

from sonarsim.noise import ambient_noise_spectrum, wenz_spectrum_level
from sonarsim.environment import Environment
from sonarsim.processing import beamform, matched_filter, pool_ranges
from sonarsim.sonar import A_HF_imaging
from sonarsim.waveforms import Waveform


def test_lfm_energy_is_in_band():
    w = Waveform.lfm(80e3, 120e3, 5e-3)
    fs = 100e3
    x = w.baseband(fs)
    assert len(x) == 500 and np.max(np.abs(x)) <= 1.0 + 1e-12
    spec = np.abs(np.fft.fft(x, 8192)) ** 2
    f = np.fft.fftfreq(8192, 1 / fs)
    in_band = np.abs(f) <= w.bandwidth / 2 + 2e3
    assert spec[in_band].sum() / spec.sum() > 0.99


def test_matched_filter_peak_at_delay_with_unit_gain():
    w = Waveform.lfm(80e3, 120e3, 5e-3)
    fs, n, delay = 100e3, 4096, 1234
    sig = np.zeros(n, complex)
    x = w.baseband(fs)
    sig[delay:delay + len(x)] = 3.0 * x
    out = matched_filter(sig, w, fs)[0]
    assert np.argmax(np.abs(out)) == delay
    np.testing.assert_allclose(np.abs(out[delay]), 3.0, rtol=1e-12)


def test_beamformer_steers_to_plane_wave_bearing():
    sonar = A_HF_imaging()
    c, fs, n = 1500.0, sonar.fs, 2048
    bearing = 10.0
    x = np.zeros(n, complex)
    x[100:600] = sonar.waveform.baseband(fs)
    spec = np.fft.fft(x)
    f = sonar.center_frequency + np.fft.fftfreq(n, 1 / fs)
    s = np.sin(np.radians(bearing))
    data = np.fft.ifft(spec[None, :] * np.exp(2j * np.pi * np.outer(sonar.element_offsets, f)
                                              * s / c), axis=1)
    beams = beamform(data, sonar, c)
    power = np.sum(np.abs(beams) ** 2, axis=1)
    step = np.diff(sonar.beam_angles_deg)[0]
    assert abs(sonar.beam_angles_deg[np.argmax(power)] - bearing) <= step / 2


def test_ambient_noise_power_matches_wenz_integral():
    env = Environment(sea_state=3)
    fs, fc, n = 100e3, 100e3, 1 << 15
    rng = np.random.default_rng(0)
    noise = np.fft.ifft(ambient_noise_spectrum(rng, 8, n, fs, fc, env), axis=1)
    f = fc + np.fft.fftfreq(n, 1 / fs)
    expected = 2 * np.sum(10 ** (wenz_spectrum_level(f, env.wind, env.shipping) / 10)) * fs / n
    np.testing.assert_allclose(np.mean(np.abs(noise) ** 2), expected, rtol=0.05)


def test_wenz_wind_raises_noise():
    quiet, windy = wenz_spectrum_level(50e3, 1.0), wenz_spectrum_level(50e3, 15.0)
    assert windy > quiet + 5


def test_pool_ranges_keeps_max_and_blanking():
    vals = np.arange(20, dtype=float)[None, :].repeat(2, 0)
    vals[:, :4] = np.nan
    pooled, r = pool_ranges(vals, np.arange(20.0), 5)
    assert pooled.shape == (2, 5) and np.isnan(pooled[0, 0])
    np.testing.assert_allclose(pooled[0, 1:], [7, 11, 15, 19])
    np.testing.assert_allclose(r, [1.5, 5.5, 9.5, 13.5, 17.5])
