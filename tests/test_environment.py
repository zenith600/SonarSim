import json

import numpy as np

from sonarsim.environment import (SEDIMENTS, Environment, eckart_reflection,
                                  francois_garrison_absorption, mackenzie_sound_speed,
                                  rayleigh_reflection)


def ainslie_mccolm(f_hz, t, s, z_m, ph=8.0):
    """Ainslie & McColm (1998) simplified absorption, dB/km, as an independent check."""
    f, z = f_hz / 1e3, z_m / 1e3
    f1 = 0.78 * np.sqrt(s / 35) * np.exp(t / 26)
    f2 = 42 * np.exp(t / 17)
    return (0.106 * f1 * f**2 / (f1**2 + f**2) * np.exp((ph - 8) / 0.56)
            + 0.52 * (1 + t / 43) * (s / 35) * f2 * f**2 / (f2**2 + f**2) * np.exp(-z / 6)
            + 0.00049 * f**2 * np.exp(-(t / 27 + z / 17)))


def test_mackenzie_reference_value():
    assert abs(mackenzie_sound_speed(0.0, 35.0, 0.0) - 1448.96) < 1e-9
    assert 1500 < mackenzie_sound_speed(15.0, 35.0, 20.0) < 1510


def test_francois_garrison_matches_ainslie_mccolm():
    f = np.array([10e3, 20e3, 40e3, 80e3, 120e3, 150e3])
    fg = francois_garrison_absorption(f, 10.0, 35.0, 20.0)
    am = ainslie_mccolm(f, 10.0, 35.0, 20.0)
    np.testing.assert_allclose(fg, am, rtol=0.15)
    assert np.all(np.diff(fg) > 0)


def test_eckart_limits():
    assert eckart_reflection(50e3, 0.3, 0.0, 1500.0) == -1.0
    low, high = abs(eckart_reflection([10e3, 100e3], 0.1, 0.05, 1500.0))
    assert 0 < high < low < 1


def test_rayleigh_normal_incidence_and_total_reflection():
    sand = SEDIMENTS["sand"]
    lossless = type(sand)("x", sand.density_ratio, sand.speed_ratio, 0.0, 0.0)
    m, n = lossless.density_ratio, 1.0 / lossless.speed_ratio
    np.testing.assert_allclose(rayleigh_reflection(np.pi / 2, lossless), (m - n) / (m + n))
    critical = np.arccos(1.0 / lossless.speed_ratio)
    assert abs(abs(rayleigh_reflection(0.5 * critical, lossless)) - 1.0) < 1e-12
    assert abs(rayleigh_reflection(0.5 * critical, sand)) < 1.0
    np.testing.assert_allclose(rayleigh_reflection(1e-6, sand), -1.0, atol=1e-4)


def test_environment_defaults_and_serialisation():
    env = Environment()
    depths, c = env.sound_speed_profile()
    assert c.min() <= env.mean_sound_speed <= c.max()
    assert env.wind == 4.4 and env.wave_height_rms > 0
    json.dumps(env.to_dict())


def test_environment_rejects_unknown_seabed():
    try:
        Environment(seabed="lava")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
