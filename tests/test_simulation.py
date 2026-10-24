import numpy as np

from sonarsim.dataset import process_ping
from sonarsim.environment import Environment
from sonarsim.processing import find_peak
from sonarsim.simulator import _band_limited_resample, simulate_ping
from sonarsim.sonar import A_HF_imaging
from sonarsim.target import Mesh, Pose, Target

SONAR = A_HF_imaging().with_overrides(max_range=170.0)
ENV = Environment()
SPHERE = Target(Mesh.sphere(0.5, 40, 80))
BEARING = 8.0
POSE = Pose([150 * np.cos(np.radians(BEARING)), 150 * np.sin(np.radians(BEARING)), -12.0])


def test_target_echo_lands_on_front_surface_and_bearing():
    ping = simulate_ping(SONAR, ENV, SPHERE, POSE, seed=1, include=("target",))
    prod = process_ping(ping)
    direct = ping.arrivals[0]
    assert direct.outgoing == direct.returning == "direct"
    peak_r, _ = find_peak(prod["profile_db"], prod["ranges"], 140, direct.range_m + 2)
    assert abs(peak_r - (direct.range_m - 0.5)) < 0.02
    window = (prod["ranges"] > peak_r - 1) & (prod["ranges"] < peak_r + 1)
    beam = np.argmax(np.max(np.abs(prod["beams"][:, window]), axis=1))
    step = np.diff(SONAR.beam_angles_deg)[0]
    assert abs(SONAR.beam_angles_deg[beam] - BEARING) <= step


def test_direct_echo_level_follows_sonar_equation():
    ping = simulate_ping(SONAR, ENV, SPHERE, POSE, seed=1, include=("target",))
    direct = ping.arrivals[0]
    r = direct.range_m
    tl = 20 * np.log10(r) + ENV.absorption_db_per_km(SONAR.center_frequency) * r / 1000
    ts = SPHERE.target_strength([SONAR.center_frequency], POSE.position - SONAR.position,
                                sound_speed=ping.sound_speed)
    np.testing.assert_allclose(direct.level_db, SONAR.source_level_db - 2 * tl + ts[0], atol=0.2)
    assert len(ping.arrivals) == 9


def test_band_limited_resample_matches_direct_evaluation():
    c = 1500.0
    support = 4 * SPHERE.mesh.radius / c
    df = 1 / (2 * support + 1e-4)
    coarse = 90e3 + df * np.arange(90)
    target = Target(Mesh.sphere(0.5, 40, 80))
    pose = Pose([0.0, 0.0, 0.0], yaw=20.0)
    incident = np.array([0.9, 0.1, -0.2])
    scattered = np.array([-0.95, 0.0, 0.3])          # bistatic, as for a mixed path
    vals = target.scattering_amplitude(coarse, incident, scattered, pose, c)
    fine = np.linspace(coarse[25], coarse[-25], 53)
    direct = target.scattering_amplitude(fine, incident, scattered, pose, c)
    got = _band_limited_resample(coarse, vals, fine, support)
    assert np.max(np.abs(got - direct)) < 1e-5 * np.max(np.abs(direct))


def test_same_seed_reproduces_ping():
    a = simulate_ping(SONAR, ENV, SPHERE, POSE, seed=7, include=("reverberation", "noise"))
    b = simulate_ping(SONAR, ENV, SPHERE, POSE, seed=7, include=("reverberation", "noise"))
    np.testing.assert_array_equal(a.data, b.data)


def test_reverberation_decays_with_range_for_a_flat_vertical_beam():
    flat = SONAR.with_overrides(vertical_beamwidth_deg=1e4, tilt_deg=0.0)
    ping = simulate_ping(flat, ENV, include=("reverberation",), seed=3)
    r = ping.ranges
    power = np.mean(np.abs(ping.data) ** 2, axis=0)
    near = power[(r > 40) & (r < 50)].mean()
    far = power[(r > 140) & (r < 150)].mean()
    assert near > 10 * far
