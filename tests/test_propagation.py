import numpy as np

from sonarsim.environment import Environment
from sonarsim.propagation import eigenpaths

S = np.array([0.0, 0.0, -7.0])
T = np.array([100.0, 20.0, -25.0])
D = 40.0


def by_name(paths):
    return {p.name: p for p in paths}


def test_path_count_and_names():
    paths = eigenpaths(S, T, D, max_bounces=2)
    assert len(paths) == 5
    assert set(by_name(paths)) == {"direct", "surface", "bottom", "surface-bottom",
                                   "bottom-surface"}


def test_image_lengths_and_grazing():
    p = by_name(eigenpaths(S, T, D, max_bounces=1))
    h = np.hypot(*(T - S)[:2])
    np.testing.assert_allclose(p["direct"].length, np.linalg.norm(T - S))
    np.testing.assert_allclose(p["surface"].length, np.hypot(h, 7.0 + 25.0))
    np.testing.assert_allclose(p["bottom"].length, np.hypot(h, (D - 7.0) + (D - 25.0)))
    np.testing.assert_allclose(np.sin(p["bottom"].grazing), (2 * D - 32.0) / p["bottom"].length)


def test_directions_are_unit_and_consistent():
    for path in eigenpaths(S, T, D, max_bounces=2):
        np.testing.assert_allclose(np.linalg.norm(path.departure), 1.0)
        np.testing.assert_allclose(np.linalg.norm(path.arrival), 1.0)
        # Horizontal components point along the sonar-target line both ways.
        np.testing.assert_allclose(path.departure[:2] / np.linalg.norm(path.departure[:2]),
                                   -path.arrival[:2] / np.linalg.norm(path.arrival[:2]))
    p = by_name(eigenpaths(S, T, D, max_bounces=1))
    assert p["surface"].departure[2] > 0 and p["bottom"].departure[2] < 0


def test_direct_path_amplitude():
    env = Environment()
    path = eigenpaths(S, T, D, 0)[0]
    f = np.array([30e3, 100e3])
    expected = 10 ** (-env.absorption_db_per_km(f) * path.length / 20000) / path.length
    np.testing.assert_allclose(path.amplitude(f, env), expected)


def test_points_outside_water_rejected():
    try:
        eigenpaths(S, [10, 0, 1.0], D)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
