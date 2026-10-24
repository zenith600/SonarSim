import tempfile
from pathlib import Path

import numpy as np

from sonarsim.target import Mesh, Pose, Target, triangle_kernel

C = 1500.0


def quad_kernel(a, b, n=200_000):
    """J(a, b) with the inner integral done analytically, outer by midpoint rule."""
    u = (np.arange(n) + 0.5) / n
    if abs(b) < 1e-12:
        inner = 1 - u
    else:
        inner = (np.exp(1j * b * (1 - u)) - 1) / (1j * b)
    return np.mean(np.exp(1j * a * u) * inner)


def test_triangle_kernel_matches_quadrature_including_degenerate_cases():
    cases = [(3.1, -7.4), (0.0, 0.0), (0.0, 5.0), (5.0, 0.0), (2.0, 2.0 + 1e-7),
             (40.0, 39.99999), (1e-6, 2e-6), (-120.0, 37.0)]
    for a, b in cases:
        got = triangle_kernel(np.array([a]), np.array([b]))[0]
        np.testing.assert_allclose(got, quad_kernel(a, b), rtol=1e-6, atol=1e-9)


def square_plate(side):
    h = side / 2
    v = np.array([[0, -h, -h], [0, h, -h], [0, h, h], [0, -h, h]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = Mesh(v, f, "plate")          # normal along +x
    return Target(mesh)


def test_flat_plate_normal_incidence_is_exact():
    side, f = 0.5, 50e3
    lam = C / f
    ts = square_plate(side).target_strength([f], [-1.0, 0, 0], sound_speed=C)[0]
    np.testing.assert_allclose(ts, 20 * np.log10(side**2 / lam), atol=1e-9)


def test_flat_plate_angular_pattern_is_sinc():
    side, f = 0.5, 50e3
    k = 2 * np.pi * f / C
    target = square_plate(side)
    for theta in np.radians([2.0, 5.0, 11.0]):
        incident = [-np.cos(theta), np.sin(theta), 0.0]   # tilted in the plate's y direction
        got = abs(target.backscatter([f], incident, sound_speed=C)[0])
        x = k * side * np.sin(theta)
        expected = side**2 / (C / f) * np.cos(theta) * abs(np.sin(x) / x)
        np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_rigid_sphere_target_strength():
    sphere = Target(Mesh.sphere(1.0, 120, 240))
    ts = sphere.target_strength([20e3, 100e3], [0.3, -0.2, -0.93], sound_speed=C)
    np.testing.assert_allclose(ts, 20 * np.log10(0.5), atol=0.3)


def test_pressure_release_flips_sign_only():
    rigid = square_plate(0.3)
    soft = Target(rigid.mesh, boundary="pressure_release")
    a = rigid.backscatter([40e3], [-1, 0.1, 0])[0]
    b = soft.backscatter([40e3], [-1, 0.1, 0])[0]
    np.testing.assert_allclose(b, -a)


def test_pose_facing_points_front_at_sonar():
    pose = Pose.facing([100.0, 30.0, -10.0], aspect_deg=0.0, sonar_position=[0, 0, -7])
    front = pose.rotation @ np.array([1.0, 0, 0])
    to_sonar = np.array([-100.0, -30.0, 0.0])
    np.testing.assert_allclose(front[:2], to_sonar[:2] / np.linalg.norm(to_sonar[:2]))
    right = Pose.facing([100.0, 30.0, -10.0], 90.0, [0, 0, -7]).rotation @ np.array([0, -1.0, 0])
    np.testing.assert_allclose(right[:2], to_sonar[:2] / np.linalg.norm(to_sonar[:2]), atol=1e-12)


def test_obj_loader_triangulates_and_reorients():
    obj = "v 0 0 0\nv 1 0 0\nv 1 2 0\nv 0 2 0\nv 0 0 3\nf 1 2 3 4\nf 1/1/1 2/2/2 5/5/5\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "box.obj"
        path.write_text(obj)
        mesh = Mesh.from_file(path, scale=2.0, up_axis="+y", front_axis="+z")
    assert mesh.n_facets == 3
    # File y (height 2 * 2 = 4) becomes world z; file z (6) becomes world x.
    np.testing.assert_allclose(mesh.extent, [6.0, 2.0, 4.0])
    np.testing.assert_allclose(mesh.vertices.min(0) + mesh.vertices.max(0), 0.0, atol=1e-12)
