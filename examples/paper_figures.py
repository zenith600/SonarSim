"""Regenerate the six figures of the SonarSim paper with this package.

    python examples/paper_figures.py --out examples/figures

Needs matplotlib. The target is the paper's humanoid mesh, shipped as
examples/assets/AlienAnimal.obj (48,918 facets; scale 0.12 gives the paper's
1.40 x 2.49 x 0.45 m extent).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sonarsim as ss
from sonarsim.processing import find_peak, local_floor, pool_ranges
from sonarsim.target import triangle_kernel

ASSETS = Path(__file__).parent / "assets"
SEABED_COLORS = {"clay": "#4c72b0", "gravel": "#55a868", "rock": "#8c564b",
                 "sand": "#7f7f7f", "silt": "#17becf"}


def load_target(mesh_path, scale):
    return ss.Target(ss.Mesh.from_file(mesh_path, scale=scale))


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def fig1_target(target, out):
    mesh = target.mesh
    pts = mesh.sample_surface(4000, rng=0)
    views = {"front": (0, 0), "3/4": (20, -35), "side": (0, -90), "top": (90, 0)}
    fig = plt.figure(figsize=(12, 6))
    lim = mesh.extent.max() / 2
    for row, kind in enumerate(("mesh", "point cloud")):
        for col, (title, (elev, azim)) in enumerate(views.items()):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")
            if kind == "mesh":
                ax.plot_trisurf(*mesh.vertices.T, triangles=mesh.faces, color="#2b5d8a",
                                linewidth=0, antialiased=False, shade=True)
            else:
                ax.scatter(*pts.T, c=pts[:, 2], s=1, cmap="viridis")
            ax.view_init(elev=elev, azim=azim)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), title=f"{kind}: {title}")
            ax.set_box_aspect((1, 1, 1))
            ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    save(fig, out, "fig1_target")


def fig1_target_compact(target, out):
    """Two-panel version (mesh and point cloud, 3/4 view) for a single column."""
    mesh = target.mesh
    pts = mesh.sample_surface(4000, rng=0)
    lim = mesh.extent.max() / 2
    fig = plt.figure(figsize=(6.5, 3.6))
    for col, kind in enumerate(("mesh", "point cloud")):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        if kind == "mesh":
            ax.plot_trisurf(*mesh.vertices.T, triangles=mesh.faces, color="#2b5d8a",
                            linewidth=0, antialiased=False, shade=True)
        else:
            ax.scatter(*pts.T, c=pts[:, 2], s=1.5, cmap="viridis")
        ax.view_init(elev=20, azim=-35)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
        ax.set_title(f"({'ab'[col]}) {kind}", fontsize=10)
    save(fig, out, "fig1_target_compact")


def fig2_autoplacer(target, out, n=300):
    sonar = ss.B_LF_search()
    placer = ss.AutoPlacer(sonar, ss.Environment(depth=40.0), target, seed=0)
    scen = placer.samples(n)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for bed, color in SEABED_COLORS.items():
        sel = [s for s in scen if s.environment.seabed == bed]
        a.scatter([s.pose.position[0] for s in sel], [s.pose.position[1] for s in sel],
                  s=10, color=color, label=bed)
    a.plot(0, 0, "k^", ms=10, label="sonar")
    a.set(xlabel="east (m)", ylabel="north (m)", aspect="equal")
    a.legend(fontsize=8, loc="upper left")
    b.scatter([s.slant_range for s in scen], [-s.depth for s in scen], s=8)
    b.axhline(0, color="tab:cyan", label="surface")
    b.axhline(-40, color="saddlebrown", label="seabed (40 m)")
    b.set(xlabel="slant range (m)", ylabel="depth z (m)")
    b.legend(fontsize=8, loc="lower right")
    save(fig, out, "fig2_autoplacer")


def point_scatterer_ts(target, freqs, incident, c):
    """Ablation: each facet as a point scatterer at its centroid (no facet integral)."""
    tri = target.mesh.vertices[target.mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    n_hat = cross / area2[:, None]
    i_hat = np.asarray(incident, float) / np.linalg.norm(incident)
    lit = n_hat @ i_hat < 0
    g = -(n_hat[lit] @ i_hat) * area2[lit]
    phase = 2 * (tri[lit].mean(axis=1) @ i_hat)          # exp(i k q . r), q = -2 i
    k = 2 * np.pi * np.asarray(freqs) / c
    j0 = triangle_kernel(np.zeros(1), np.zeros(1))[0]   # = 1/2, the facet integral at k = 0
    f = 1j * k / (2 * np.pi) * ((np.exp(-1j * np.outer(k, phase)) * j0) @ g)
    return 20 * np.log10(np.abs(f))


def fig3_sphere(out, c=1500.0):
    freqs = np.geomspace(5e3, 150e3, 25)
    incident = np.array([0.31, -0.22, -0.92])
    fine = ss.Target(ss.Mesh.sphere(1.0, 300, 600))
    coarse = ss.Target(ss.Mesh.sphere(1.0, 60, 120))
    ts_fine = fine.target_strength(freqs, incident, sound_speed=c)
    ts_coarse = coarse.target_strength(freqs, incident, sound_speed=c)
    ts_point = point_scatterer_ts(coarse, freqs, incident, c)
    ref = 20 * np.log10(0.5)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.semilogx(freqs / 1e3, ts_fine, "o-", ms=4, label=f"facet integral, {fine.mesh.n_facets} facets")
    ax.semilogx(freqs / 1e3, ts_coarse, "s-", ms=3, label=f"facet integral, {coarse.mesh.n_facets} facets")
    ax.semilogx(freqs / 1e3, ts_point, "x--", ms=4, label=f"point scatterers, {coarse.mesh.n_facets} facets")
    ax.axhline(ref, color="k", ls=":", label="analytic 20 log10(a/2) = -6.02 dB")
    ax.set(xlabel="frequency (kHz)", ylabel="TS (dB re 1 m)")
    ax.legend(fontsize=7)
    save(fig, out, "fig3_sphere_validation")
    for name, ts in (("fine", ts_fine), ("coarse", ts_coarse), ("point", ts_point)):
        print(f"  sphere {name}: max |TS - ref| = {np.max(np.abs(ts - ref)):.2f} dB")


def fig_validation_combined(out, c=1500.0):
    """Sphere (a) and flat plate (b) validation side by side, single column."""
    freqs = np.geomspace(5e3, 150e3, 25)
    incident = np.array([0.31, -0.22, -0.92])
    fine = ss.Target(ss.Mesh.sphere(1.0, 300, 600))
    coarse = ss.Target(ss.Mesh.sphere(1.0, 60, 120))
    ref = 20 * np.log10(0.5)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    a.semilogx(freqs / 1e3, fine.target_strength(freqs, incident, sound_speed=c), "o-", ms=3,
               lw=0.9, label=f"facets, {fine.mesh.n_facets // 1000}k")
    a.semilogx(freqs / 1e3, coarse.target_strength(freqs, incident, sound_speed=c), "s-", ms=2.5,
               lw=0.9, label=f"facets, {coarse.mesh.n_facets // 1000}k")
    a.semilogx(freqs / 1e3, point_scatterer_ts(coarse, freqs, incident, c), "x--", ms=3, lw=0.9,
               label=f"point scatterers, {coarse.mesh.n_facets // 1000}k")
    a.axhline(ref, color="k", ls=":", label="analytic $-6.02$ dB")
    a.set(xlabel="frequency (kHz)", ylabel="TS (dB re 1 m)", title="(a) rigid sphere, $a$ = 1 m")
    a.legend(fontsize=6.5, loc="upper left")
    side, f = 0.5, 50e3
    k, lam = 2 * np.pi * f / c, c / f
    h = side / 2
    plate = ss.Target(ss.Mesh(np.array([[0, -h, -h], [0, h, -h], [0, h, h], [0, -h, h]], float),
                              np.array([[0, 1, 2], [0, 2, 3]]), "plate"))
    angles = np.linspace(-40, 40, 801)
    sim = [20 * np.log10(abs(plate.backscatter([f], [-np.cos(t), np.sin(t), 0], sound_speed=c)[0]))
           for t in np.radians(angles)]
    ana = [20 * np.log10(side**2 / lam * abs(np.cos(t)) * abs(np.sinc(k * side * np.sin(t) / np.pi)))
           for t in np.radians(angles)]
    b.plot(angles, ana, "k-", lw=2.2, alpha=0.3, label="closed form")
    b.plot(angles, sim, "C0-", lw=0.7, label="SonarSim, 2 facets")
    b.set(xlabel="incidence angle (deg)", ylabel="TS (dB re 1 m)", ylim=(-40, 25),
          title="(b) 0.5 m plate, 50 kHz")
    b.legend(fontsize=6.5, loc="upper right")
    for ax in (a, b):
        ax.title.set_fontsize(9)
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
    fig.tight_layout()
    save(fig, out, "fig_validation_combined")


def fig4_aspect(target, out, c=1500.0, step=0.5):
    aspects = np.arange(0.0, 360.0, step)
    ts = {}
    for f in (40e3, 100e3):
        vals = []
        for a in aspects:
            pose = ss.Pose.facing([100.0, 0, -10.0], a, [0, 0, -10.0])
            vals.append(target.target_strength([f], [1.0, 0, 0], pose, c)[0])
        ts[f] = np.array(vals)
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    for f, vals in ts.items():
        ax.plot(np.radians(aspects), vals, lw=0.6, label=f"{f / 1e3:.0f} kHz")
    ax.set_rlabel_position(135)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("TS (dB re 1 m) vs aspect; 0 deg = front, 90 deg = right side", fontsize=9)
    save(fig, out, "fig4_aspect_ts")


def scenario_ping(target, sonar, rng_m, bearing, depth, seabed, sea_state, aspect=0.0, seed=0):
    env = ss.Environment(depth=40.0, seabed=seabed, sea_state=sea_state)
    phi = np.radians(bearing)
    dz = depth - sonar.depth
    h = np.sqrt(rng_m**2 - dz**2)
    pose = ss.Pose.facing([h * np.cos(phi), h * np.sin(phi), -depth], aspect, sonar.position)
    return ss.simulate_ping(sonar, env, target, pose, seed=seed)


def fig5_and_6(target, out):
    cases = [(ss.A_HF_imaging(), 150.0, 8.0, 12.0, "sand", 2),
             (ss.B_LF_search(), 380.0, -15.0, 20.0, "rock", 4)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    first = None
    for ax, (sonar, r, brg, depth, bed, sea) in zip(axes, cases):
        ping = scenario_ping(target, sonar, r, brg, depth, bed, sea)
        prod = ss.process_ping(ping)
        first = first or (ping, prod)
        image, ranges = pool_ranges(prod["image_db"], prod["ranges"], 1500)
        im = ax.pcolormesh(ranges, prod["angles_deg"], image, vmin=0, vmax=40, cmap="viridis",
                           shading="auto")
        ax.plot(r, brg, "o", mfc="none", mec="r", ms=14, mew=1.2, label="ground truth")
        ax.set(xlabel="range (m)", ylabel="bearing (deg)",
               xlim=(sonar.blanking_range, sonar.max_range),
               title=f"{sonar.name}: {sonar.waveform.f_low / 1e3:.0f}-"
                     f"{sonar.waveform.f_high / 1e3:.0f} kHz, {bed}, sea state {sea}")
        ax.legend(fontsize=8, loc="upper left")
        fig.colorbar(im, ax=ax, label="contrast above range median (dB)")
    save(fig, out, "fig5_sector_images")

    ping, prod = first
    r, prof = prod["ranges"], prod["profile_db"]
    direct = ping.arrivals[0].range_m
    last = max(a.range_m for a in ping.arrivals if a.level_db > 0)
    pk_r, pk_db = find_peak(prof, r, direct - 3, last + 3)
    floor = local_floor(prof, r, 0.5 * (direct + last), 0.5 * (last - direct) + 3, width=10)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(r, prof, lw=0.5)
    ax.axvspan(0, ping.sonar.blanking_range, color="0.85", label="blanking")
    ax.axvline(direct, color="r", ls="--", lw=1, label=f"label: direct range {direct:.1f} m")
    for a in ping.arrivals:
        if a.level_db > 0 and a.range_m > direct + 0.1:
            ax.axvline(a.range_m, color="orange", ls=":", lw=1)
    ax.plot([], [], color="orange", ls=":", label="predicted multipath arrivals")
    ax.plot(pk_r, pk_db, "rv", label=f"peak {pk_r:.1f} m, {pk_db - floor:.1f} dB above local floor")
    ax.set(xlabel="range (m)", ylabel="matched-filter envelope (dB re 1 uPa)",
           xlim=(0, ping.sonar.max_range))
    ax.legend(fontsize=7, loc="lower left")
    save(fig, out, "fig6_range_profile")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mesh", default=str(ASSETS / "AlienAnimal.obj"))
    p.add_argument("--scale", type=float, default=0.12)
    p.add_argument("--out", default=str(Path(__file__).parent / "figures"))
    p.add_argument("--only", nargs="*", type=int, help="figure numbers to make (default all)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = load_target(args.mesh, args.scale)
    todo = set(args.only or range(1, 7))
    steps = [(1, lambda: (fig1_target(target, out), fig1_target_compact(target, out))),
             (2, lambda: fig2_autoplacer(target, out)),
             (3, lambda: (fig3_sphere(out), fig_validation_combined(out))),
             (4, lambda: fig4_aspect(target, out)),
             (5, lambda: fig5_and_6(target, out))]
    for num, fn in steps:
        if num in todo or (num == 5 and 6 in todo):
            t = time.perf_counter()
            fn()
            print(f"  ({time.perf_counter() - t:.1f} s)")


if __name__ == "__main__":
    main()
