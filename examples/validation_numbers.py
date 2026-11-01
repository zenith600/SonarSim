"""Quantitative checks quoted in the paper (Sections IV-A to IV-D).

    python examples/validation_numbers.py --out examples/figures

Writes validation.json with every number and two figures: the flat-plate
angular pattern against its closed form, and sphere mesh convergence.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sonarsim as ss
from sonarsim.processing import find_peak, local_floor

ASSETS = Path(__file__).parent / "assets"
C = 1500.0


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plate_target(side):
    h = side / 2
    v = np.array([[0, -h, -h], [0, h, -h], [0, h, h], [0, -h, h]], float)
    return ss.Target(ss.Mesh(v, np.array([[0, 1, 2], [0, 2, 3]]), "plate"))


def plate_check(out, side=0.5, f=50e3):
    k, lam = 2 * np.pi * f / C, C / f
    target = plate_target(side)
    angles = np.linspace(-40, 40, 801)
    sim, ana = [], []
    for a in np.radians(angles):
        inc = [-np.cos(a), np.sin(a), 0.0]
        sim.append(20 * np.log10(abs(target.backscatter([f], inc, sound_speed=C)[0])))
        x = k * side * np.sin(a)
        ana.append(20 * np.log10(side**2 / lam * abs(np.cos(a)) * abs(np.sinc(x / np.pi))))
    sim, ana = np.array(sim), np.array(ana)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(angles, ana, "k-", lw=2.5, alpha=0.35, label="closed form (Kirchhoff plate)")
    ax.plot(angles, sim, "C0-", lw=0.9, label="SonarSim, two facets")
    ax.set(xlabel="incidence angle from normal (deg)", ylabel="TS (dB re 1 m)",
           ylim=(ana.max() - 60, ana.max() + 5))
    ax.legend(fontsize=8)
    save(fig, out, "fig_plate_validation")
    keep = ana > ana.max() - 50
    return {"side_m": side, "frequency_hz": f, "ts_normal_db": float(sim[400]),
            "ts_normal_analytic_db": float(20 * np.log10(side**2 / lam)),
            "max_abs_error_db_within_50dB": float(np.max(np.abs(sim - ana)[keep])),
            "n_angles": len(angles)}


def sphere_convergence(out, freqs=(20e3, 50e3, 100e3, 150e3)):
    inc = np.array([0.31, -0.22, -0.92])
    ref = 20 * np.log10(0.5)
    rows = []
    for n_lat in (30, 60, 120, 240):
        t = ss.Target(ss.Mesh.sphere(1.0, n_lat, 2 * n_lat))
        edge = np.pi / n_lat                       # facet size (m) for a = 1 m
        ts = t.target_strength(list(freqs), inc, sound_speed=C)
        rows.append({"n_facets": t.mesh.n_facets, "facet_edge_m": edge,
                     "edge_over_lambda_at_100kHz": edge / (C / 100e3),
                     "max_abs_error_db": float(np.max(np.abs(ts - ref))),
                     "error_db_per_freq": {f"{f / 1e3:.0f}": float(v - ref) for f, v in zip(freqs, ts)}})
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.semilogx([r["n_facets"] for r in rows], [r["max_abs_error_db"] for r in rows], "o-")
    for r in rows:
        ax.annotate(f"{r['edge_over_lambda_at_100kHz']:.1f} λ", (r["n_facets"], r["max_abs_error_db"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set(xlabel="facets", ylabel="max |TS − analytic| (dB), 20–150 kHz")
    ax.grid(True, which="both", alpha=0.3)
    save(fig, out, "fig_sphere_convergence")
    return rows


def aspect_stats(target):
    stats = {}
    for f in (40e3, 100e3):
        vals = []
        for a in np.arange(0.0, 360.0, 0.5):
            pose = ss.Pose.facing([100.0, 0, -10.0], a, [0, 0, -10.0])
            vals.append(target.target_strength([f], [1.0, 0, 0], pose, C)[0])
        v = np.array(vals)
        stats[f"{f / 1e3:.0f}kHz"] = {"min_db": float(v.min()), "max_db": float(v.max()),
                                      "spread_db": float(v.max() - v.min()),
                                      "median_db": float(np.median(v)),
                                      "front_db": float(v[0]), "side_db": float(v[180]),
                                      "back_db": float(v[360])}
    return stats


def hann_beamwidth_deg(sonar, f):
    """-3 dB beamwidth of the Hann-shaded array at frequency f (numerical)."""
    w = sonar.shading_weights
    y = sonar.element_offsets
    th = np.radians(np.linspace(-10, 10, 20001))
    pattern = np.abs(np.exp(2j * np.pi * f / C * np.outer(np.sin(th), y)) @ w)
    pattern /= pattern.max()
    half = th[pattern >= 10 ** (-3 / 20)]
    return float(np.degrees(half.max() - half.min()))


def config_checks(target):
    out = {}
    for sonar, rng, brg, depth, bed, sea in ((ss.A_HF_imaging(), 150.0, 8.0, 12.0, "sand", 2),
                                             (ss.B_LF_search(), 380.0, -15.0, 20.0, "rock", 4)):
        env = ss.Environment(depth=40.0, seabed=bed, sea_state=sea)
        phi = np.radians(brg)
        dz = depth - sonar.depth
        h = np.sqrt(rng**2 - dz**2)
        pose = ss.Pose.facing([h * np.cos(phi), h * np.sin(phi), -depth], 0.0, sonar.position)
        t0 = time.perf_counter()
        ping = ss.simulate_ping(sonar, env, target, pose, seed=0)
        t_sim = time.perf_counter() - t0
        t0 = time.perf_counter()
        prod = ss.process_ping(ping)
        t_proc = time.perf_counter() - t0
        r, prof = prod["ranges"], prod["profile_db"]
        direct = ping.arrivals[0]
        last = max(a.range_m for a in ping.arrivals if a.level_db > 0)
        pk_r, pk_db = find_peak(prof, r, direct.range_m - 3, last + 3)
        floor = local_floor(prof, r, 0.5 * (direct.range_m + last), 0.5 * (last - direct.range_m) + 3)
        # sonar-equation check on the direct arrival
        fc = sonar.center_frequency
        tl = 20 * np.log10(direct.range_m) + float(env.absorption_db_per_km(fc)) * direct.range_m / 1000
        ts = target.target_strength([fc], pose.position - sonar.position, pose, ping.sound_speed)[0]
        # beam of the brightest cell near the target
        win = (r > direct.range_m - 3) & (r < last + 3)
        beam = int(np.argmax(np.nanmax(prod["image_db"][:, win], axis=1)))
        out[sonar.name] = {
            "n_samples": ping.n_samples, "sim_seconds": t_sim, "proc_seconds": t_proc,
            "sound_speed": ping.sound_speed, "fs": sonar.fs, "spacing_m": sonar.spacing,
            "range_resolution_m": sonar.range_resolution(ping.sound_speed),
            "beamwidth_hann_deg": hann_beamwidth_deg(sonar, fc),
            "beamwidth_uniform_deg": float(np.degrees(0.886 * C / fc / (sonar.n_elements * sonar.spacing))),
            "near_field_2L2_over_lambda_m": float(2 * target.mesh.extent.max() ** 2 / (C / sonar.waveform.f_high)),
            "label_direct_range_m": direct.range_m, "peak_range_m": pk_r, "peak_db": pk_db,
            "local_floor_db": floor, "snr_db": pk_db - floor,
            "brightest_beam_deg": float(prod["angles_deg"][beam]), "true_bearing_deg": brg,
            "direct_level_db": direct.level_db,
            "sonar_equation_db": float(sonar.source_level_db - 2 * tl + ts),
            "arrivals": [a.to_dict() for a in ping.arrivals if a.level_db > 0],
            "absorption_db_per_km_fc": float(env.absorption_db_per_km(fc)),
            "wind_speed": env.wind, "wave_height_rms_m": env.wave_height_rms,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path(__file__).parent / "figures"))
    p.add_argument("--mesh", default=str(ASSETS / "AlienAnimal.obj"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = ss.Target(ss.Mesh.from_file(args.mesh, scale=0.12))
    env = ss.Environment(depth=40.0)
    depths, c = env.sound_speed_profile()
    results = {
        "target": target.info(),
        "environment": {"c_surface": float(c[0]), "c_bottom": float(c[-1]),
                        "c_mean": env.mean_sound_speed,
                        "absorption_db_per_km": {f"{f / 1e3:.0f}": float(env.absorption_db_per_km(f))
                                                 for f in (20e3, 30e3, 40e3, 80e3, 100e3, 120e3)},
                        "sediments": {k: v.__dict__ for k, v in ss.SEDIMENTS.items()},
                        "wind_by_sea_state": ss.environment.SEA_STATE_WIND_SPEED,
                        "surface_mu_by_sea_state": ss.environment.SEA_STATE_SURFACE_MU_DB},
        "plate": plate_check(out),
        "sphere_convergence": sphere_convergence(out),
        "aspect": aspect_stats(target),
        "configs": config_checks(target),
    }
    (out / "validation.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
