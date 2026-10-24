"""Labelled dataset generation: AutoPlacer -> ping simulation -> processing -> files."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sonarsim._version import __version__
from sonarsim.autoplacer import AutoPlacer, PlacementBounds
from sonarsim.environment import Environment
from sonarsim.labels import make_label, save_ping
from sonarsim.processing import (beamform, matched_filter, pool_ranges, range_profile,
                                 sector_image)
from sonarsim.simulator import Ping, simulate_ping
from sonarsim.sonar import SonarConfig
from sonarsim.target import Target


def process_ping(ping: Ping, normalize: str = "range_median") -> dict:
    """Matched filter, beamform and form the sector image and range profile."""
    sonar = ping.sonar
    mf = matched_filter(ping.data, sonar.waveform, ping.sample_rate)
    beams = beamform(mf, sonar, ping.sound_speed)
    ranges = ping.ranges
    return {
        "ranges": ranges,
        "angles_deg": sonar.beam_angles_deg,
        "image_db": sector_image(beams, ranges, sonar.blanking_range, normalize),
        "profile_db": range_profile(mf),
        "beams": beams,
    }


def generate_dataset(sonar: SonarConfig, environment: Environment, target: Target, out_dir,
                     n: int, seed: int | None = None, bounds: PlacementBounds | None = None,
                     image_range_bins: int = 1024, save_raw: bool = False,
                     progress=print) -> Path:
    """Simulate ``n`` AutoPlacer scenarios and write one npz + json pair per ping.

    Images are max-pooled to ``image_range_bins`` range bins; the range
    profile is kept at full resolution. Returns the dataset directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    placer = AutoPlacer(sonar, environment, target, bounds, seed)
    manifest = {"sonarsim_version": __version__, "seed": seed, "n": n,
                "sonar": sonar.to_dict(), "base_environment": environment.to_dict(),
                "target": target.info(), "placement_bounds": placer.bounds.__dict__,
                "files": []}
    start = time.perf_counter()
    for i in range(n):
        scenario = placer.sample()
        ping = simulate_ping(sonar, scenario.environment, target, scenario.pose,
                             seed=scenario.seed)
        prod = process_ping(ping)
        label = make_label(ping, scenario, prod["profile_db"], prod["ranges"], prod["angles_deg"])
        image, image_ranges = pool_ranges(prod["image_db"], prod["ranges"], image_range_bins)
        stem = f"ping_{i:05d}"
        save_ping(out, stem, label, image_db=image.astype(np.float32), ranges=image_ranges,
                  angles_deg=prod["angles_deg"], profile_db=prod["profile_db"].astype(np.float32),
                  raw=ping.data.astype(np.complex64) if save_raw else None)
        manifest["files"].append(stem)
        if progress:
            elapsed = time.perf_counter() - start
            progress(f"[{i + 1}/{n}] {stem}: r={scenario.slant_range:.1f} m, "
                     f"brg={scenario.bearing_deg:.1f} deg, snr="
                     f"{label.get('detection', {}).get('snr_db', float('nan')):.1f} dB "
                     f"({elapsed / (i + 1):.1f} s/ping)")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                       encoding="utf-8")
    return out
