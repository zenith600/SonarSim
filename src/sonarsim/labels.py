"""Ground-truth labels and on-disk format for simulated pings.

Each ping is stored as ``<stem>.npz`` (arrays) and ``<stem>.json`` (label).
The label is exact by construction: it records the scenario that produced
the observation, plus the predicted multipath arrivals and a measured SNR.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sonarsim._version import __version__
from sonarsim.processing import find_peak, local_floor

SCHEMA_VERSION = "1.0"


def image_box(ping, angles_deg: np.ndarray, ranges: np.ndarray) -> dict:
    """Target extent in image coordinates (beam and range-sample indices)."""
    rel = ping.target.world_vertices(ping.pose) - ping.sonar.position
    slant = np.linalg.norm(rel, axis=1)
    bearing = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    r0, r1 = slant.min(), slant.max()
    b0, b1 = bearing.min(), bearing.max()
    rows = np.nonzero((angles_deg >= b0) & (angles_deg <= b1))[0]
    if rows.size == 0:
        rows = np.array([np.argmin(np.abs(angles_deg - 0.5 * (b0 + b1)))])
    cols = np.nonzero((ranges >= r0) & (ranges <= r1))[0]
    return {"range_m": [float(r0), float(r1)], "bearing_deg": [float(b0), float(b1)],
            "beam_index": [int(rows.min()), int(rows.max())],
            "range_index": [int(cols.min()), int(cols.max())] if cols.size else None}


def make_label(ping, scenario=None, profile_db=None, image_ranges=None, angles_deg=None) -> dict:
    """Assemble the JSON-serialisable label for a simulated ping."""
    sonar, env = ping.sonar, ping.environment
    label = {
        "schema_version": SCHEMA_VERSION,
        "sonarsim_version": __version__,
        "seed": ping.seed,
        "sonar": sonar.to_dict(),
        "environment": env.to_dict(),
        "processing": {"sound_speed": ping.sound_speed, "sample_rate": ping.sample_rate,
                       "range_resolution_m": sonar.range_resolution(ping.sound_speed)},
    }
    if ping.target is None:
        return label

    rel = ping.pose.position - sonar.position
    slant = float(np.linalg.norm(rel))
    bearing = float(np.degrees(np.arctan2(rel[1], rel[0])))
    label["target"] = {
        **ping.target.info(), **ping.pose.to_dict(),
        "slant_range_m": slant,                      # direct path, sonar to mesh centre
        "horizontal_range_m": float(np.hypot(rel[0], rel[1])),
        "bearing_deg": bearing,
        "depth_m": float(-ping.pose.position[2]),
        "in_fov": bool(abs(bearing) <= sonar.sector_deg and slant <= sonar.max_range),
    }
    label["arrivals"] = [a.to_dict() for a in ping.arrivals]
    if scenario is not None:
        label["scenario"] = scenario.to_dict()
    if profile_db is not None and image_ranges is not None:
        extent = ping.target.mesh.radius + 1.0
        last = max(a.range_m for a in ping.arrivals)
        peak_r, peak_db = find_peak(profile_db, image_ranges, slant - extent, last + extent)
        floor_db = local_floor(profile_db, image_ranges, 0.5 * (slant + last),
                               0.5 * (last - slant) + extent)
        label["detection"] = {"peak_range_m": peak_r, "peak_level_db": peak_db,
                              "local_floor_db": floor_db, "snr_db": peak_db - floor_db}
    if angles_deg is not None and image_ranges is not None:
        label["bbox"] = image_box(ping, np.asarray(angles_deg), np.asarray(image_ranges))
    return label


def save_ping(directory, stem: str, label: dict, *, image_db=None, ranges=None,
              angles_deg=None, profile_db=None, raw=None) -> tuple[Path, Path]:
    """Write ``<stem>.npz`` with the arrays given and ``<stem>.json`` with the label."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in dict(image_db=image_db, ranges=ranges, angles_deg=angles_deg,
                                    profile_db=profile_db, raw=raw).items() if v is not None}
    npz = directory / f"{stem}.npz"
    np.savez_compressed(npz, **{k: np.asarray(v) for k, v in arrays.items()})
    js = directory / f"{stem}.json"
    js.write_text(json.dumps(label, indent=2, default=_json_default), encoding="utf-8")
    return npz, js


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")
