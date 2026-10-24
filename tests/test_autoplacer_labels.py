import json
import tempfile
from pathlib import Path

import numpy as np

from sonarsim.autoplacer import AutoPlacer, PlacementBounds
from sonarsim.dataset import generate_dataset, process_ping
from sonarsim.environment import Environment
from sonarsim.labels import make_label
from sonarsim.simulator import simulate_ping
from sonarsim.sonar import A_HF_imaging
from sonarsim.target import Mesh, Target

SONAR = A_HF_imaging().with_overrides(max_range=120.0, max_bounces=0)
ENV = Environment(depth=40.0)
TARGET = Target(Mesh.sphere(0.5, 20, 40))


def test_samples_respect_bounds_and_clearance():
    placer = AutoPlacer(SONAR, ENV, TARGET, PlacementBounds(clearance=1.0), seed=4)
    for sc in placer.samples(50):
        b = placer.bounds
        assert b.slant_range[0] <= sc.slant_range <= b.slant_range[1]
        assert abs(sc.bearing_deg) <= SONAR.sector_deg
        z = TARGET.world_vertices(sc.pose)[:, 2]
        assert z.max() <= -1.0 + 1e-9 and z.min() >= -ENV.depth + 1.0 - 1e-9
        rel = sc.pose.position - SONAR.position
        np.testing.assert_allclose(np.linalg.norm(rel), sc.slant_range)
        assert sc.environment.seabed in b.seabeds and sc.environment.sea_state in b.sea_states


def test_autoplacer_is_reproducible():
    a = [s.to_dict() for s in AutoPlacer(SONAR, ENV, TARGET, seed=11).samples(5)]
    b = [s.to_dict() for s in AutoPlacer(SONAR, ENV, TARGET, seed=11).samples(5)]
    assert a == b


def test_label_contents():
    sc = AutoPlacer(SONAR, ENV, TARGET, seed=2).sample()
    ping = simulate_ping(SONAR, sc.environment, TARGET, sc.pose, seed=sc.seed)
    prod = process_ping(ping)
    label = make_label(ping, sc, prod["profile_db"], prod["ranges"], prod["angles_deg"])
    json.dumps(label)
    t = label["target"]
    np.testing.assert_allclose(t["slant_range_m"], sc.slant_range)
    assert t["in_fov"] and label["arrivals"][0]["outgoing"] == "direct"
    assert label["detection"]["snr_db"] > 10
    assert label["bbox"]["range_m"][0] < t["slant_range_m"] < label["bbox"]["range_m"][1]


def test_generate_dataset_writes_files():
    with tempfile.TemporaryDirectory() as tmp:
        out = generate_dataset(SONAR, ENV, TARGET, tmp, n=2, seed=0, image_range_bins=256,
                               progress=None)
        files = sorted(p.name for p in Path(out).iterdir())
        assert files == ["manifest.json", "ping_00000.json", "ping_00000.npz",
                         "ping_00001.json", "ping_00001.npz"]
        with np.load(Path(out) / "ping_00000.npz") as z:
            assert z["image_db"].shape == (SONAR.n_beams, len(z["ranges"]))
            assert z["image_db"].shape[1] <= 256
        manifest = json.loads((Path(out) / "manifest.json").read_text())
        assert manifest["files"] == ["ping_00000", "ping_00001"]
