"""Target detection experiment across sonar configurations (paper Section IV-E).

    python examples/ml_experiment.py generate --preset A_HF_imaging -n 160 --out data/ml
    python examples/ml_experiment.py generate --preset B_LF_search  -n 160 --out data/ml
    python examples/ml_experiment.py train --data data/ml --out data/ml/results

``generate`` writes <preset>.npz holding n target pings (AutoPlacer scenarios)
and n target-free pings (reverberation and noise only) as sector images in dB
above the range median, max-pooled to a fixed number of range bins, with the
target's image coordinates. ``train`` fits a patch classifier (a one-hidden-
layer network in numpy) on A, on B and on both, and tests each on A and B:
ping-level detection AUC and localisation accuracy of the highest-scoring
cell (correct when it lies within the target's echo span, from the direct-path
front surface to the last multipath arrival, and within 2 beams). Only numpy
is needed.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import sonarsim as ss
from sonarsim.processing import pool_ranges

RANGE_BINS = 1024
PATCH = (5, 11)            # beams x range bins
TOL_RANGE_M = 2.0
TOL_BEAMS = 2


# ----------------------------------------------------------------------------
# data generation
# ----------------------------------------------------------------------------

def generate(args):
    sonar = ss.preset(args.preset)
    env = ss.Environment(depth=40.0)
    target = ss.Target(ss.Mesh.from_file(args.mesh, scale=args.scale))
    placer = ss.AutoPlacer(sonar, env, target, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    images, truth, meta = [], [], []
    t0 = time.perf_counter()
    for i in range(2 * args.num):
        with_target = i < args.num
        if with_target:
            sc = placer.sample()
            ping = ss.simulate_ping(sonar, sc.environment, target, sc.pose, seed=sc.seed)
        else:
            e = replace(env, seabed=str(rng.choice(list(ss.SEDIMENTS))),
                        sea_state=int(rng.choice([1, 2, 3, 4])))
            ping = ss.simulate_ping(sonar, e, seed=int(rng.integers(2**31 - 1)),
                                    include=("reverberation", "noise"))
        prod = ss.process_ping(ping)
        img, ranges = pool_ranges(prod["image_db"], prod["ranges"], RANGE_BINS)
        images.append(np.nan_to_num(img, nan=0.0).astype(np.float16))
        if with_target:
            # The echo spans the direct-path front surface to the last
            # multipath arrival above the noise; both bound a correct detection.
            r_front = sc.slant_range - target.mesh.radius
            r_last = max(a.range_m for a in ping.arrivals if a.level_db > 0) + target.mesh.radius
            truth.append([np.argmin(np.abs(ranges - r_front)),
                          np.argmin(np.abs(prod["angles_deg"] - sc.bearing_deg)),
                          np.argmin(np.abs(ranges - r_last)), sc.slant_range, sc.bearing_deg])
            meta.append({"seabed": sc.environment.seabed, "sea_state": sc.environment.sea_state})
        else:
            truth.append([-1, -1, -1, np.nan, np.nan])
            meta.append({"seabed": e.seabed, "sea_state": e.sea_state})
        print(f"[{args.preset}] {i + 1}/{2 * args.num} {'target' if with_target else 'clutter'}"
              f" ({(time.perf_counter() - t0) / (i + 1):.1f} s/ping)", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{args.preset}.npz", images=np.stack(images),
                        truth=np.array(truth, dtype=float), ranges=ranges,
                        angles_deg=prod["angles_deg"], meta=json.dumps(meta),
                        n_target=args.num)
    print("wrote", out / f"{args.preset}.npz")


# ----------------------------------------------------------------------------
# detector
# ----------------------------------------------------------------------------

def windows(img, patch=PATCH):
    """All patches of an image as (n_beams, n_bins, pb*pr); edges are padded."""
    pb, pr = patch
    padded = np.pad(img, ((pb // 2, pb // 2), (pr // 2, pr // 2)), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, patch)
    return view.reshape(img.shape[0], img.shape[1], pb * pr)


def features(patches):
    """Normalised patch plus its mean, max and centre value, scaled to about [0, 1]."""
    x = np.clip(patches, 0.0, 40.0) / 40.0
    centre = x[..., x.shape[-1] // 2:x.shape[-1] // 2 + 1]
    return np.concatenate([x, x.mean(-1, keepdims=True), x.max(-1, keepdims=True), centre], -1)


class PatchNet:
    """One-hidden-layer classifier trained with Adam on the logistic loss."""

    def __init__(self, n_in, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, np.sqrt(1 / hidden), hidden)
        self.b2 = 0.0

    def logits(self, x):
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        return h @ self.w2 + self.b2, h

    def predict(self, x):
        return 1.0 / (1.0 + np.exp(-self.logits(x)[0]))

    def fit(self, x, y, epochs=60, lr=2e-3, batch=256, l2=1e-4, seed=0):
        rng = np.random.default_rng(seed)
        params = [self.w1, self.b1, self.w2]
        m = [np.zeros_like(p) for p in params] + [0.0]
        v = [np.zeros_like(p) for p in params] + [0.0]
        step = 0
        for _ in range(epochs):
            for idx in np.array_split(rng.permutation(len(x)), max(1, len(x) // batch)):
                xb, yb = x[idx], y[idx]
                z, h = self.logits(xb)
                p = 1.0 / (1.0 + np.exp(-z))
                dz = (p - yb) / len(idx)
                gw2 = h.T @ dz + l2 * self.w2
                gb2 = dz.sum()
                dh = np.outer(dz, self.w2) * (h > 0)
                gw1 = xb.T @ dh + l2 * self.w1
                gb1 = dh.sum(0)
                grads = [gw1, gb1, gw2, gb2]
                step += 1
                for k, (pname, grad) in enumerate(zip(("w1", "b1", "w2", "b2"), grads)):
                    m[k] = 0.9 * m[k] + 0.1 * grad
                    v[k] = 0.999 * v[k] + 0.001 * grad**2
                    mh = m[k] / (1 - 0.9**step)
                    vh = v[k] / (1 - 0.999**step)
                    setattr(self, pname, getattr(self, pname) - lr * mh / (np.sqrt(vh) + 1e-8))
        return self


def training_set(ds, rng, negatives_per_ping=40):
    """Positive patch at the target cell; random and bright-spot negatives."""
    xs, ys = [], []
    n_target = int(ds["n_target"])
    for i, img in enumerate(ds["images"].astype(np.float32)):
        w = windows(img)
        nb, nr = img.shape
        if i < n_target:
            ri, bi, rl = (int(v) for v in ds["truth"][i, :3])
            # teacher-forced: the brightest cell inside the echo span
            b0, b1 = max(0, bi - TOL_BEAMS), min(nb, bi + TOL_BEAMS + 1)
            r0, r1 = max(0, ri - 8), min(nr, rl + 9)
            sub = img[b0:b1, r0:r1]
            db, dr = np.unravel_index(np.argmax(sub), sub.shape)
            xs.append(w[b0 + db, r0 + dr]); ys.append(1.0)
            far = np.ones((nb, nr), bool)
            far[max(0, bi - 6):bi + 7, max(0, ri - 40):rl + 41] = False
        else:
            far = np.ones((nb, nr), bool)
        cand = np.flatnonzero(far & (img > 0))
        pick = rng.choice(cand, negatives_per_ping // 2, replace=False)
        bright = cand[np.argsort(img.ravel()[cand])[-negatives_per_ping // 2:]]
        for flat in np.concatenate([pick, bright]):
            xs.append(w.reshape(-1, w.shape[-1])[flat]); ys.append(0.0)
    return features(np.array(xs)), np.array(ys)


def evaluate(net, ds):
    n_target = int(ds["n_target"])
    ranges = ds["ranges"]
    dr = float(ranges[1] - ranges[0])
    scores, hits = [], []
    for i, img in enumerate(ds["images"].astype(np.float32)):
        s = net.predict(features(windows(img)))
        s[:, ranges < 32.0] = 0.0                      # blanking zone
        scores.append(float(s.max()))
        if i < n_target:
            bi, ri = np.unravel_index(np.argmax(s), s.shape)
            ti, tb, tl = ds["truth"][i, :3]
            tol = TOL_RANGE_M / dr
            hits.append(ti - tol <= ri <= tl + tol and abs(bi - tb) <= TOL_BEAMS)
    scores = np.array(scores)
    pos, neg = scores[:n_target], scores[n_target:]
    auc = float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
    thr = np.quantile(neg, 0.9)                        # 10 % false-alarm pings
    return {"auc": auc, "localisation": float(np.mean(hits)),
            "pd_at_10pct_fa": float(np.mean(pos > thr)), "n_target": n_target,
            "n_clutter": int(len(neg))}


def train(args):
    data = Path(args.data)
    sets = {k: np.load(data / f"{k}.npz") for k in ("A_HF_imaging", "B_LF_search")}
    rng = np.random.default_rng(0)
    split = {}
    for k, ds in sets.items():
        n_t = int(ds["n_target"]); n = len(ds["images"])
        tr_t = np.arange(0, n_t // 2); te_t = np.arange(n_t // 2, n_t)
        tr_c = np.arange(n_t, n_t + (n - n_t) // 2); te_c = np.arange(n_t + (n - n_t) // 2, n)
        split[k] = (np.concatenate([tr_t, tr_c]), np.concatenate([te_t, te_c]))

    def subset(ds, idx):
        n_t = int(np.sum(ds["truth"][idx, 0] >= 0))
        order = np.concatenate([idx[ds["truth"][idx, 0] >= 0], idx[ds["truth"][idx, 0] < 0]])
        return {"images": ds["images"][order], "truth": ds["truth"][order], "n_target": n_t,
                "ranges": ds["ranges"], "angles_deg": ds["angles_deg"]}

    train_sets = {k: subset(ds, split[k][0]) for k, ds in sets.items()}
    test_sets = {k: subset(ds, split[k][1]) for k, ds in sets.items()}
    results = {}
    for name, keys in (("A", ["A_HF_imaging"]), ("B", ["B_LF_search"]),
                       ("A+B", ["A_HF_imaging", "B_LF_search"])):
        xs, ys = zip(*(training_set(train_sets[k], rng) for k in keys))
        x, y = np.concatenate(xs), np.concatenate(ys)
        net = PatchNet(x.shape[1]).fit(x, y)
        for tk, short in (("A_HF_imaging", "A"), ("B_LF_search", "B")):
            results[f"train {name} -> test {short}"] = evaluate(net, test_sets[tk])
            print(f"train {name:>3s} -> test {short}: {results[f'train {name} -> test {short}']}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("wrote", out / "results.json")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--preset", required=True, choices=sorted(ss.PRESETS))
    g.add_argument("-n", "--num", type=int, default=160, help="target pings (and as many clutter)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--mesh", default=str(Path(__file__).parent / "assets" / "AlienAnimal.obj"))
    g.add_argument("--scale", type=float, default=0.12)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=generate)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=train)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
