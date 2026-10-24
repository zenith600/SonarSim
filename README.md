# sonarsim

Physics-based generation of labelled active sonar datasets for arbitrary
triangular-mesh targets and configurable sonars. This is the code behind the
paper *SonarSim: A Physics-Based Framework for Generating Labelled Active
Sonar Datasets*.

Each simulated ping gives the complex signals on every hydrophone, the
beamformed sector image and matched-filter range profile, and a JSON label
holding the exact scenario that produced them.

> **Provenance.** The version used for the paper was lost with a laptop. This
> package was rebuilt in October 2026 from the paper's description and the
> surviving older code (`acousticsim`, kept as the first commit in this
> repository's history). See [CHANGELOG.md](CHANGELOG.md) for what that means
> for reproducing the paper's numbers.

## Install

```bash
pip install -e ".[dev]"
```

Only `numpy` and `scipy` are required. OBJ meshes load without extra
packages; `pip install sonarsim[mesh]` adds STL/GLTF/PLY via trimesh, and
`sonarsim[plot]` adds matplotlib for the figure script.

## Quick start

```python
import numpy as np
import sonarsim as ss

sonar = ss.A_HF_imaging()                      # 80-120 kHz LFM, 48-element array
env = ss.Environment(depth=40.0, seabed="sand", sea_state=2)
target = ss.Target(ss.Mesh.from_file("examples/assets/AlienAnimal.obj", scale=0.12))

# Target about 150 m away at bearing 8 deg, 12 m deep, chest towards the sonar.
pos = [149.3 * np.cos(np.radians(8)), 149.3 * np.sin(np.radians(8)), -12.0]
pose = ss.Pose.facing(pos, aspect_deg=0.0, sonar_position=sonar.position)

ping = ss.simulate_ping(sonar, env, target, pose, seed=0)
products = ss.process_ping(ping)               # image_db, profile_db, ranges, angles_deg
label = ss.make_label(ping, profile_db=products["profile_db"],
                      image_ranges=products["ranges"], angles_deg=products["angles_deg"])
```

Generate a dataset with randomised placements (AutoPlacer):

```bash
sonarsim-generate --mesh examples/assets/AlienAnimal.obj --scale 0.12 \
    --preset A_HF_imaging -n 300 --seed 0 --out data/A_HF
```

Each ping is written as `ping_NNNNN.npz` (sector image max-pooled to 1024
range bins, full-resolution range profile, axes; raw element signals with
`--save-raw`) and `ping_NNNNN.json` (label), plus a `manifest.json`.

## Physics models

| Component | Model |
|---|---|
| Sound speed | Mackenzie (1981) nine-term equation; straight rays at the depth-averaged speed |
| Absorption | Francois-Garrison (1982) |
| Multipath | Method of images in a flat waveguide, up to `max_bounces` reflections per one-way path |
| Surface reflection | Eckart (1953) coherent loss; Pierson-Moskowitz wave height from wind speed |
| Bottom reflection | Rayleigh fluid-fluid coefficient; five seabed types |
| Target scattering | Kirchhoff (physical optics) with closed-form triangle integrals; rigid or pressure-release |
| Reverberation | Lambert's law for seabed and surface, complex-Gaussian patches, direct paths only |
| Ambient noise | Wenz curves (Coates 1990 parametric form) |
| Receiver | Horizontal uniform line array, true-time-delay beamforming, matched filter |
| Waveforms | LFM and CW, Tukey-windowed |

Conventions: x east, y north, z up, sea surface at z = 0. The sonar sits at
(0, 0, -depth) looking east; bearing 0 is broadside and positive bearings are
towards north. Signals are complex baseband pressures in uPa and the
time convention is exp(+i 2 pi f t).

## Reproducing the paper's figures

```bash
python examples/paper_figures.py --out examples/figures
```

`examples/assets/AlienAnimal.obj` is the humanoid mesh used in the paper
(48,918 facets; scale 0.12 gives its 1.40 x 2.49 x 0.45 m extent).

## Tests

```bash
pytest
```

The tests check the physics against closed-form results: flat-plate target
strength and its sinc pattern, the rigid-sphere -6.02 dB limit, the triangle
integral against quadrature, image-path geometry, matched-filter timing,
beam steering and the ambient-noise level.

## Limitations

Straight-ray propagation (no refraction); Kirchhoff scattering only, so no
multiple scattering, diffraction, creeping waves, elastic response or
self-shadowing; far-field scattering, so targets closer than about 2L^2/lambda
are approximate; reverberation via direct paths only, no volume
reverberation; no platform or target motion.
