# acousticsim

**3D Underwater Acoustic Ray Tracing Simulator with ML Object Detection**

[![PyPI version](https://badge.fury.io/py/acousticsim.svg)](https://pypi.org/project/acousticsim/)
[![Python](https://img.shields.io/pypi/pyversions/acousticsim.svg)](https://pypi.org/project/acousticsim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`acousticsim` is a physics-based Python library for simulating underwater acoustic propagation using 3D ray tracing (30–200 kHz sonar range), and training machine learning models to detect and localize objects from acoustic echoes.

---

## Features

- **Accurate acoustic physics** — Mackenzie sound speed, Thorp absorption, geometric spreading, multipath propagation
- **3D ray tracing** — reflection off walls, seabed, surface and arbitrary objects; refraction via depth-varying sound speed
- **Import any 3D object** — load `.stl`, `.obj`, `.gltf`, `.glb` files and drop them into the simulation
- **Dual intersection modes** — fast AABB bounding box or exact per-triangle Möller–Trumbore
- **Full 6-DoF labeling** — position (x, y, z) + orientation (roll, pitch, yaw) per object
- **sklearn baseline** — Random Forest classifier + pose regressor
- **PyTorch multi-task MLP** — shared backbone → class head (CrossEntropy) + pose head (MSE), with early stopping
- **Dataset generation** — randomized placement → run simulation → extract features → labeled dataset (.npz)
- **Real-time streaming** — WebSocket server + browser 3D viewer

---

## Installation

```bash
# Core library (ray tracing only)
pip install acousticsim

# With ML object detection pipeline
pip install acousticsim[ml]

# With 3D mesh import (STL / OBJ / GLTF)
pip install acousticsim[mesh]

# Everything
pip install acousticsim[all]
```

---

## Quick Start

### Basic Simulation

```python
from acousticsim import Water, WaveEmitter, Hydrophone, Object3D, Simulation

water = Water(size=(100, 100, 50))          # 100m × 100m × 50m deep
sim   = Simulation(water)

wall = Object3D("Wall", position=[60, 50, 25], size=[1, 80, 40],
                reflection_coefficient=0.85)
sim.add_object(wall)

emitter = WaveEmitter(position=[10, 50, 25], frequency_khz=85.0,
                      beam_pattern='spherical', power_db=180.0)
sim.add_emitter(emitter)

hydro = Hydrophone(position=[50, 50, 25], name="Sensor_1")
sim.add_hydrophone(hydro)

results = sim.run(num_rays_per_emitter=1000)
print(f"Arrivals detected: {hydro.get_arrival_count()}")
print(f"First arrival:     {hydro.get_first_arrival()['time']*1000:.2f} ms")
```

### Import a 3D Object

```python
from acousticsim import MeshImporter, ImportedObject3D, Simulation, Water

mesh = MeshImporter.load("submarine.stl", scale=0.001)   # STL in mm → meters

water = Water(size=(200, 200, 100))
sim   = Simulation(water)

sub = ImportedObject3D(
    mesh_data=mesh,
    position=[100, 100, 50],
    rotation_deg=(0, 0, 45),        # yaw 45°
    intersection_mode='mesh',        # exact triangle intersection
    reflection_coefficient=0.7,
)
sim.add_object(sub)
```

### Generate a Training Dataset

```python
from acousticsim import MeshImporter, DatasetGenerator, DatasetConfig

meshes = {
    'submarine': MeshImporter.load('models/submarine.stl', scale=0.001),
    'mine':      MeshImporter.load('models/mine.obj',      scale=0.001),
    'rock':      MeshImporter.load('models/rock.glb'),
}

cfg = DatasetConfig(water_size=(100, 100, 50), num_rays=500)
gen = DatasetGenerator(meshes, config=cfg)

dataset = gen.generate(num_samples=2000, verbose=True)
dataset.save('acoustic_dataset.npz')
```

### Train & Detect

```python
from acousticsim import (
    AcousticDataset, train_sklearn, train_pytorch, evaluate, AcousticDetector
)

dataset = AcousticDataset.load('acoustic_dataset.npz')

# Train both models
sklearn_det, X_test, yc_test, yp_test = train_sklearn(dataset)
torch_det, _, _, _                    = train_pytorch(dataset, epochs=150)

# Evaluate
evaluate(X_test, yc_test, yp_test, dataset.class_names,
         sklearn_det=sklearn_det, pytorch_det=torch_det)

# Save & deploy
sklearn_det.save('sklearn_detector.pkl')

# Live detection on an unknown object
detector = AcousticDetector.from_sklearn('sklearn_detector.pkl')
result = detector.simulate_and_detect('unknown_object.stl', position=[40, 50, 25])
# → {'class_name': 'submarine', 'x': 41.2, 'y': 48.7, 'z': 24.1, ...}
```

---

## Coordinate System

```
Y (Length / front-back)
↑
|___→ X (Width / left-right)
/
↓
Z (Depth, positive downward)
```

Origin is at the front-left surface corner. `Z=0` = water surface, `Z=depth` = seabed.

---

## Physics

| Model | Implementation |
|---|---|
| Sound speed | Mackenzie (1981) equation — temperature, salinity, depth |
| Geometric spreading | Spherical: TL = 20 log₁₀(r) dB |
| Frequency absorption | Thorp equation (dB/km vs. kHz) |
| Reflection | Energy-based with configurable coefficient |
| Frequency-selective absorption | Per-frequency-band absorption coefficients on objects |
| Phase | Updated per ray step: Δφ = 2π · f · Δt |
| Ray termination | amplitude < 1×10⁻⁶, bounces > 15, or path > 1000 m |

---

## Project Structure

```
acousticsim/
├── water.py              Water environment (sound speed, absorption)
├── ray.py                Acoustic ray (propagation, phase, amplitude)
├── emitter.py            Wave emitter (spherical, cone, chirp)
├── hydrophone.py         Acoustic sensor (multipath detection)
├── object3d.py           Walls, seabed, surface (AABB geometry)
├── simulation.py         Ray tracing engine
├── mesh_importer.py      STL/OBJ/GLTF loader (trimesh)
├── imported_object3d.py  Mesh objects with AABB or triangle ray intersection
├── feature_extractor.py  Hydrophone arrivals → ML feature vectors
├── dataset_generator.py  Synthetic labeled dataset generation
├── train_detector.py     sklearn RF + PyTorch MLP training
└── inference.py          AcousticDetector deployment wrapper
```

---

## Requirements

- Python ≥ 3.8
- `numpy`, `scipy` (always)
- `trimesh[easy]` — for 3D mesh import
- `scikit-learn`, `torch` — for ML pipeline
- `aiohttp`, `websockets` — for real-time streaming
- `matplotlib` — for plotting

---

## License

MIT — see [LICENSE](LICENSE)
