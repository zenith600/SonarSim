"""
Acoustic Object Detector — Inference
Run a trained detector against a live or pre-computed simulation.

Two modes:
  1. simulate_and_detect():
       Place an unknown object in water, run full simulation, predict class + pose.

  2. detect_from_hydrophones():
       Pass already-populated Hydrophone objects → predict class + pose.

Example:
    from acousticsim.inference import AcousticDetector

    detector = AcousticDetector.from_sklearn('sklearn_detector.pkl')
    # -- or --
    detector = AcousticDetector.from_pytorch(
        'pytorch_detector.pt', input_dim=50, num_classes=3
    )

    result = detector.simulate_and_detect(
        mesh_filepath='models/unknown.stl',
        position=[40, 50, 25],
        water_size=(100, 100, 50),
    )
    print(result)
"""

import numpy as np
from pathlib import Path

from acousticsim.water import Water
from acousticsim.emitter import WaveEmitter
from acousticsim.hydrophone import Hydrophone
from acousticsim.simulation import Simulation
from acousticsim.mesh_importer import MeshImporter
from acousticsim.imported_object3d import ImportedObject3D
from acousticsim.feature_extractor import extract_features
from acousticsim.dataset_generator import DatasetConfig


class AcousticDetector:
    """
    High-level inference wrapper for trained acoustic object detectors.

    Supports both sklearn (SklearnDetector) and PyTorch (PyTorchDetector) backends.
    """

    def __init__(self, detector, backend_name="unknown", config=None):
        """
        Args:
            detector:      SklearnDetector or PyTorchDetector
            backend_name:  Human-readable name for display
            config:        DatasetConfig used during training (for sensor layout etc.)
        """
        self.detector     = detector
        self.backend_name = backend_name
        self.config       = config or DatasetConfig()
        self.class_names  = getattr(detector, 'class_names', None) or []

    # ─── Factory constructors ─────────────────────────────────────────────

    @classmethod
    def from_sklearn(cls, filepath, config=None):
        """
        Load a saved SklearnDetector.

        Args:
            filepath: Path to .pkl file saved by SklearnDetector.save()
            config:   DatasetConfig matching the training setup

        Returns:
            AcousticDetector
        """
        from acousticsim.train_detector import SklearnDetector
        det = SklearnDetector.load(filepath)
        return cls(det, backend_name="sklearn (RF)", config=config)

    @classmethod
    def from_pytorch(cls, filepath, input_dim, num_classes,
                     hidden_dims=(256, 128, 64), dropout=0.3, config=None):
        """
        Load a saved PyTorchDetector.

        Args:
            filepath:    Path to .pt checkpoint saved by PyTorchDetector.save()
            input_dim:   Feature vector length (must match saved model)
            num_classes: Number of classes (must match saved model)
            config:      DatasetConfig matching the training setup

        Returns:
            AcousticDetector
        """
        from acousticsim.train_detector import PyTorchDetector
        det = PyTorchDetector.load(
            filepath, input_dim=input_dim, num_classes=num_classes,
            hidden_dims=hidden_dims, dropout=dropout,
        )
        return cls(det, backend_name="PyTorch MLP", config=config)

    # ─── Inference modes ─────────────────────────────────────────────────

    def simulate_and_detect(
        self,
        mesh_filepath,
        position,
        rotation_deg=(0.0, 0.0, 0.0),
        mesh_scale=1.0,
        intersection_mode='aabb',
        config=None,
        verbose=True,
    ):
        """
        Import a 3D object, run a full acoustic simulation, predict class and pose.

        Args:
            mesh_filepath:     Path to STL / OBJ / GLTF file
            position:          (x, y, z) where the object is placed [meters]
            rotation_deg:      (roll, pitch, yaw) orientation [degrees]
            mesh_scale:        Scale factor applied to the mesh
            intersection_mode: 'aabb' or 'mesh'
            config:            Override DatasetConfig (uses self.config if None)
            verbose:           Print progress

        Returns:
            dict with keys:
                class_id, class_name, class_proba,
                x, y, z, roll_deg, pitch_deg, yaw_deg,
                feature_vector, hydrophones
        """
        cfg = config or self.config

        if verbose:
            print(f"\n[AcousticDetector] Loading mesh: {mesh_filepath}")

        mesh_data = MeshImporter.load(mesh_filepath, scale=mesh_scale)

        if verbose:
            print(f"  Loaded: {mesh_data}")
            print(f"  Simulating at position={position}, rotation={rotation_deg}")

        hydrophones = self._run_simulation(
            mesh_data=mesh_data,
            position=position,
            rotation_deg=rotation_deg,
            intersection_mode=intersection_mode,
            cfg=cfg,
            verbose=verbose,
        )

        feat_vec, _ = extract_features(hydrophones)
        result = self.detector.predict_single(feat_vec)
        result['feature_vector'] = feat_vec
        result['hydrophones']    = hydrophones

        if verbose:
            self._print_result(result)

        return result

    def detect_from_hydrophones(self, hydrophones, verbose=True):
        """
        Predict directly from already-populated Hydrophone objects.

        Use this if you have already run a simulation manually and just want
        to classify the echoes.

        Args:
            hydrophones: list of Hydrophone objects (with .arrivals populated)
            verbose:     Print result

        Returns:
            dict with class_id, class_name, class_proba, x, y, z, roll/pitch/yaw_deg
        """
        feat_vec, _ = extract_features(hydrophones)
        result = self.detector.predict_single(feat_vec)
        result['feature_vector'] = feat_vec

        if verbose:
            self._print_result(result)

        return result

    def detect_batch(self, mesh_filepaths, positions, rotation_degs=None,
                     mesh_scale=1.0, intersection_mode='aabb', config=None):
        """
        Run detection on a batch of objects.

        Args:
            mesh_filepaths:  List of paths to mesh files
            positions:       List of (x, y, z) positions
            rotation_degs:   List of (roll, pitch, yaw) or None (zeros)
            mesh_scale:      Common scale for all meshes
            intersection_mode: 'aabb' or 'mesh'
            config:          DatasetConfig override

        Returns:
            List of result dicts (same format as simulate_and_detect)
        """
        if rotation_degs is None:
            rotation_degs = [(0.0, 0.0, 0.0)] * len(mesh_filepaths)

        results = []
        for i, (path, pos, rot) in enumerate(zip(mesh_filepaths, positions, rotation_degs)):
            print(f"\n[{i+1}/{len(mesh_filepaths)}] Detecting: {Path(path).name}")
            result = self.simulate_and_detect(
                mesh_filepath=path,
                position=pos,
                rotation_deg=rot,
                mesh_scale=mesh_scale,
                intersection_mode=intersection_mode,
                config=config,
                verbose=False,
            )
            self._print_result(result)
            results.append(result)

        return results

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _run_simulation(self, mesh_data, position, rotation_deg,
                        intersection_mode, cfg, verbose):
        """Build and run one acoustic simulation, return populated Hydrophone list."""
        water = Water(
            size=cfg.water_size,
            temperature_surface=cfg.temperature_surface,
            temperature_bottom=cfg.temperature_bottom,
            salinity=cfg.salinity,
        )
        sim = Simulation(water)

        obj = ImportedObject3D(
            mesh_data=mesh_data,
            position=position,
            rotation_deg=rotation_deg,
            intersection_mode=intersection_mode,
            reflection_coefficient=0.85,
        )
        sim.add_object(obj)

        for pos in cfg.emitter_positions:
            em = WaveEmitter(
                position=pos,
                frequency_khz=cfg.emitter_frequency_khz,
                beam_pattern=cfg.emitter_beam,
                power_db=cfg.emitter_power_db,
            )
            sim.add_emitter(em)

        hydrophones = []
        for pos in cfg.hydrophone_positions:
            h = Hydrophone(position=pos)
            sim.add_hydrophone(h)
            hydrophones.append(h)

        if verbose:
            print(f"  Running simulation ({cfg.num_rays} rays/emitter)...")
        sim.run(num_rays_per_emitter=cfg.num_rays, verbose=False)
        if verbose:
            total_arrivals = sum(len(h.arrivals) for h in hydrophones)
            print(f"  Simulation complete. Total arrivals: {total_arrivals}")

        return hydrophones

    def _print_result(self, result):
        """Print formatted detection result."""
        print(f"\n  ┌─ Detection Result ({self.backend_name}) ─────────────────")
        print(f"  │  Class:  {result['class_name']}  (id={result['class_id']})")
        proba = result.get('class_proba', {})
        for cname, p in sorted(proba.items(), key=lambda x: -x[1]):
            bar = '█' * int(p * 20)
            print(f"  │    {cname:20s}: {p*100:5.1f}%  {bar}")
        print(f"  │  Position: x={result.get('x',0):.2f}  "
              f"y={result.get('y',0):.2f}  z={result.get('z',0):.2f} m")
        print(f"  │  Rotation: roll={result.get('roll_deg',0):.1f}°  "
              f"pitch={result.get('pitch_deg',0):.1f}°  "
              f"yaw={result.get('yaw_deg',0):.1f}°")
        print(f"  └────────────────────────────────────────────────────")


# ════════════════════════════════════════════════════════════════════════════
#  Standalone quick-start example
# ════════════════════════════════════════════════════════════════════════════

def quick_start_demo():
    """
    End-to-end example: generate → train → detect (no real mesh needed).

    Simulates two 'synthetic' object classes by using different reflection
    coefficients, without importing actual mesh files.
    """
    print("="*65)
    print("ACOUSTIC OBJECT DETECTOR — QUICK START DEMO (no mesh files)")
    print("="*65)

    import os
    import tempfile

    # Try to create a trivial STL box using trimesh for demo purposes
    try:
        import trimesh
        box1 = trimesh.creation.box([4, 4, 4])
        box2 = trimesh.creation.box([2, 8, 2])
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=2.0)

        tmpdir = tempfile.mkdtemp()
        paths = {
            'box':    os.path.join(tmpdir, 'box.stl'),
            'slab':   os.path.join(tmpdir, 'slab.stl'),
            'sphere': os.path.join(tmpdir, 'sphere.stl'),
        }
        box1.export(paths['box'])
        box2.export(paths['slab'])
        sphere.export(paths['sphere'])
    except Exception as e:
        print(f"Could not create demo meshes: {e}")
        print("Install trimesh: pip install trimesh[easy]")
        return

    from acousticsim.mesh_importer import MeshImporter
    from acousticsim.dataset_generator import DatasetGenerator, DatasetConfig
    from acousticsim.train_detector import train_sklearn, train_pytorch, evaluate

    # Load meshes
    meshes = {name: MeshImporter.load(path) for name, path in paths.items()}
    print(f"\nLoaded {len(meshes)} object classes: {list(meshes.keys())}")

    # Configure
    cfg = DatasetConfig(
        water_size=(60, 60, 30),
        num_rays=200,
        seed=7,
    )

    # Generate dataset
    gen = DatasetGenerator(meshes, config=cfg)
    dataset = gen.generate(num_samples=300, verbose=True)
    print(f"\nDataset: {dataset}")
    print(f"Class distribution: {dataset.class_distribution()}")

    # Train sklearn baseline
    sklearn_det, X_test, yc_test, yp_test = train_sklearn(dataset, n_estimators=100)

    # Train PyTorch MLP
    torch_det, _, _, _ = train_pytorch(
        dataset, epochs=60, batch_size=32, patience=10
    )

    # Evaluate both
    evaluate(X_test, yc_test, yp_test, dataset.class_names,
             sklearn_det=sklearn_det, pytorch_det=torch_det)

    # Save models
    sklearn_det.save('sklearn_detector.pkl')
    torch_det.save('pytorch_detector.pt')

    # Live detection using saved sklearn model
    print("\n\n── Live Detection Demo ─────────────────────────────────────")
    det = AcousticDetector.from_sklearn('sklearn_detector.pkl', config=cfg)
    det.simulate_and_detect(
        mesh_filepath=paths['sphere'],
        position=[30, 30, 15],
        rotation_deg=(0, 30, 45),
    )

    print("\nDemo complete.")


if __name__ == '__main__':
    quick_start_demo()
