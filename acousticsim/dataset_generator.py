"""
Dataset Generator Module
Generates labeled acoustic simulation datasets for ML object detection training.

For each sample:
  1. Pick a random object class (mesh file)
  2. Randomize position and rotation within configured bounds
  3. Build and run the acoustic simulation
  4. Extract hydrophone feature vectors
  5. Record label: (class_id, x, y, z, roll, pitch, yaw)

Usage:
    from acousticsim.mesh_importer import MeshImporter
    from acousticsim.dataset_generator import DatasetGenerator

    meshes = {
        'submarine': MeshImporter.load('models/submarine.stl', scale=0.001),
        'mine':      MeshImporter.load('models/mine.obj',      scale=0.001),
    }

    gen = DatasetGenerator(meshes, water_size=(100, 100, 50))
    dataset = gen.generate(num_samples=500)
    dataset.save('acoustic_dataset.npz')
"""

import numpy as np
import time
from pathlib import Path

from acousticsim.water import Water
from acousticsim.emitter import WaveEmitter
from acousticsim.hydrophone import Hydrophone
from acousticsim.simulation import Simulation
from acousticsim.imported_object3d import ImportedObject3D
from acousticsim.feature_extractor import extract_features, feature_size


class DatasetConfig:
    """
    Configuration for dataset generation.

    Args:
        water_size:          (lx, ly, lz) simulation volume [meters]
        num_rays:            Rays per emitter per simulation run
        emitter_positions:   List of (x,y,z) emitter positions
        hydrophone_positions: List of (x,y,z) hydrophone positions
        emitter_frequency_khz: Emitter center frequency [kHz]
        emitter_power_db:    Source level [dB re 1 μPa @ 1m]
        emitter_beam:        'spherical' or 'cone'
        placement_margin:    Minimum distance from walls for object placement [m]
        rotation_ranges:     ((roll_min, roll_max), (pitch_min, pitch_max), (yaw_min, yaw_max)) [deg]
        intersection_mode:   'aabb' or 'mesh' for imported objects
        object_scale:        Scale factor applied to all loaded meshes
        seed:                Random seed (None = random)
        temperature_surface: Water surface temperature [°C]
        temperature_bottom:  Water bottom temperature [°C]
        salinity:            Water salinity [ppt]
    """

    def __init__(
        self,
        water_size=(100, 100, 50),
        num_rays=500,
        emitter_positions=None,
        hydrophone_positions=None,
        emitter_frequency_khz=85.0,
        emitter_power_db=180.0,
        emitter_beam='spherical',
        placement_margin=10.0,
        rotation_ranges=((-180, 180), (-45, 45), (-180, 180)),
        intersection_mode='aabb',
        object_scale=1.0,
        seed=42,
        temperature_surface=20.0,
        temperature_bottom=10.0,
        salinity=35.0,
    ):
        self.water_size = tuple(water_size)
        self.num_rays = num_rays
        self.emitter_frequency_khz = emitter_frequency_khz
        self.emitter_power_db = emitter_power_db
        self.emitter_beam = emitter_beam
        self.placement_margin = placement_margin
        self.rotation_ranges = rotation_ranges
        self.intersection_mode = intersection_mode
        self.object_scale = object_scale
        self.seed = seed
        self.temperature_surface = temperature_surface
        self.temperature_bottom = temperature_bottom
        self.salinity = salinity

        lx, ly, lz = water_size
        cx, cy, cz = lx / 2, ly / 2, lz / 2

        # Default emitter: center of one face
        self.emitter_positions = emitter_positions or [
            (10.0, cy, cz),
        ]

        # Default hydrophone array: line of 5 sensors on opposite side
        self.hydrophone_positions = hydrophone_positions or [
            (lx - 10, cy - 20, cz),
            (lx - 10, cy - 10, cz),
            (lx - 10, cy,      cz),
            (lx - 10, cy + 10, cz),
            (lx - 10, cy + 20, cz),
        ]


class AcousticDataset:
    """
    Container for a generated acoustic dataset.

    Attributes:
        X:              (N, F) float32 feature matrix
        y_class:        (N,)   int32 class IDs
        y_pose:         (N, 6) float32 pose labels [x, y, z, roll, pitch, yaw]
        class_names:    list of str, class_names[i] = name for class i
        feature_names:  list of str, name of each feature column
        config:         DatasetConfig used to generate this dataset
    """

    def __init__(self, X, y_class, y_pose, class_names, feature_names, config):
        self.X = X
        self.y_class = y_class
        self.y_pose = y_pose
        self.class_names = class_names
        self.feature_names = feature_names
        self.config = config

    @property
    def num_samples(self):
        return len(self.X)

    @property
    def num_classes(self):
        return len(self.class_names)

    @property
    def feature_dim(self):
        return self.X.shape[1]

    def save(self, filepath):
        """
        Save dataset to a compressed .npz file.

        Args:
            filepath: Output path (e.g., 'dataset.npz')
        """
        filepath = Path(filepath)
        np.savez_compressed(
            filepath,
            X=self.X,
            y_class=self.y_class,
            y_pose=self.y_pose,
            class_names=np.array(self.class_names, dtype=object),
            feature_names=np.array(self.feature_names, dtype=object),
        )
        print(f"Dataset saved: {filepath}  ({self.num_samples} samples, {self.feature_dim} features)")

    @staticmethod
    def load(filepath):
        """
        Load dataset from a .npz file.

        Args:
            filepath: Path to .npz file

        Returns:
            AcousticDataset (config will be None)
        """
        data = np.load(filepath, allow_pickle=True)
        return AcousticDataset(
            X=data['X'],
            y_class=data['y_class'],
            y_pose=data['y_pose'],
            class_names=list(data['class_names']),
            feature_names=list(data['feature_names']),
            config=None,
        )

    def class_distribution(self):
        """Return dict of {class_name: count}."""
        dist = {}
        for i, name in enumerate(self.class_names):
            dist[name] = int(np.sum(self.y_class == i))
        return dist

    def __repr__(self):
        return (
            f"AcousticDataset(samples={self.num_samples}, "
            f"features={self.feature_dim}, classes={self.class_names})"
        )


class DatasetGenerator:
    """
    Generates labeled acoustic simulation datasets using imported 3D meshes.

    Args:
        mesh_dict: dict mapping class_name (str) → MeshData
                   e.g., {'submarine': mesh_data_1, 'mine': mesh_data_2}
        config:    DatasetConfig (or None to use defaults)
        water_size: Shortcut if config=None (overrides default water size)
    """

    def __init__(self, mesh_dict, config=None, water_size=None):
        if not mesh_dict:
            raise ValueError("mesh_dict must contain at least one class")

        self.mesh_dict = mesh_dict
        self.class_names = sorted(mesh_dict.keys())
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}

        if config is None:
            config = DatasetConfig(water_size=water_size or (100, 100, 50))
        self.config = config

        self._rng = np.random.default_rng(config.seed)

    def generate(self, num_samples=1000, verbose=True):
        """
        Generate a labeled dataset by running many simulations.

        Args:
            num_samples: Total number of simulation runs
            verbose:     Print progress

        Returns:
            AcousticDataset
        """
        cfg = self.config
        n_hydro = len(cfg.hydrophone_positions)
        f_dim = feature_size(n_hydro)

        X = np.zeros((num_samples, f_dim), dtype=np.float32)
        y_class = np.zeros(num_samples, dtype=np.int32)
        y_pose  = np.zeros((num_samples, 6), dtype=np.float32)
        feature_names = None

        if verbose:
            print(f"\nGenerating {num_samples} acoustic simulation samples")
            print(f"  Classes: {self.class_names}")
            print(f"  Water: {cfg.water_size} m")
            print(f"  Emitters: {len(cfg.emitter_positions)}, "
                  f"Hydrophones: {n_hydro}")
            print(f"  Rays/run: {cfg.num_rays}")
            print(f"  Intersection: {cfg.intersection_mode}")
            print()

        t_start = time.time()

        for i in range(num_samples):
            # --- Pick random class ---
            class_name = self.class_names[
                int(self._rng.integers(0, len(self.class_names)))
            ]
            class_id = self.class_to_id[class_name]
            mesh_data = self.mesh_dict[class_name]

            # --- Random placement ---
            position = self._random_position(mesh_data)
            rotation = self._random_rotation()

            # --- Build simulation ---
            feats, feat_names, _ = self._run_simulation(mesh_data, position, rotation)

            X[i] = feats
            y_class[i] = class_id
            y_pose[i] = np.concatenate([position, rotation])

            if feature_names is None:
                feature_names = feat_names

            if verbose and (i + 1) % max(1, num_samples // 10) == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                remaining = (num_samples - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:5d}/{num_samples}] "
                      f"{elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining | "
                      f"last: class='{class_name}' pos={position.round(1)}")

        elapsed = time.time() - t_start
        if verbose:
            print(f"\nDone. Generated {num_samples} samples in {elapsed:.1f}s")

        return AcousticDataset(
            X=X,
            y_class=y_class,
            y_pose=y_pose,
            class_names=self.class_names,
            feature_names=feature_names or [],
            config=self.config,
        )

    def generate_single(self, class_name, position, rotation_deg=(0, 0, 0)):
        """
        Run a single labeled simulation with a specific object pose.

        Args:
            class_name:    One of the registered class names
            position:      (x, y, z) object position [meters]
            rotation_deg:  (roll, pitch, yaw) [degrees]

        Returns:
            (feature_vector, label_dict, hydrophones)
        """
        if class_name not in self.mesh_dict:
            raise KeyError(f"Unknown class '{class_name}'. Available: {self.class_names}")

        mesh_data = self.mesh_dict[class_name]
        position  = np.array(position, dtype=float)
        rotation  = np.array(rotation_deg, dtype=float)

        feats, feat_names, hydrophones = self._run_simulation(mesh_data, position, rotation)

        label = {
            'class_id':   self.class_to_id[class_name],
            'class_name': class_name,
            'x':          float(position[0]),
            'y':          float(position[1]),
            'z':          float(position[2]),
            'roll_deg':   float(rotation[0]),
            'pitch_deg':  float(rotation[1]),
            'yaw_deg':    float(rotation[2]),
        }

        return feats, label, hydrophones

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_position(self, mesh_data):
        """Sample a random placement position that keeps the object inside the water volume."""
        cfg = self.config
        lx, ly, lz = cfg.water_size
        m = cfg.placement_margin
        half = (mesh_data.extents * cfg.object_scale) / 2.0

        lo = np.array([m + half[0], m + half[1], m + half[2]])
        hi = np.array([lx - m - half[0], ly - m - half[1], lz - m - half[2]])

        # Clamp to ensure lo < hi
        lo = np.minimum(lo, hi - 1.0)

        return self._rng.uniform(lo, hi).astype(np.float32)

    def _random_rotation(self):
        """Sample random rotation within configured rotation_ranges."""
        (r0, r1), (p0, p1), (y0, y1) = self.config.rotation_ranges
        roll  = float(self._rng.uniform(r0, r1))
        pitch = float(self._rng.uniform(p0, p1))
        yaw   = float(self._rng.uniform(y0, y1))
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _run_simulation(self, mesh_data, position, rotation):
        """
        Build and run one simulation, returning extracted features.

        Returns:
            (feature_vector, feature_names, list[Hydrophone])
        """
        cfg = self.config

        water = Water(
            size=cfg.water_size,
            temperature_surface=cfg.temperature_surface,
            temperature_bottom=cfg.temperature_bottom,
            salinity=cfg.salinity,
        )

        sim = Simulation(water)

        # Insert imported object
        obj = ImportedObject3D(
            mesh_data=mesh_data,
            position=position,
            rotation_deg=rotation,
            scale=cfg.object_scale,
            intersection_mode=cfg.intersection_mode,
            reflection_coefficient=0.85,
        )
        sim.add_object(obj)

        # Add emitters
        for pos in cfg.emitter_positions:
            em = WaveEmitter(
                position=pos,
                frequency_khz=cfg.emitter_frequency_khz,
                beam_pattern=cfg.emitter_beam,
                power_db=cfg.emitter_power_db,
            )
            sim.add_emitter(em)

        # Add hydrophones
        hydrophones = []
        for pos in cfg.hydrophone_positions:
            h = Hydrophone(position=pos)
            sim.add_hydrophone(h)
            hydrophones.append(h)

        # Run
        sim.run(num_rays_per_emitter=cfg.num_rays, verbose=False)

        # Extract features
        feats, feat_names = extract_features(hydrophones)

        return feats, feat_names, hydrophones
