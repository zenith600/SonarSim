"""
acousticsim — Underwater Acoustic Ray Tracing & Object Detection
Public API
"""

from acousticsim.water import Water
from acousticsim.ray import Ray
from acousticsim.emitter import WaveEmitter
from acousticsim.hydrophone import Hydrophone
from acousticsim.object3d import Object3D, Surface, Seabed
from acousticsim.simulation import Simulation
from acousticsim.mesh_importer import MeshImporter, MeshData
from acousticsim.imported_object3d import ImportedObject3D
from acousticsim.feature_extractor import extract_features, feature_size, normalize_features
from acousticsim.dataset_generator import DatasetGenerator, DatasetConfig, AcousticDataset
from acousticsim.train_detector import (
    SklearnDetector,
    PyTorchDetector,
    AcousticMLP,
    train_sklearn,
    train_pytorch,
    evaluate,
)
from acousticsim.inference import AcousticDetector
from acousticsim.noise import NoiseModel, add_noise_to_signal

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = [
    # Core simulation
    "Water",
    "Ray",
    "WaveEmitter",
    "Hydrophone",
    "Object3D", "Surface", "Seabed",
    "Simulation",
    # Mesh import
    "MeshImporter", "MeshData",
    "ImportedObject3D",
    # ML pipeline
    "extract_features", "feature_size", "normalize_features",
    "DatasetGenerator", "DatasetConfig", "AcousticDataset",
    "SklearnDetector", "PyTorchDetector", "AcousticMLP",
    "train_sklearn", "train_pytorch", "evaluate",
    "AcousticDetector",
    # Noise
    "NoiseModel",
    "add_noise_to_signal",
]
