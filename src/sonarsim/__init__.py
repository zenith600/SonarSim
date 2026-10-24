"""SonarSim: physics-based generation of labelled active sonar datasets."""
from sonarsim._version import __version__
from sonarsim.autoplacer import AutoPlacer, PlacementBounds, Scenario
from sonarsim.dataset import generate_dataset, process_ping
from sonarsim.environment import SEDIMENTS, Environment, Sediment
from sonarsim.labels import make_label, save_ping
from sonarsim.processing import beamform, matched_filter, range_profile, sector_image
from sonarsim.propagation import Path, eigenpaths
from sonarsim.simulator import Ping, TargetArrival, simulate_ping
from sonarsim.sonar import PRESETS, A_HF_imaging, B_LF_search, SonarConfig, preset
from sonarsim.target import Mesh, Pose, Target
from sonarsim.waveforms import Waveform

__all__ = [
    "__version__",
    "AutoPlacer", "PlacementBounds", "Scenario",
    "generate_dataset", "process_ping",
    "Environment", "Sediment", "SEDIMENTS",
    "make_label", "save_ping",
    "beamform", "matched_filter", "range_profile", "sector_image",
    "Path", "eigenpaths",
    "Ping", "TargetArrival", "simulate_ping",
    "SonarConfig", "A_HF_imaging", "B_LF_search", "PRESETS", "preset",
    "Mesh", "Pose", "Target",
    "Waveform",
]
