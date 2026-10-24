"""
Feature Extractor Module
Converts raw hydrophone arrival data from simulations into fixed-size
feature vectors suitable for machine learning.

Feature vector layout (per run):
    For each hydrophone h:
        - n_arrivals          : total ray arrivals
        - first_time_ms       : earliest arrival time [ms]
        - last_time_ms        : latest arrival time [ms]
        - delay_spread_ms     : std of arrival times [ms]
        - max_amplitude       : peak amplitude received
        - total_energy        : sum of amplitude²
        - amplitude_mean      : mean amplitude
        - amplitude_std       : std of amplitudes

    Cross-hydrophone (all pairs i < j):
        - tdoa_ij_ms          : first_time[i] - first_time[j]  [ms]

Total features = H * 8 + C(H, 2)
    e.g., H=5: 5*8 + 10 = 50 features
"""

import numpy as np
from itertools import combinations


def extract_features(hydrophones, sound_speed=1500.0):
    """
    Extract a fixed-size feature vector from a list of Hydrophone objects
    after a simulation run.

    Args:
        hydrophones: list of Hydrophone objects (must have .arrivals populated)
        sound_speed: average sound speed [m/s] (used only for documentation)

    Returns:
        feature_vector: 1D numpy float32 array
        feature_names:  list of strings (one per feature)
    """
    H = len(hydrophones)
    per_hydro_features = []
    per_hydro_names = []
    first_times = []

    for h_idx, hydro in enumerate(hydrophones):
        arrivals = hydro.arrivals

        if len(arrivals) == 0:
            # No arrivals: zero-fill
            stats = np.zeros(8, dtype=np.float32)
            first_times.append(0.0)
        else:
            times = np.array([a['time'] for a in arrivals], dtype=np.float64) * 1000.0  # → ms
            amps  = np.array([a['amplitude'] for a in arrivals], dtype=np.float64)

            # Guard against negatives/zeros
            amps = np.clip(amps, 0.0, None)

            first_t = times.min()
            last_t  = times.max()
            first_times.append(first_t)

            stats = np.array([
                float(len(arrivals)),           # n_arrivals
                first_t,                        # first_time_ms
                last_t,                         # last_time_ms
                float(np.std(times)),           # delay_spread_ms
                float(amps.max()),              # max_amplitude
                float(np.sum(amps ** 2)),       # total_energy
                float(np.mean(amps)),           # amplitude_mean
                float(np.std(amps)),            # amplitude_std
            ], dtype=np.float32)

        prefix = f"h{h_idx}_"
        names = [
            prefix + "n_arrivals",
            prefix + "first_time_ms",
            prefix + "last_time_ms",
            prefix + "delay_spread_ms",
            prefix + "max_amplitude",
            prefix + "total_energy",
            prefix + "amplitude_mean",
            prefix + "amplitude_std",
        ]

        per_hydro_features.append(stats)
        per_hydro_names.extend(names)

    # --- Cross-hydrophone TDOA ---
    tdoa_features = []
    tdoa_names = []
    for i, j in combinations(range(H), 2):
        tdoa = float(first_times[i] - first_times[j])
        tdoa_features.append(tdoa)
        tdoa_names.append(f"tdoa_h{i}_h{j}_ms")

    feature_vector = np.concatenate(
        per_hydro_features + [np.array(tdoa_features, dtype=np.float32)]
    ).astype(np.float32)

    feature_names = per_hydro_names + tdoa_names

    return feature_vector, feature_names


def feature_size(num_hydrophones):
    """
    Return the length of the feature vector for a given number of hydrophones.

    Args:
        num_hydrophones: H

    Returns:
        int: H * 8 + C(H, 2)
    """
    H = num_hydrophones
    return H * 8 + H * (H - 1) // 2


def normalize_features(X, means=None, stds=None):
    """
    Z-score normalize a feature matrix.

    Args:
        X:     (N, F) float32 array
        means: (F,) precomputed means (if None, computed from X)
        stds:  (F,) precomputed stds  (if None, computed from X)

    Returns:
        X_norm: (N, F) normalized array
        means:  (F,) means used
        stds:   (F,) stds used
    """
    if means is None:
        means = X.mean(axis=0)
    if stds is None:
        stds = X.std(axis=0)
        stds[stds < 1e-12] = 1.0  # Avoid division by zero

    X_norm = (X - means) / stds
    return X_norm.astype(np.float32), means, stds
