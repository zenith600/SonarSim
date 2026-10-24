"""
Acoustic Object Detector — Training Pipeline
Trains two types of models from acoustic simulation datasets:

    1. sklearn baseline:
        - RandomForestClassifier      → object class
        - MultiOutputRegressor+RF     → pose (x, y, z, roll, pitch, yaw)

    2. PyTorch multi-task MLP:
        - Shared backbone             → class head (CrossEntropy)
                                     → pose head  (MSE)

Usage:
    from acousticsim.dataset_generator import AcousticDataset
    from acousticsim.train_detector import train_sklearn, train_pytorch, evaluate

    dataset = AcousticDataset.load('acoustic_dataset.npz')
    sklearn_models = train_sklearn(dataset)
    pytorch_model  = train_pytorch(dataset)
    evaluate(dataset, sklearn_models=sklearn_models, pytorch_model=pytorch_model)
"""

import numpy as np
from pathlib import Path
import pickle
import warnings

# ── sklearn ──────────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not found. Install: pip install scikit-learn")

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not found. Install: pip install torch")

from acousticsim.feature_extractor import normalize_features


# ════════════════════════════════════════════════════════════════════════════
#  sklearn baseline
# ════════════════════════════════════════════════════════════════════════════

class SklearnDetector:
    """
    Baseline detector using Random Forests.

    Attributes:
        classifier:    RandomForestClassifier (predicts class_id)
        regressor:     MultiOutputRegressor[RF] (predicts x,y,z,roll,pitch,yaw)
        scaler:        StandardScaler fitted on training features
        class_names:   list of str
    """

    POSE_LABELS = ['x', 'y', 'z', 'roll_deg', 'pitch_deg', 'yaw_deg']

    def __init__(self, n_estimators=200, n_jobs=-1):
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.classifier = None
        self.regressor = None
        self.scaler = StandardScaler()
        self.class_names = None

    def fit(self, X_train, y_class_train, y_pose_train, class_names):
        """
        Train classifier and regressor.

        Args:
            X_train:        (N, F) float32 features
            y_class_train:  (N,)   int32 class IDs
            y_pose_train:   (N, 6) float32 pose targets
            class_names:    list of str
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required. pip install scikit-learn")

        self.class_names = class_names

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        print("  Training RandomForest classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=42,
        )
        self.classifier.fit(X_scaled, y_class_train)

        print("  Training RandomForest pose regressor...")
        self.regressor = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=self.n_estimators,
                n_jobs=self.n_jobs,
                random_state=42,
            )
        )
        self.regressor.fit(X_scaled, y_pose_train)

        print("  sklearn training complete.")

    def predict(self, X):
        """
        Predict class and pose.

        Args:
            X: (N, F) feature matrix

        Returns:
            pred_class: (N,) int32 predicted class IDs
            pred_pose:  (N, 6) float32 predicted poses
        """
        X_scaled = self.scaler.transform(X)
        pred_class = self.classifier.predict(X_scaled).astype(np.int32)
        pred_pose  = self.regressor.predict(X_scaled).astype(np.float32)
        return pred_class, pred_pose

    def predict_single(self, feature_vector):
        """
        Predict one sample.

        Args:
            feature_vector: (F,) float32

        Returns:
            dict: class_id, class_name, class_proba, x, y, z, roll, pitch, yaw
        """
        X = feature_vector.reshape(1, -1)
        pred_class, pred_pose = self.predict(X)
        cid = int(pred_class[0])
        proba = self.classifier.predict_proba(self.scaler.transform(X))[0]
        result = {
            'class_id':   cid,
            'class_name': self.class_names[cid] if self.class_names else str(cid),
            'class_proba': {
                self.class_names[i]: float(proba[i])
                for i in range(len(proba))
            },
        }
        for k, v in zip(self.POSE_LABELS, pred_pose[0]):
            result[k] = float(v)
        return result

    def feature_importance(self, feature_names=None):
        """
        Return classification feature importances sorted descending.

        Returns:
            list of (feature_name, importance) tuples
        """
        if self.classifier is None:
            raise RuntimeError("Must call fit() first")
        imps = self.classifier.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(imps))]
        pairs = sorted(zip(feature_names, imps), key=lambda x: -x[1])
        return pairs

    def save(self, filepath):
        """Save to .pkl file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"SklearnDetector saved: {filepath}")

    @staticmethod
    def load(filepath):
        """Load from .pkl file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# ════════════════════════════════════════════════════════════════════════════
#  PyTorch multi-task MLP
# ════════════════════════════════════════════════════════════════════════════

class AcousticMLPDataset(Dataset):
    """PyTorch Dataset wrapping (X, y_class, y_pose)."""

    def __init__(self, X, y_class, y_pose):
        self.X       = torch.from_numpy(X).float()
        self.y_class = torch.from_numpy(y_class).long()
        self.y_pose  = torch.from_numpy(y_pose).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_pose[idx]


class AcousticMLP(nn.Module):
    """
    Multi-task MLP for acoustic object detection.

    Architecture:
        Input → Shared backbone → [Classification head | Pose regression head]

    Classification head: CE loss (object type)
    Pose head:           MSE loss (x, y, z, roll, pitch, yaw)

    Args:
        input_dim:   Feature vector length
        num_classes: Number of object classes
        hidden_dims: Tuple of hidden layer sizes for shared backbone
        dropout:     Dropout probability
    """

    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()

        # Shared backbone
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        # Pose regression head (x, y, z, roll, pitch, yaw)
        self.pose_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) batch of feature vectors

        Returns:
            logits: (B, num_classes) unnormalized class scores
            pose:   (B, 6) pose predictions [x, y, z, roll, pitch, yaw]
        """
        shared = self.backbone(x)
        logits = self.class_head(shared)
        pose   = self.pose_head(shared)
        return logits, pose


class PyTorchDetector:
    """
    Wrapper for the AcousticMLP model with training loop and inference.

    Args:
        input_dim:       Feature vector length
        num_classes:     Number of object classes
        class_names:     List of class name strings
        hidden_dims:     Backbone hidden layer sizes
        dropout:         Dropout rate
        pose_loss_weight: Weight on pose MSE vs. classification CE loss
        device:          'cuda', 'cpu', or None (auto-detect)
    """

    POSE_LABELS = ['x', 'y', 'z', 'roll_deg', 'pitch_deg', 'yaw_deg']

    def __init__(
        self,
        input_dim,
        num_classes,
        class_names,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
        pose_loss_weight=0.5,
        device=None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")

        self.class_names = class_names
        self.pose_loss_weight = pose_loss_weight
        self.feature_means = None
        self.feature_stds  = None

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = AcousticMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(self.device)

        # Loss and optimizer (configured in fit())
        self._ce_loss   = nn.CrossEntropyLoss()
        self._mse_loss  = nn.MSELoss()
        self._optimizer = None

    def fit(
        self,
        X_train,
        y_class_train,
        y_pose_train,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        val_fraction=0.1,
        patience=15,
    ):
        """
        Train the MLP with early stopping.

        Args:
            X_train:        (N, F) float32
            y_class_train:  (N,)   int32
            y_pose_train:   (N, 6) float32
            epochs:         Maximum training epochs
            batch_size:     Mini-batch size
            lr:             Adam learning rate
            val_fraction:   Fraction of training data held out for validation
            patience:       Early stopping patience (epochs without val improvement)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. pip install torch")

        # Normalize features
        X_norm, self.feature_means, self.feature_stds = normalize_features(X_train)

        # Normalize pose targets (for better gradient scaling)
        self._pose_mean = y_pose_train.mean(axis=0)
        self._pose_std  = y_pose_train.std(axis=0)
        self._pose_std[self._pose_std < 1e-8] = 1.0
        y_pose_norm = ((y_pose_train - self._pose_mean) / self._pose_std).astype(np.float32)

        full_dataset = AcousticMLPDataset(X_norm, y_class_train, y_pose_norm)
        n_val = max(1, int(len(full_dataset) * val_fraction))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, patience=patience // 2, factor=0.5, verbose=False
        )

        best_val_loss = np.inf
        best_state    = None
        wait          = 0

        print(f"\n  PyTorch MLP training on {self.device}")
        print(f"  Train: {n_train}  Val: {n_val}  Epochs: {epochs}  BS: {batch_size}")
        print()

        for epoch in range(1, epochs + 1):
            # --- Train ---
            self.model.train()
            train_losses = []
            for X_b, yc_b, yp_b in train_loader:
                X_b  = X_b.to(self.device)
                yc_b = yc_b.to(self.device)
                yp_b = yp_b.to(self.device)

                self._optimizer.zero_grad()
                logits, pose_pred = self.model(X_b)

                loss_cls  = self._ce_loss(logits, yc_b)
                loss_pose = self._mse_loss(pose_pred, yp_b)
                loss = loss_cls + self.pose_loss_weight * loss_pose

                loss.backward()
                self._optimizer.step()
                train_losses.append(loss.item())

            # --- Validate ---
            self.model.eval()
            val_losses = []
            val_correct = 0
            val_total   = 0
            with torch.no_grad():
                for X_b, yc_b, yp_b in val_loader:
                    X_b  = X_b.to(self.device)
                    yc_b = yc_b.to(self.device)
                    yp_b = yp_b.to(self.device)

                    logits, pose_pred = self.model(X_b)
                    loss_cls  = self._ce_loss(logits, yc_b)
                    loss_pose = self._mse_loss(pose_pred, yp_b)
                    val_losses.append((loss_cls + self.pose_loss_weight * loss_pose).item())

                    preds = logits.argmax(dim=1)
                    val_correct += (preds == yc_b).sum().item()
                    val_total   += len(yc_b)

            t_loss = np.mean(train_losses)
            v_loss = np.mean(val_losses)
            v_acc  = 100.0 * val_correct / max(1, val_total)

            scheduler.step(v_loss)

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{epochs}  "
                      f"train_loss={t_loss:.4f}  "
                      f"val_loss={v_loss:.4f}  "
                      f"val_acc={v_acc:.1f}%")

            # Early stopping
            if v_loss < best_val_loss - 1e-6:
                best_val_loss = v_loss
                best_state    = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        print(f"\n  Training complete. Best val_loss={best_val_loss:.4f}")

    @torch.no_grad()
    def predict(self, X):
        """
        Predict class and pose for a batch.

        Args:
            X: (N, F) float32 feature matrix (raw, unnormalized)

        Returns:
            pred_class: (N,) int32 class IDs
            pred_pose:  (N, 6) float32 denormalized pose predictions
        """
        self.model.eval()

        # Normalize
        X_norm, _, _ = normalize_features(X, self.feature_means, self.feature_stds)
        X_tensor = torch.from_numpy(X_norm).float().to(self.device)

        logits, pose_pred = self.model(X_tensor)

        pred_class = logits.argmax(dim=1).cpu().numpy().astype(np.int32)
        pose_norm  = pose_pred.cpu().numpy().astype(np.float32)

        # Denormalize pose
        pred_pose = pose_norm * self._pose_std + self._pose_mean

        return pred_class, pred_pose

    @torch.no_grad()
    def predict_single(self, feature_vector):
        """
        Predict for one sample.

        Args:
            feature_vector: (F,) float32

        Returns:
            dict with class_id, class_name, class_proba, x, y, z, roll, pitch, yaw
        """
        self.model.eval()

        X = feature_vector.reshape(1, -1).astype(np.float32)
        X_norm, _, _ = normalize_features(X, self.feature_means, self.feature_stds)
        X_tensor = torch.from_numpy(X_norm).float().to(self.device)

        logits, pose_pred = self.model(X_tensor)
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        cid   = int(proba.argmax())

        # Denormalize pose
        pose = (pose_pred.cpu().numpy()[0] * self._pose_std + self._pose_mean).tolist()

        result = {
            'class_id':   cid,
            'class_name': self.class_names[cid] if self.class_names else str(cid),
            'class_proba': {
                self.class_names[i]: float(proba[i])
                for i in range(len(proba))
            },
        }
        for k, v in zip(self.POSE_LABELS, pose):
            result[k] = float(v)

        return result

    def save(self, filepath):
        """Save model weights and normalization stats to .pt file."""
        state = {
            'model_state': self.model.state_dict(),
            'class_names':    self.class_names,
            'feature_means':  self.feature_means,
            'feature_stds':   self.feature_stds,
            'pose_mean':      self._pose_mean,
            'pose_std':       self._pose_std,
            'pose_loss_weight': self.pose_loss_weight,
        }
        torch.save(state, filepath)
        print(f"PyTorchDetector saved: {filepath}")

    @staticmethod
    def load(filepath, input_dim, num_classes, hidden_dims=(256, 128, 64), dropout=0.3):
        """
        Load model from .pt file.

        Args:
            filepath:    Path to .pt checkpoint
            input_dim:   Must match the saved model
            num_classes: Must match the saved model

        Returns:
            PyTorchDetector with restored weights
        """
        state = torch.load(filepath, map_location='cpu')
        det = PyTorchDetector(
            input_dim=input_dim,
            num_classes=num_classes,
            class_names=state['class_names'],
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        det.model.load_state_dict(state['model_state'])
        det.feature_means     = state['feature_means']
        det.feature_stds      = state['feature_stds']
        det._pose_mean        = state['pose_mean']
        det._pose_std         = state['pose_std']
        det.pose_loss_weight  = state['pose_loss_weight']
        det.model.eval()
        return det


# ════════════════════════════════════════════════════════════════════════════
#  High-level training functions
# ════════════════════════════════════════════════════════════════════════════

def train_sklearn(dataset, test_size=0.2, n_estimators=200, n_jobs=-1):
    """
    Train a sklearn baseline detector from an AcousticDataset.

    Args:
        dataset:       AcousticDataset
        test_size:     Fraction of data reserved for testing
        n_estimators:  Number of trees in random forests
        n_jobs:        Parallel jobs (-1 = all cores)

    Returns:
        (SklearnDetector, X_test, y_class_test, y_pose_test)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required. pip install scikit-learn")

    print("\n" + "="*60)
    print("TRAINING sklearn BASELINE DETECTOR")
    print("="*60)

    X_train, X_test, yc_train, yc_test, yp_train, yp_test = train_test_split(
        dataset.X, dataset.y_class, dataset.y_pose,
        test_size=test_size, random_state=42, stratify=dataset.y_class,
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    det = SklearnDetector(n_estimators=n_estimators, n_jobs=n_jobs)
    det.fit(X_train, yc_train, yp_train, dataset.class_names)

    return det, X_test, yc_test, yp_test


def train_pytorch(
    dataset,
    test_size=0.2,
    hidden_dims=(256, 128, 64),
    dropout=0.3,
    epochs=150,
    batch_size=64,
    lr=1e-3,
    patience=20,
    pose_loss_weight=0.5,
    device=None,
):
    """
    Train the PyTorch MLP detector from an AcousticDataset.

    Args:
        dataset:           AcousticDataset
        test_size:         Test split fraction
        hidden_dims:       Backbone layer widths
        dropout:           Dropout rate
        epochs:            Max training epochs
        batch_size:        Mini-batch size
        lr:                Adam learning rate
        patience:          Early stopping patience
        pose_loss_weight:  Weight of pose MSE vs. classification CE
        device:            'cuda' / 'cpu' / None (auto)

    Returns:
        (PyTorchDetector, X_test, y_class_test, y_pose_test)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. pip install torch")
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for train-test split. pip install scikit-learn")

    print("\n" + "="*60)
    print("TRAINING PyTorch MULTI-TASK MLP DETECTOR")
    print("="*60)

    X_train, X_test, yc_train, yc_test, yp_train, yp_test = train_test_split(
        dataset.X, dataset.y_class, dataset.y_pose,
        test_size=test_size, random_state=42, stratify=dataset.y_class,
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    det = PyTorchDetector(
        input_dim=dataset.feature_dim,
        num_classes=dataset.num_classes,
        class_names=dataset.class_names,
        hidden_dims=hidden_dims,
        dropout=dropout,
        pose_loss_weight=pose_loss_weight,
        device=device,
    )
    det.fit(
        X_train, yc_train, yp_train,
        epochs=epochs, batch_size=batch_size,
        lr=lr, patience=patience,
    )

    return det, X_test, yc_test, yp_test


# ════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ════════════════════════════════════════════════════════════════════════════

POSE_LABELS = ['x', 'y', 'z', 'roll_deg', 'pitch_deg', 'yaw_deg']


def evaluate(X_test, y_class_test, y_pose_test, class_names,
             sklearn_det=None, pytorch_det=None):
    """
    Evaluate one or both detectors on a held-out test set.

    Args:
        X_test:         (N, F) float32 feature matrix
        y_class_test:   (N,)   int32 ground truth class IDs
        y_pose_test:    (N, 6) float32 ground truth poses
        class_names:    list of str
        sklearn_det:    SklearnDetector (or None to skip)
        pytorch_det:    PyTorchDetector (or None to skip)
    """
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)

    detectors = []
    if sklearn_det is not None:
        detectors.append(("sklearn (RF baseline)", sklearn_det))
    if pytorch_det is not None:
        detectors.append(("PyTorch MLP", pytorch_det))

    for name, det in detectors:
        print(f"\n── {name} ──────────────────────────────────")
        pred_class, pred_pose = det.predict(X_test)

        # Classification metrics
        if SKLEARN_AVAILABLE:
            print(classification_report(
                y_class_test, pred_class,
                target_names=class_names,
                zero_division=0,
            ))

        # Pose metrics
        mae_total = np.mean(np.abs(pred_pose - y_pose_test))
        print(f"  Pose MAE (all 6 DoF): {mae_total:.3f}")
        for i, label in enumerate(POSE_LABELS):
            mae = float(np.mean(np.abs(pred_pose[:, i] - y_pose_test[:, i])))
            print(f"    {label:12s}: MAE = {mae:.3f}")
