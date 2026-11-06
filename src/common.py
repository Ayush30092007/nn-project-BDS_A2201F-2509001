"""Helpers shared by the TRAINING and the TESTING container.

Only numpy + torch are needed here (no pandas / scikit-learn), so the test image stays small.
Using the SAME functions for scaling, prediction and metrics in both containers is what
makes the test numbers match the training numbers exactly.
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from model import BeanMLP

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def set_seed(seed):
    """Fix every source of randomness so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def apply_scaler(X, mean, scale):
    """Standardisation: (x - mean) / std, using the mean/std learned from the TRAINING set only."""
    return ((np.asarray(X, dtype=np.float64) - np.asarray(mean)) / np.asarray(scale)).astype(np.float32)


@torch.no_grad()
def predict_proba(model, X_scaled):
    """Softmax probabilities, shape (n_samples, n_classes)."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(X_scaled, dtype=np.float32))
    return torch.softmax(model(x), dim=1).numpy()


# ----------------------------------------------------------------------------- metrics
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm  # rows = true class, columns = predicted class


def _safe_divide(a, b):
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(a, b, out=out, where=b > 0)
    return out


def classification_metrics(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    predicted_positive = cm.sum(axis=0).astype(np.float64)
    actual_positive = cm.sum(axis=1).astype(np.float64)
    precision = _safe_divide(tp, predicted_positive)
    recall = _safe_divide(tp, actual_positive)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    weights = actual_positive / actual_positive.sum()
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * weights).sum()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": actual_positive.astype(int).tolist(),
        "confusion_matrix": cm.tolist(),
    }


# ----------------------------------------------------------------------------- loading
def load_artifacts(models_dir=MODELS_DIR):
    """Load the trained model + preprocessing info saved by train.py --final."""
    models_dir = Path(models_dir)
    needed = ["model.pt", "model_config.json", "preprocessing.json"]
    missing = [name for name in needed if not (models_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {models_dir}. Run the training container first "
            "(python src/train.py --final) and make sure the models folder is mounted/copied."
        )
    with open(models_dir / "model_config.json") as f:
        model_cfg = json.load(f)
    with open(models_dir / "preprocessing.json") as f:
        prep = json.load(f)
    model = BeanMLP(**model_cfg)
    state = torch.load(models_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, model_cfg, prep


def load_test_set(models_dir, prep):
    """Read the held-out test rows (raw, unscaled) saved by the training container."""
    path = Path(models_dir) / "test_set.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. It is written by 'python src/train.py --final'.")
    label_col = prep["label_column"]
    class_to_index = {name: i for i, name in enumerate(prep["class_names"])}
    rows, labels = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for record in reader:
            rows.append([float(record[name]) for name in prep["feature_names"]])
            labels.append(class_to_index[record[label_col]])
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int64)
