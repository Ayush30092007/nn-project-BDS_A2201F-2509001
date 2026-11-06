"""Load, clean and split the Dry Bean dataset (used by the TRAINING container only).

Order of operations matters (this is how we avoid data leakage):
    1. load            -> read the file
    2. clean           -> drop missing rows and exact duplicate rows
    3. split           -> train / validation / test (stratified, fixed seed)
    4. scale           -> fit the scaler on the TRAINING rows only, then apply to all three
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import DATA_DIR, apply_scaler
from config import SPLIT

DOWNLOAD_HELP = (
    "No dataset found in the data/ folder.\n"
    "Download 'Dry Bean Dataset' from https://archive.ics.uci.edu/dataset/602/dry+bean+dataset\n"
    "(click Download, unzip, and copy Dry_Bean_Dataset.xlsx into the data/ folder).\n"
    "A .csv version with a 'Class' column also works."
)


def find_data_file(explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return path
    candidates = sorted(list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.csv")))
    if not candidates:
        raise FileNotFoundError(DOWNLOAD_HELP)
    bean_files = [c for c in candidates if "bean" in c.name.lower()]
    chosen = (bean_files or candidates)[0]
    if len(candidates) > 1:
        print(f"[data] Several data files found, using {chosen.name}")
    return chosen


def load_dataset(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use the .xlsx or a .csv version.")
    df.columns = [str(c).strip() for c in df.columns]
    label_col = next((c for c in df.columns if c.lower() == "class"), df.columns[-1])
    df[label_col] = df[label_col].astype(str).str.strip()
    return df, label_col


def clean(df, label_col):
    """Remove missing rows and exact duplicates. Outliers are only REPORTED, not removed
    (they are real beans, and removing rows could delete rare-class samples)."""
    report = {"rows_loaded": int(len(df))}
    features = [c for c in df.columns if c != label_col]
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="raise")
    report["missing_values"] = int(df.isna().sum().sum())
    df = df.dropna().copy()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(df))
    report["rows_after_cleaning"] = int(len(df))
    q1, q3 = df[features].quantile(0.25), df[features].quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (df[features] < q1 - 1.5 * iqr) | (df[features] > q3 + 1.5 * iqr)
    report["outliers_per_feature_iqr"] = {c: int(outlier_mask[c].sum()) for c in features}
    return df, report


def prepare_data(path=None, seed=42):
    path = find_data_file(path)
    df, label_col = load_dataset(path)
    df, clean_report = clean(df, label_col)

    feature_names = [c for c in df.columns if c != label_col]
    class_names = sorted(df[label_col].unique())
    class_to_index = {name: i for i, name in enumerate(class_names)}
    y = df[label_col].map(class_to_index).to_numpy(dtype=np.int64)
    X = df[feature_names].to_numpy(dtype=np.float64)

    idx = np.arange(len(df))
    hold_out = SPLIT["val"] + SPLIT["test"]
    train_idx, temp_idx = train_test_split(idx, test_size=hold_out, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=SPLIT["test"] / hold_out, stratify=y[temp_idx], random_state=seed
    )

    # Fit the scaler on TRAINING data only (no information from val/test leaks in).
    scaler = StandardScaler().fit(X[train_idx])
    mean, scale = scaler.mean_, scaler.scale_

    counts = np.bincount(y[train_idx], minlength=len(class_names))
    majority = int(counts.argmax())

    return {
        "path": str(path),
        "label_column": label_col,
        "feature_names": feature_names,
        "class_names": class_names,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "X_train": apply_scaler(X[train_idx], mean, scale), "y_train": y[train_idx],
        "X_val": apply_scaler(X[val_idx], mean, scale), "y_val": y[val_idx],
        "X_test": apply_scaler(X[test_idx], mean, scale), "y_test": y[test_idx],
        "X_test_raw": X[test_idx],
        "majority_class_index": majority,
        "clean_report": clean_report,
    }


def describe_split(d):
    c = d["clean_report"]
    lines = [
        f"File: {d['path']}",
        f"Rows loaded: {c['rows_loaded']} | missing values: {c['missing_values']} | "
        f"duplicates removed: {c['duplicates_removed']} | rows kept: {c['rows_after_cleaning']}",
        f"Features: {len(d['feature_names'])} | classes: {len(d['class_names'])} {d['class_names']}",
        f"Split -> train: {len(d['y_train'])} | validation: {len(d['y_val'])} | test: {len(d['y_test'])}",
    ]
    return "\n".join(lines)
