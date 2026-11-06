"""Test container entry point: load the saved model and evaluate it on the saved test set.

    python src/test.py
Needs only torch + numpy (no pandas / scikit-learn), and nothing from the training environment
except the files in the models/ folder.
"""
import argparse
import csv
import json
import platform
import socket
from pathlib import Path

import numpy as np
import torch

import common


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved model on the held-out test set")
    parser.add_argument("--models-dir", default=str(common.MODELS_DIR))
    parser.add_argument("--show-errors", type=int, default=10, help="how many wrong predictions to print")
    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    model, model_cfg, prep = common.load_artifacts(models_dir)
    X_raw, y_true = common.load_test_set(models_dir, prep)
    classes = prep["class_names"]
    n_classes = len(classes)

    print("=== TEST ENVIRONMENT (screenshot this) ===")
    print(f"Host: {socket.gethostname()} | OS: {platform.platform()} | CPU: {platform.machine()}")
    print(f"Python {platform.python_version()} | torch {torch.__version__} | numpy {np.__version__}")
    print(f"Model: hidden layers {model_cfg['hidden']}, dropout {model_cfg['dropout']}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Test samples: {len(y_true)}\n")

    # Same scaling (training mean/std) + same prediction + same metric functions as in training.
    X = common.apply_scaler(X_raw, prep["mean"], prep["scale"])
    probs = common.predict_proba(model, X)
    y_pred = probs.argmax(axis=1)
    m = common.classification_metrics(y_true, y_pred, n_classes)

    print("=== TEST RESULTS ===")
    print(f"Accuracy        : {m['accuracy']:.4f}")
    print(f"Macro precision : {m['macro_precision']:.4f}")
    print(f"Macro recall    : {m['macro_recall']:.4f}")
    print(f"Macro F1        : {m['macro_f1']:.4f}   (weighted F1 {m['weighted_f1']:.4f})\n")
    print(f"{'Class':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for i, name in enumerate(classes):
        print(f"{name:<12}{m['per_class_precision'][i]:>10.4f}{m['per_class_recall'][i]:>10.4f}"
              f"{m['per_class_f1'][i]:>10.4f}{m['support'][i]:>10d}")

    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    print(f"{'':<12}" + "".join(f"{c[:8]:>9}" for c in classes))
    for i, row in enumerate(m["confusion_matrix"]):
        print(f"{classes[i]:<12}" + "".join(f"{v:>9d}" for v in row))

    # Simple baseline: always predict the most common training class.
    majority = prep["majority_class_index"]
    base = common.classification_metrics(y_true, np.full_like(y_true, majority), n_classes)
    print(f"\nBaseline (always '{prep['majority_class_name']}'): accuracy {base['accuracy']:.4f}, "
          f"macro F1 {base['macro_f1']:.4f}")

    print_errors(X_raw, y_true, y_pred, probs, classes, prep["feature_names"], args.show_errors, models_dir, m)
    check_reproducibility(models_dir, m)


def print_errors(X_raw, y_true, y_pred, probs, classes, feature_names, how_many, models_dir, metrics):
    wrong = np.where(y_pred != y_true)[0]
    print(f"\n=== WRONG PREDICTIONS: {len(wrong)} of {len(y_true)} ===")
    if len(wrong) == 0:
        return
    confidence = probs[np.arange(len(y_pred)), y_pred]
    order = wrong[np.argsort(-confidence[wrong])]  # most confident mistakes first

    cm = np.array(metrics["confusion_matrix"])
    pairs = [(cm[i, j], i, j) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
    print("Most common confusions (true -> predicted : count):")
    for count, i, j in sorted(pairs, reverse=True)[:5]:
        print(f"  {classes[i]} -> {classes[j]} : {count}")

    print(f"\nFirst {min(how_many, len(order))} mistakes, most confident first:")
    print(f"{'Row':>5}  {'True':<10}{'Predicted':<11}{'Conf.':>7}  {'2nd choice':<11}")
    for k in order[:how_many]:
        second = np.argsort(-probs[k])[1]
        print(f"{k:>5}  {classes[y_true[k]]:<10}{classes[y_pred[k]]:<11}{confidence[k]:>7.2f}  "
              f"{classes[second]:<11}")

    path = Path(models_dir) / "test_errors.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "true", "predicted", "confidence"] + feature_names)
        for k in order:
            writer.writerow([k, classes[y_true[k]], classes[y_pred[k]], f"{confidence[k]:.4f}"]
                            + [repr(float(v)) for v in X_raw[k]])
    print(f"All {len(order)} mistakes (with feature values) saved to {path}")


def check_reproducibility(models_dir, metrics):
    report_path = Path(models_dir) / "train_report.json"
    print("\n=== REPRODUCIBILITY CHECK ===")
    if not report_path.exists():
        print("train_report.json not found - cannot compare with the numbers reported during training.")
        return
    with open(report_path) as f:
        trained = json.load(f)["test"]
    same_cm = trained["confusion_matrix"] == metrics["confusion_matrix"]
    same_acc = abs(trained["accuracy"] - metrics["accuracy"]) < 1e-12
    same_f1 = abs(trained["macro_f1"] - metrics["macro_f1"]) < 1e-9
    print(f"Accuracy  training {trained['accuracy']:.6f} | now {metrics['accuracy']:.6f} | match: {same_acc}")
    print(f"Macro F1  training {trained['macro_f1']:.6f} | now {metrics['macro_f1']:.6f} | match: {same_f1}")
    print(f"Confusion matrix identical: {same_cm}")
    print("RESULT:", "PASS - test metrics reproduced exactly" if (same_cm and same_acc and same_f1)
          else "DIFFERENCE FOUND - see numbers above")


if __name__ == "__main__":
    main()
