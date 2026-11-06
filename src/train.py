"""Train the dry bean MLP.

Normal run  (experiments):  python src/train.py --name my_run         -> uses train + validation ONLY
Final run   (once, at the end): python src/train.py --final           -> also evaluates the TEST set once,
                                                                         saves model + preprocessing + test rows.
"""
import argparse
import copy
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import common
import data_prep
from config import BASELINE, FINAL
from model import BeanMLP, architecture_table, count_parameters, format_architecture


# ----------------------------------------------------------------------------- training core
def make_optimizer(model, cfg):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                               weight_decay=cfg["weight_decay"])
    raise ValueError(f"Unknown optimizer {cfg['optimizer']}")


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """Loss and accuracy on a whole set, with dropout switched off (eval mode)."""
    model.eval()
    logits = model(X)
    loss = criterion(logits, y).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def run_training(cfg, d, verbose=True):
    """Train one model. Returns the model (restored to its best-validation epoch) and a results dict."""
    common.set_seed(cfg["seed"])
    X_tr, y_tr = torch.from_numpy(d["X_train"]), torch.from_numpy(d["y_train"])
    X_va, y_va = torch.from_numpy(d["X_val"]), torch.from_numpy(d["y_val"])
    n_classes = len(d["class_names"])

    model = BeanMLP(n_features=X_tr.shape[1], hidden=cfg["hidden"], n_classes=n_classes, dropout=cfg["dropout"])
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg)

    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(cfg["seed"])
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg["batch_size"], shuffle=True,
                        generator=shuffle_gen)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for xb, yb in loader:                 # one iteration = one batch = one weight update
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)   # forward pass + loss
            loss.backward()                   # backpropagation: gradients of the loss
            optimizer.step()                  # update the weights
        train_loss, train_acc = evaluate(model, X_tr, y_tr, criterion)
        val_loss, val_acc = evaluate(model, X_va, y_va, criterion)
        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc)):
            history[key].append(value)

        if val_loss < best["val_loss"]:
            best.update(val_loss=val_loss, epoch=epoch, state=copy.deepcopy(model.state_dict()))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == cfg["epochs"]):
            print(f"  epoch {epoch:3d}/{cfg['epochs']}  train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}")
        if cfg["patience"] and epochs_without_improvement >= cfg["patience"]:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best epoch {best['epoch']})")
            break

    model.load_state_dict(best["state"])      # go back to the best epoch
    val_pred = common.predict_proba(model, d["X_val"]).argmax(axis=1)
    val_metrics = common.classification_metrics(d["y_val"], val_pred, n_classes)
    best_i = best["epoch"] - 1
    results = {
        "epochs_run": len(history["val_loss"]),
        "best_epoch": best["epoch"],
        "best_val_loss": history["val_loss"][best_i],
        "best_val_acc": history["val_acc"][best_i],
        "train_loss_at_best": history["train_loss"][best_i],
        "train_acc_at_best": history["train_acc"][best_i],
        "val_macro_f1": val_metrics["macro_f1"],
        "n_params": count_parameters(model),
        "seconds": round(time.time() - start, 1),
        "history": history,
    }
    return model, results


# ----------------------------------------------------------------------------- plots
def plot_curves(history, best_epoch, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["val_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Training loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[1].plot(epochs, history["train_acc"], label="Training accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Validation accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    for ax in axes:
        ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    normalised = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Confusion matrix (test set)\ncolour = row %, number = count")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if normalised[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- final run
def guard_test_set_used_once(force):
    """Stop early if a final run already exists (so the test set is not used twice by accident)."""
    if (common.MODELS_DIR / "train_report.json").exists() and not force:
        sys.exit(
            "\nSTOP: models/train_report.json already exists, so the test set has already been used.\n"
            "The test set must only be used once. If you really changed the code and need to redo the\n"
            "final run, run again with --force.\n"
        )


def final_evaluation(model, cfg, d, results, force):
    n_classes = len(d["class_names"])
    report_path = common.MODELS_DIR / "train_report.json"
    guard_test_set_used_once(force)

    # ---- the one and only look at the test set
    test_pred = common.predict_proba(model, d["X_test"]).argmax(axis=1)
    test_metrics = common.classification_metrics(d["y_test"], test_pred, n_classes)
    majority = d["majority_class_index"]
    baseline_metrics = common.classification_metrics(d["y_test"], np.full_like(d["y_test"], majority), n_classes)

    models_dir = common.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), models_dir / "model.pt")
    model_cfg = {"n_features": len(d["feature_names"]), "hidden": list(cfg["hidden"]),
                 "n_classes": n_classes, "dropout": cfg["dropout"]}
    with open(models_dir / "model_config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)

    preprocessing = {
        "label_column": d["label_column"],
        "feature_names": d["feature_names"],
        "class_names": d["class_names"],
        "scaler": "StandardScaler (fitted on the training set only)",
        "mean": d["mean"],
        "scale": d["scale"],
        "majority_class_index": majority,
        "majority_class_name": d["class_names"][majority],
        "seed": cfg["seed"],
        "rows_after_cleaning": d["clean_report"]["rows_after_cleaning"],
    }
    with open(models_dir / "preprocessing.json", "w") as f:
        json.dump(preprocessing, f, indent=2)

    # Raw (unscaled) held-out test rows, so the test container can use exactly the same test set.
    with open(models_dir / "test_set.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(d["feature_names"] + [d["label_column"]])
        for row, label in zip(d["X_test_raw"], d["y_test"]):
            writer.writerow([repr(float(v)) for v in row] + [d["class_names"][int(label)]])

    report = {
        "config": cfg,
        "n_params": results["n_params"],
        "best_epoch": results["best_epoch"],
        "validation": {"loss": results["best_val_loss"], "accuracy": results["best_val_acc"],
                       "macro_f1": results["val_macro_f1"]},
        "test": test_metrics,
        "majority_class_baseline_test": baseline_metrics,
        "class_names": d["class_names"],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(),
                        "machine": platform.machine()},
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    out = common.OUTPUTS_DIR / "final"
    plot_curves(results["history"], results["best_epoch"], "Final model: training vs validation", out / "final_curves.png")
    plot_confusion_matrix(test_metrics["confusion_matrix"], d["class_names"], out / "confusion_matrix.png")
    with open(out / "architecture.txt", "w") as f:
        f.write(format_architecture(model) + "\n")

    print("\n================ FINAL TEST RESULTS (test set used once) ================")
    print(f"Test accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"Macro precision : {test_metrics['macro_precision']:.4f}")
    print(f"Macro recall    : {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1        : {test_metrics['macro_f1']:.4f}")
    print(f"Majority-class baseline accuracy: {baseline_metrics['accuracy']:.4f} "
          f"(always predicting '{d['class_names'][majority]}')")
    print(f"Saved model + preprocessing + test set to {models_dir}")


# ----------------------------------------------------------------------------- command line
def parse_args():
    p = argparse.ArgumentParser(description="Train the dry bean MLP")
    p.add_argument("--final", action="store_true", help="final run: also evaluate the test set ONCE and save the model")
    p.add_argument("--force", action="store_true", help="allow --final to overwrite an earlier final run")
    p.add_argument("--baseline", action="store_true", help="use BASELINE settings instead of FINAL")
    p.add_argument("--name", default="run", help="run name (used for output folder when not --final)")
    p.add_argument("--data", default=None, help="path to the dataset file (default: first file in data/)")
    p.add_argument("--hidden", default=None, help="e.g. 128,64,32")
    for key, kind in [("dropout", float), ("lr", float), ("momentum", float), ("weight_decay", float),
                      ("batch_size", int), ("epochs", int), ("patience", int), ("seed", int)]:
        p.add_argument(f"--{key.replace('_', '-')}", type=kind, default=None)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default=None)
    return p.parse_args()


def build_config(args):
    cfg = dict(BASELINE if args.baseline else FINAL)
    for key in ["dropout", "lr", "momentum", "weight_decay", "batch_size", "epochs", "patience", "seed", "optimizer"]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.hidden:
        cfg["hidden"] = [int(x) for x in args.hidden.split(",")]
    return cfg


def main():
    args = parse_args()
    if args.final:
        guard_test_set_used_once(args.force)
    cfg = build_config(args)
    print("Config:", json.dumps(cfg))
    d = data_prep.prepare_data(args.data, seed=cfg["seed"])
    print(data_prep.describe_split(d))

    model_preview = BeanMLP(len(d["feature_names"]), cfg["hidden"], len(d["class_names"]), cfg["dropout"])
    print("\nArchitecture:\n" + format_architecture(model_preview) + "\n")

    print("Training...")
    model, results = run_training(cfg, d)
    print(f"\nBest epoch {results['best_epoch']}: val loss {results['best_val_loss']:.4f}, "
          f"val acc {results['best_val_acc']:.4f}, val macro-F1 {results['val_macro_f1']:.4f}, "
          f"train acc {results['train_acc_at_best']:.4f} ({results['seconds']} s)")

    if args.final:
        final_evaluation(model, cfg, d, results, args.force)
    else:
        out = common.OUTPUTS_DIR / "runs" / args.name
        plot_curves(results["history"], results["best_epoch"], args.name, out / "curves.png")
        with open(out / "summary.json", "w") as f:
            json.dump({"config": cfg, **results}, f, indent=2)
        print(f"Saved curves + summary to {out}  (test set NOT used)")


if __name__ == "__main__":
    main()
