"""Run the baseline + all experiments from config.py and build the Experiment Log (Appendix B).

Uses train + validation data only. The test set is NOT touched here.
    python src/run_experiments.py
"""
import argparse
import json
from pathlib import Path

import common
import data_prep
from config import BASELINE, EXPERIMENTS
from train import plot_curves, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    d = data_prep.prepare_data(args.data, seed=BASELINE["seed"])
    print(data_prep.describe_split(d))

    runs = [("exp0_baseline", "Baseline", {})] + [(n, text, o) for n, text, o in EXPERIMENTS]
    out_root = common.OUTPUTS_DIR / "experiments"
    rows = []
    for index, (name, change, overrides) in enumerate(runs):
        cfg = {**BASELINE, **overrides}
        print(f"\n=== {name}: {change} ===")
        _, results = run_training(cfg, d)
        folder = out_root / name
        plot_curves(results["history"], results["best_epoch"], f"{name}: {change}", folder / "curves.png")
        with open(folder / "summary.json", "w") as f:
            json.dump({"config": cfg, "change": change, **results}, f, indent=2)
        rows.append({"exp": index, "name": name, "change": change, "cfg": cfg, "res": results})
        print(f"  -> best epoch {results['best_epoch']}, val loss {results['best_val_loss']:.4f}, "
              f"val acc {results['best_val_acc']:.4f}, train acc {results['train_acc_at_best']:.4f}")

    write_log(rows, out_root)
    plot_comparison(rows, out_root)
    print(f"\nExperiment log written to {out_root / 'experiment_log.md'}")


def write_log(rows, out_root):
    header = ("| Exp | What changed (one thing only) | Epochs (best) | LR | Batch | Val Loss | Val Acc | "
              "Train Acc | Val macro-F1 | Params |\n|---|---|---|---|---|---|---|---|---|---|")
    lines = [header]
    for r in rows:
        c, s = r["cfg"], r["res"]
        lines.append(
            f"| {r['exp']} | {r['change']} | {s['epochs_run']} ({s['best_epoch']}) | {c['lr']} | {c['batch_size']} | "
            f"{s['best_val_loss']:.4f} | {s['best_val_acc']:.4f} | {s['train_acc_at_best']:.4f} | "
            f"{s['val_macro_f1']:.4f} | {s['n_params']:,} |"
        )
    text = "\n".join(lines) + (
        "\n\nValues are taken at the best-validation-loss epoch. Under the table, write 2-3 sentences per "
        "experiment: what you expected, what happened, what you concluded from the loss curves.\n"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "experiment_log.md").write_text(text)
    print("\n" + text)


def plot_comparison(rows, out_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in rows:
        label = f"{r['exp']}: {r['change'].split(' (')[0]}"
        axes[0].plot(r["res"]["history"]["val_loss"], label=label)
        axes[1].plot(r["res"]["history"]["val_acc"], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Validation loss", title="Validation loss by experiment")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy", title="Validation accuracy by experiment")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_root / "compare_experiments.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
