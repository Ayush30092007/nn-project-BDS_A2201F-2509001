"""Exploratory data analysis (Task 2). Creates the visualisations and a text summary.

    python src/explore.py
Outputs go to outputs/figures/ and outputs/eda_summary.txt
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common
import data_prep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    path = data_prep.find_data_file(args.data)
    df_raw, label = data_prep.load_dataset(path)
    features = [c for c in df_raw.columns if c != label]
    df, report = data_prep.clean(df_raw.copy(), label)

    fig_dir = common.OUTPUTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- text summary
    counts = df[label].value_counts()
    lines = [
        f"File: {path}",
        f"Rows loaded: {report['rows_loaded']} | columns: {df_raw.shape[1]} ({len(features)} features + 1 label)",
        f"Missing values: {report['missing_values']}",
        f"Exact duplicate rows removed: {report['duplicates_removed']}",
        f"Rows after cleaning: {report['rows_after_cleaning']}",
        "",
        "Class distribution (after cleaning):",
    ] + [f"  {name:<12}{n:>6}  ({n / len(df):.1%})" for name, n in counts.items()]
    lines += [
        f"Imbalance ratio (largest / smallest class): {counts.max() / counts.min():.1f}",
        "",
        "Feature ranges (min .. max) - note the very different scales:",
    ] + [f"  {c:<18}{df[c].min():>14.4f} .. {df[c].max():<14.4f}" for c in features]
    lines += ["", "Outliers per feature (IQR rule, kept in the data):"] + [
        f"  {c:<18}{n:>6}" for c, n in report["outliers_per_feature_iqr"].items()
    ]
    text = "\n".join(lines)
    print(text)
    (common.OUTPUTS_DIR / "eda_summary.txt").write_text(text + "\n")

    # ---------- 1. class distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.bar_label(bars)
    ax.set(ylabel="Number of beans", title="Class distribution (imbalanced)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "1_class_distribution.png", dpi=150)
    plt.close(fig)

    # ---------- 2. feature histograms (different scales)
    cols = 4
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows))
    for ax, c in zip(axes.ravel(), features):
        ax.hist(df[c], bins=40, color="#55A868")
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.suptitle("Feature distributions (note the different scales)")
    fig.tight_layout()
    fig.savefig(fig_dir / "2_feature_histograms.png", dpi=150)
    plt.close(fig)

    # ---------- 3. correlation heatmap
    corr = df[features].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=8)
    ax.set_yticks(range(len(features)), features, fontsize=8)
    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{corr[i, j]:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title("Feature correlation (many size features move together)")
    fig.tight_layout()
    fig.savefig(fig_dir / "3_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # ---------- 4. boxplots of standardised features (shows outliers)
    z = (df[features] - df[features].mean()) / df[features].std()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.boxplot([z[c] for c in features], tick_labels=features, flierprops={"markersize": 2})
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    ax.set(ylabel="Standard deviations from the mean", title="Outliers in each feature (standardised)")
    fig.tight_layout()
    fig.savefig(fig_dir / "4_outlier_boxplots.png", dpi=150)
    plt.close(fig)

    # ---------- 5. the 4 features that best separate the classes
    class_names = sorted(df[label].unique())
    means = np.array([z[df[label] == k].mean().to_numpy() for k in class_names])
    spread = means.std(axis=0)
    best = [features[i] for i in np.argsort(-spread)[:4]]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, c in zip(axes, best):
        ax.boxplot([df.loc[df[label] == k, c] for k in class_names], tick_labels=class_names,
                   flierprops={"markersize": 2})
        ax.set_title(c)
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    fig.suptitle("Features that separate the bean varieties best")
    fig.tight_layout()
    fig.savefig(fig_dir / "5_features_by_class.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved 5 figures to {fig_dir} and the summary to {common.OUTPUTS_DIR / 'eda_summary.txt'}")


if __name__ == "__main__":
    main()
