"""Predict the bean variety for ONE new sample (runs inside the test container too).

Examples
    python src/predict.py --demo                      # use a row from the saved test set
    python src/predict.py --demo --index 5
    python src/predict.py --features "28395,610.291,208.178,173.889,1.197,0.55,28715,190.141,0.764,0.989,0.958,0.913,0.007,0.003,0.834,0.999"
    python src/predict.py --show-order                # print the 16 feature names in the required order

The values must be RAW (unscaled) numbers in exactly the order of the feature names; the saved
scaler is applied for you.
"""
import argparse
from pathlib import Path

import numpy as np

import common


def main():
    parser = argparse.ArgumentParser(description="Predict the variety of one dry bean")
    parser.add_argument("--models-dir", default=str(common.MODELS_DIR))
    parser.add_argument("--features", default=None, help="comma-separated raw feature values")
    parser.add_argument("--demo", action="store_true", help="use a sample from the saved test set")
    parser.add_argument("--index", type=int, default=0, help="which test row to use with --demo")
    parser.add_argument("--show-order", action="store_true", help="print the feature order and exit")
    args = parser.parse_args()

    model, _, prep = common.load_artifacts(Path(args.models_dir))
    names = prep["feature_names"]
    classes = prep["class_names"]

    if args.show_order:
        print("Feature order:", ", ".join(f"{i + 1}.{n}" for i, n in enumerate(names)))
        return

    true_label = None
    if args.demo:
        X_raw, y = common.load_test_set(args.models_dir, prep)
        if not 0 <= args.index < len(y):
            raise SystemExit(f"--index must be between 0 and {len(y) - 1}")
        values = X_raw[args.index]
        true_label = classes[y[args.index]]
    elif args.features:
        try:
            values = np.array([float(v) for v in args.features.split(",")], dtype=np.float64)
        except ValueError:
            raise SystemExit("--features must be comma-separated numbers, e.g. 28395,610.291,...")
        if len(values) != len(names):
            raise SystemExit(f"Expected {len(names)} values, got {len(values)}. Use --show-order to see them.")
        if not np.all(np.isfinite(values)):
            raise SystemExit("All feature values must be finite numbers.")
    else:
        raise SystemExit("Give --features \"v1,...,v16\" or use --demo (see --help).")

    probs = common.predict_proba(model, common.apply_scaler(values.reshape(1, -1), prep["mean"], prep["scale"]))[0]
    order = np.argsort(-probs)

    print("Input features:")
    for n, v in zip(names, values):
        print(f"  {n:<18}{v:>14.4f}")
    print(f"\nPredicted variety: {classes[order[0]]}  (confidence {probs[order[0]]:.1%})")
    if true_label is not None:
        print(f"True variety     : {true_label}  ->  {'CORRECT' if true_label == classes[order[0]] else 'WRONG'}")
    print("\nAll class probabilities:")
    for i in order:
        print(f"  {classes[i]:<12}{probs[i]:>8.2%}")


if __name__ == "__main__":
    main()
