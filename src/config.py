"""All hyperparameters live here so every experiment is easy to read and reproduce.

BASELINE  -> the first model you train. Never edit this after you have run the
             experiments (otherwise your Experiment Log no longer matches the code).
EXPERIMENTS -> each one changes exactly ONE thing compared with BASELINE.
FINAL     -> the configuration used for the final model. After the experiments,
             copy the best settings (judged on VALIDATION results only) into FINAL.
"""

# Seed used everywhere (data split, weight initialisation, shuffling).
SEED = 42

# 70 % train / 15 % validation / 15 % test, stratified by class.
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

BASELINE = {
    "hidden": [128, 64, 32],   # three hidden layers (neurons per layer)
    "dropout": 0.0,            # 0.0 = no dropout
    "optimizer": "adam",       # "adam" or "sgd"
    "lr": 1e-3,                # learning rate
    "momentum": 0.0,           # only used by SGD
    "weight_decay": 0.0,       # L2 regularisation strength
    "batch_size": 64,
    "epochs": 60,
    "patience": None,          # None = no early stopping (train all epochs)
    "seed": SEED,
}

# (name, "what changed" text for the log, overrides applied on top of BASELINE)
EXPERIMENTS = [
    ("exp1_dropout", "Dropout 0.0 -> 0.3 (regularisation)", {"dropout": 0.3}),
    ("exp2_small_model", "Hidden layers [128,64,32] -> [32,16] (capacity)", {"hidden": [32, 16]}),
    ("exp3_sgd", "Optimiser Adam -> SGD with momentum 0.9", {"optimizer": "sgd", "momentum": 0.9}),
    ("exp4_lr_0.01", "Learning rate 0.001 -> 0.01", {"lr": 1e-2}),
]

# TODO (after experiments): edit this to the best settings found on the validation set.
FINAL = {**BASELINE, "hidden": [32, 16]}
