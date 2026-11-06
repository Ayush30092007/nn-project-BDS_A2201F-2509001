# Dry Bean Variety Classification Using a Custom Multilayer Perceptron

**Course:** BAI13143 Neural Network Basics — Individual Project
**Student:** `<Your Name>` (`<Student ID>`)
**Framework:** PyTorch (CPU) · **Reproducibility:** Docker (separate train and test containers)

A custom MLP (3 hidden layers, ReLU, softmax output) that classifies **seven dry bean varieties**
(Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) from **16 shape/geometry features** extracted from
bean images. Automated grading like this can help growers and processors sort seed stock consistently.

## Dataset

Dry Bean Dataset — Koklu, M. and Ozkan, I.A. (2020). *Multiclass classification of dry beans using computer
vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507.
UCI Machine Learning Repository, <https://doi.org/10.24432/C50S4B>. Licence: **CC BY 4.0**.
13,611 samples · 16 features · 7 classes. The file is **not** copied into any Docker image; it is mounted.

Download it from <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset>, unzip, and place
`Dry_Bean_Dataset.xlsx` in the `data/` folder (a `.csv` with a `Class` column also works).

## Project structure

```
nn-project-<studentID>/
├── data/                  dataset (mounted into the container, not copied into the image)
├── models/                saved model + preprocessing + held-out test rows (shared between containers)
├── outputs/               figures, experiment log, final curves, confusion matrix
├── src/
│   ├── config.py          all hyperparameters (BASELINE, EXPERIMENTS, FINAL)
│   ├── data_prep.py       load -> clean -> stratified split -> scale (training image only)
│   ├── model.py           the MLP architecture
│   ├── common.py          scaling, prediction and metric functions shared by both containers
│   ├── explore.py         exploratory data analysis (figures + summary)
│   ├── train.py           training; `--final` also evaluates the test set once and saves the model
│   ├── run_experiments.py baseline + controlled experiments -> experiment log
│   ├── test.py            test container: loads the model, prints metrics, error analysis
│   └── predict.py         predict one new sample
├── Dockerfile.train       training image (torch, numpy, pandas, scikit-learn, matplotlib)
├── Dockerfile.test        small test image (torch + numpy only)
├── requirements-train.txt / requirements-test.txt   pinned versions
├── docker-compose.yml     optional shortcuts
└── docs/GUIDE.md          plain-English explanation of the whole project
```

## Network design

`16 inputs -> Dense 128 (ReLU) -> Dense 64 (ReLU) -> Dense 32 (ReLU) -> Dense 7 (softmax)` — 12,743 trainable
parameters for the baseline. Loss: cross-entropy. Optimiser: Adam. The exact table for the final model is
written to `outputs/final/architecture.txt` after the final run.

## Run it with Docker

Run all commands from the project root. Bash / macOS / Linux shown; in **PowerShell** replace `$(pwd)` with `${PWD}`.

### 1. Training environment

```bash
docker build -f Dockerfile.train -t nn-train .

# (a) exploratory data analysis -> outputs/figures, outputs/eda_summary.txt
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" nn-train python src/explore.py

# (b) baseline + experiments (train/validation only) -> outputs/experiments/experiment_log.md
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" nn-train python src/run_experiments.py

# (c) FINAL training - run ONCE. Evaluates the test set once and saves everything into models/
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/outputs:/app/outputs" nn-train
```

`models/` now contains `model.pt`, `model_config.json`, `preprocessing.json` (scaler mean/std, class names),
`test_set.csv` (the held-out rows) and `train_report.json` (the test metrics reported at training time).

### 2. Testing environment (a different machine or a clean user account)

Copy the project (at minimum `Dockerfile.test`, `requirements-test.txt`, `src/` and `models/`) to the other machine, then:

```bash
docker build -f Dockerfile.test -t nn-test .

# Evaluate on the test set and compare with the numbers reported during training
docker run --rm -v "$(pwd)/models:/app/models" nn-test

# Predict one new sample (16 raw feature values, in this order)
docker run --rm -v "$(pwd)/models:/app/models" nn-test python src/predict.py --show-order
docker run --rm -v "$(pwd)/models:/app/models" nn-test python src/predict.py --features "v1,v2,...,v16"

# Or take a row from the saved test set (row numbers match test_errors.csv)
docker run --rm -v "$(pwd)/models:/app/models" nn-test python src/predict.py --demo --index 3
```

The test run prints the environment (host, OS, Python/torch versions), accuracy, precision, recall, F1,
the confusion matrix, the majority-class baseline, the wrong predictions, and finally
`RESULT: PASS - test metrics reproduced exactly` if the numbers equal those in `models/train_report.json`.

### Shortcut with docker compose (optional)

```bash
docker compose build
docker compose run --rm explore
docker compose run --rm experiments
docker compose run --rm train
docker compose run --rm test
```

### Docker Hub (optional)

```bash
docker tag nn-test <dockerhub-user>/nn-test:1.0
docker push <dockerhub-user>/nn-test:1.0
# anyone can then run it (with the repo's models/ folder):
docker run --rm -v "$(pwd)/models:/app/models" <dockerhub-user>/nn-test:1.0
```

## Reproducibility

- Library versions are pinned in the requirements files; the same base image (`python:3.13-slim`) is used for both containers.
- Random seed 42 is fixed for the data split, weight initialisation and batch shuffling; PyTorch deterministic mode and a single CPU thread are enabled.
- Duplicates are removed **before** splitting and the scaler is fitted on the **training** rows only (no data leakage).
- The test set is used once: `train.py --final` refuses to run twice unless `--force` is given.
- The same scaling, prediction and metric functions (`common.py`) are used in training and testing, so the test container reproduces the reported numbers.
- Train once and copy `models/` to the other machine. Retraining on a different CPU architecture can differ in the last decimal places.

## Running without Docker (for quick debugging)

```bash
pip install -r requirements-train.txt
python src/explore.py
python src/run_experiments.py
python src/train.py --final
python src/test.py
```

## Results

_Fill in after the final run (numbers come from `models/train_report.json` / the test container)._

| Metric | Value |
|---|---|
| Test accuracy | |
| Macro F1 | |
| Majority-class baseline accuracy | |
| Target (from registration) | accuracy > 90 %, macro F1 > 0.90 |

## Troubleshooting

- **`No dataset found in the data/ folder`** — put `Dry_Bean_Dataset.xlsx` (or a `.csv`) in `data/`.
- **`STOP: models/train_report.json already exists`** — the final run was already done (the test set is used once). Use `--force` only if you changed the code and must redo it.
- **Build error "No matching distribution found for torch==2.15.0"** — the CPU wheel index does not offer that version for your platform. Remove the `--extra-index-url` line from both requirements files and rebuild (larger image, same results).
- **Empty `models/` in the test container** — the `-v .../models:/app/models` path is wrong or the folder was not copied to this machine.

## Generative AI declaration

_Declare here (and in the report) how generative AI was used, e.g. "Claude was used to help structure the repository, explain neural network concepts and debug code. I reviewed, ran and can explain all code."_
