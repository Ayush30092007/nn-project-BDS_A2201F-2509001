# Your project explained from zero

Read this once today, again before the presentation. Questions from the lecturer's question bank are marked **[A#]**, **[B#]**, **[C#]**.

---

## 1. The big picture (30 seconds)

Each dry bean was photographed and measured by a computer: size, perimeter, roundness, etc. That gives **16 numbers per bean**.
A human expert labelled each bean with one of **7 varieties**. Your neural network learns the pattern
"these 16 numbers -> this variety". You then check, on beans it has never seen, how often it is right.

That is **supervised multiclass classification**: inputs (16 features) -> one of 7 classes.

## 2. The pipeline, step by step

| Step | What happens | Why |
|---|---|---|
| Load | Read `Dry_Bean_Dataset.xlsx` | |
| Clean | Remove missing rows + exact duplicates | Duplicates could land in both train and test = cheating |
| Split | 70 % train / 15 % validation / 15 % test, keeping class proportions (stratified) | See section 4 |
| Scale | Standardise: `(x - mean) / std`, mean/std from the **training set only** | Features are on wildly different scales (Area ~ 50,000 vs Extent ~ 0.7) |
| Train | Adjust weights to reduce the loss | |
| Experiment | Change ONE thing, compare on validation | Find what helps |
| Test (once) | Final honest score | |
| Docker | Same results in a different, clean environment | Reproducibility |

## 3. What each file does

| File | Job |
|---|---|
| `config.py` | All settings. BASELINE (never edit after running), EXPERIMENTS, FINAL (edit after experiments) |
| `data_prep.py` | Load, clean, split, scale |
| `model.py` | The network itself |
| `common.py` | Scaling, prediction, metrics - shared so train and test behave identically |
| `explore.py` | Makes the 5 data plots (needed: at least 3) |
| `train.py` | Trains. With `--final` also tests once and saves the model |
| `run_experiments.py` | Trains baseline + 4 experiments, writes the Experiment Log + curves |
| `test.py` | Test container: loads the model, prints all metrics, wrong predictions, PASS/FAIL reproducibility |
| `predict.py` | One new sample -> predicted variety + probabilities |

## 4. Concepts in plain English

**Neuron [A1].** Takes inputs, multiplies each by a *weight*, adds them up, adds a *bias*, then passes the result through an *activation function*. Output = `activation(w1*x1 + w2*x2 + ... + bias)`.

**Bias [A2].** A number added before the activation. It shifts the neuron's decision point so it need not pass through zero.

**Layers [A5].** Input layer = your 16 numbers (no computation). Hidden layers = neurons that learn useful combinations of features. Output layer = 7 numbers, one score per variety.

**Your architecture.** `16 -> 128 -> 64 -> 32 -> 7`. Three hidden layers (the brief needs at least two). Decreasing widths = gradually compress the information into what separates the classes.

**Activation function: ReLU [A3, A4].** `ReLU(x) = max(0, x)`. Without a non-linear activation, any stack of layers collapses into one straight-line formula and cannot learn curved boundaries. ReLU is cheap and does not saturate for positive inputs (sigmoid/tanh flatten out at the ends, which causes vanishing gradients). Weakness: a neuron stuck at 0 can "die".

**Softmax [A8].** Turns the 7 output scores into probabilities between 0 and 1 that add up to 1. Matches "exactly one class is right". In the code, `CrossEntropyLoss` applies it internally; `predict.py` applies it explicitly.

**Loss function [A7].** One number saying how wrong the predictions are. You use **cross-entropy** (classification). It punishes confident wrong answers heavily. Regression would use MSE/MAE. [C2]: softmax + categorical cross-entropy matches a 7-class, one-answer task.

**Forward pass [A6].** Data flows in -> layer by layer (weighted sum + bias, activation) -> 7 scores out.

**Backpropagation [B1].** After the forward pass, compute the loss, then use the chain rule to find how much each weight contributed to the error (the *gradient*), working backwards from output to input. In the code: `loss.backward()`.

**Gradient descent [B2].** Move each weight a small step opposite to its gradient: `w = w - learning_rate * gradient`. In the code: `optimizer.step()`.

**Learning rate [B3].** Step size. Too high -> loss bounces or explodes. Too low -> learns painfully slowly. (Your experiment 4 tests 0.01 vs 0.001.)

**Adam vs SGD [B5].** SGD uses one learning rate for all weights (with momentum it smooths direction). Adam adapts a separate step size per weight using running averages of gradients and squared gradients - usually converges faster with less tuning (your experiment 3).

**Epoch / batch / iteration [B4].** Epoch = one full pass through the training data. Batch = the small group (64) used for one weight update. Iteration = one update. Training rows / 64 = iterations per epoch.

**Weights vs hyperparameters [A9].** Weights are learned by training. Hyperparameters (layers, neurons, learning rate, epochs, dropout) are chosen by you.

**MLP vs CNN [A10, A11].** MLP = every neuron connects to every neuron of the next layer; right for a table of numbers (your data). CNN = small shared filters slide over an image; right for pictures. Your 16 features were already extracted from images, so an MLP fits.

**Overfitting [B6, B7].** The model memorises the training set. Sign: training loss keeps falling but **validation loss starts rising**. Fixes: dropout, L2 weight decay, early stopping, smaller model, more data.
**Underfitting [B8].** Both losses stay high - model too small or trained too little. Fixes: bigger model, train longer, lower regularisation.
**Reading your curves [C4].** Look at the gap between training and validation. Small gap, both low = fits well. Big and growing gap = overfitting. Both high and flat = underfitting.

**Dropout.** During training, randomly switch off a fraction of neurons each step so the network cannot rely on any single one. Switched off at test time.

**Train / validation / test [B9, C6].** Train = learn weights. Validation = compare experiments and tune settings. Test = one final, unbiased score. If you pick settings by looking at test results, the test score is no longer honest (information has leaked). That is why `train.py` only touches the test set with `--final`, once.

**Why scale features [B10].** With features of very different size, gradient descent zigzags and converges slowly. After standardising, each feature has mean 0, std 1. The scaler is fitted on the training set only, otherwise information from validation/test leaks in.

**Why save the scaler too [C8].** A new bean must be transformed with exactly the same mean/std as the training data, otherwise the network sees numbers on the wrong scale and predicts nonsense. Your `preprocessing.json` stores it.

**Class imbalance and metrics [B12, B13].** Dermason has about 3,500 beans, Bombay about 500. A lazy model that always says "Dermason" already gets ~26 % accuracy (the majority baseline), so accuracy alone can mislead. Use:
- **Precision** = of the beans predicted as X, how many really are X.
- **Recall** = of the beans that really are X, how many were found.
- **F1** = balance of the two. **Macro F1** averages F1 over the 7 classes equally, so small classes count as much as big ones.
- **Confusion matrix** = shows which varieties get mixed up.

**Vanishing gradient [B11].** In deep nets with sigmoid/tanh, gradients shrink as they travel back, so early layers barely learn. ReLU and normalisation help.

**Parameter count [C3].** One dense layer = `inputs x neurons + neurons (biases)`.
Your baseline: 16x128+128 = 2,176 · 128x64+64 = 8,256 · 64x32+32 = 2,080 · 32x7+7 = 231 -> **12,743 total**.

## 5. Docker in plain English [B14]

- **Image** = a frozen recipe + ingredients: Python + exact library versions + your code.
- **Container** = a running copy of an image. Starts clean, disappears when finished (`--rm`).
- **Volume / bind mount** = a shared folder between your computer and the container (`-v host:container`). Data and models live outside the image so the image stays small and results survive after the container is gone.
- **Why two Dockerfiles:** training needs many libraries; testing only needs torch + numpy, so `Dockerfile.test` makes a lean image. Because the test container shares nothing with the training one except the `models/` folder, a passing test proves the result does not depend on your training setup.
- **Why it helps ML:** the same library versions everywhere -> the same results on the lecturer's machine.

## 6. How to read your experiment results

After `run_experiments.py`, open `outputs/experiments/experiment_log.md` and the curves in each experiment folder. Compare **validation loss**, not training loss.

| Experiment | Question it answers | What you might see |
|---|---|---|
| 0 Baseline | Where do we start? | Training loss keeps dropping; check if validation loss turns up (overfitting) |
| 1 Dropout 0.3 | Does regularisation shrink the train/validation gap? | Smaller gap, maybe slightly lower training accuracy |
| 2 Smaller model | Is the baseline too big? | If accuracy barely drops, extra capacity was not needed |
| 3 SGD | Does the optimiser matter? | Often slower to converge than Adam in the same epochs |
| 4 LR 0.01 | Is the step size right? | Noisier curves, possibly worse or faster early |

Write 2-3 sentences each: **what you expected, what happened, what you concluded.** Then copy the best settings (by validation results) into `FINAL` in `config.py`. Judge fairly: differences of a few hundredths of a percent are noise, so say so rather than overclaiming.

## 7. Rules that protect your marks

1. Never choose settings using test results. Test once, at the end.
2. Do not edit `BASELINE` after running experiments.
3. Run the test container on a **different machine** (or a clean user account with no Python packages) and screenshot **both** environments.
4. Declare AI use in the report, and **be able to explain every file** - the Q&A (30 marks) is closed-book.
5. Bring your laptop with images already built, plus a recorded backup of the demo.
6. Put your real name/ID in the README, and name the repo `nn-project-<studentID>`.
