| Exp | What changed (one thing only) | Epochs (best) | LR | Batch | Val Loss | Val Acc | Train Acc | Val macro-F1 | Params |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Baseline | 60 (16) | 0.001 | 64 | 0.2074 | 0.9276 | 0.9373 | 0.9370 | 12,743 |
| 1 | Dropout 0.0 -> 0.3 (regularisation) | 60 (58) | 0.001 | 64 | 0.2131 | 0.9227 | 0.9386 | 0.9336 | 12,743 |
| 2 | Hidden layers [128,64,32] -> [32,16] (capacity) | 60 (50) | 0.001 | 64 | 0.2032 | 0.9286 | 0.9363 | 0.9374 | 1,191 |
| 3 | Optimiser Adam -> SGD with momentum 0.9 | 60 (59) | 0.001 | 64 | 0.2108 | 0.9252 | 0.9321 | 0.9336 | 12,743 |
| 4 | Learning rate 0.001 -> 0.01 | 60 (3) | 0.01 | 64 | 0.2234 | 0.9183 | 0.9301 | 0.9285 | 12,743 |

Values are taken at the best-validation-loss epoch. Under the table, write 2-3 sentences per experiment: what you expected, what happened, what you concluded from the loss curves.
