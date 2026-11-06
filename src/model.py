"""The neural network itself: a custom Multilayer Perceptron (MLP) for dry bean classification.

Architecture (baseline):
    16 inputs -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(7)

The last layer outputs 7 raw scores ("logits"), one per bean variety.
Softmax is NOT written in the model on purpose: nn.CrossEntropyLoss applies
log-softmax internally (more numerically stable), and at prediction time we
apply softmax ourselves to get probabilities that sum to 1.
"""
import torch
from torch import nn


class BeanMLP(nn.Module):
    def __init__(self, n_features=16, hidden=(128, 64, 32), n_classes=7, dropout=0.0):
        super().__init__()
        layers = []
        previous = n_features
        for width in hidden:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # raw scores (logits)


def count_parameters(model):
    """Number of trainable parameters (weights + biases)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def architecture_table(model):
    """Rows describing each Linear layer: name, shape, activation, parameter count.

    For one dense layer: parameters = (inputs x neurons) + neurons (the biases).
    """
    modules = list(model.net)
    rows = []
    linear_index = 0
    for i, module in enumerate(modules):
        if not isinstance(module, nn.Linear):
            continue
        linear_index += 1
        is_last = all(not isinstance(m, nn.Linear) for m in modules[i + 1:])
        activation = "Softmax (inside the loss)" if is_last else "ReLU"
        params = module.in_features * module.out_features + module.out_features
        rows.append({
            "layer": "output" if is_last else f"hidden{linear_index}",
            "shape": f"{module.in_features} -> {module.out_features}",
            "neurons": module.out_features,
            "activation": activation,
            "params": params,
        })
    return rows


def format_architecture(model):
    rows = architecture_table(model)
    lines = [f"{'Layer':<9}{'Shape':<12}{'Neurons':<9}{'Activation':<28}{'Parameters':>10}"]
    for r in rows:
        lines.append(f"{r['layer']:<9}{r['shape']:<12}{r['neurons']:<9}{r['activation']:<28}{r['params']:>10,}")
    lines.append(f"{'TOTAL':<58}{count_parameters(model):>10,}")
    return "\n".join(lines)
