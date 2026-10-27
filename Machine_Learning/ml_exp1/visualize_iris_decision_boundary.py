"""
Iris decision-boundary visualization in 2D style.
Uses two features (Sepal.Length, Sepal.Width) and a scratch softmax regression.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = Path("/Users/previous_rium/Desktop/PY/ml_exp1/dataset/iris")
IRIS_CSV = DATA_ROOT / "train" / "iris.csv"
OUT_FIG = Path("/Users/previous_rium/Desktop/PY/ml_exp1/iris_decision_boundary.png")

FEATURE_COLS = ["Sepal.Length", "Sepal.Width"]
CLASS_MAP = {"setosa": 0, "versicolor": 1, "virginica": 2}
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def standardize_fit_transform(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def standardize_transform(x, mean, std):
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    return (x - mean) / std_safe


def softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


class SoftmaxRegressionScratch:
    def __init__(self, n_features, n_classes, lr=0.05, epochs=600):
        self.w = np.zeros((n_features, n_classes), dtype=float)
        self.b = np.zeros((1, n_classes), dtype=float)
        self.lr = lr
        self.epochs = epochs

    def fit(self, x, y):
        y_oh = one_hot(y, self.b.shape[1])
        for _ in range(self.epochs):
            logits = x @ self.w + self.b
            probs = softmax(logits)
            grad_logits = (probs - y_oh) / len(y)
            grad_w = x.T @ grad_logits
            grad_b = np.sum(grad_logits, axis=0, keepdims=True)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict(self, x):
        logits = x @ self.w + self.b
        return np.argmax(logits, axis=1)


def main():
    df = pd.read_csv(IRIS_CSV, encoding="utf-8")
    x_raw = df[FEATURE_COLS].values.astype(float)
    y = df["Species"].map(CLASS_MAP).values.astype(int)

    x, mean, std = standardize_fit_transform(x_raw)
    model = SoftmaxRegressionScratch(n_features=2, n_classes=3, lr=0.05, epochs=600)
    model.fit(x, y)


    x_min, x_max = x_raw[:, 0].min() - 0.3, x_raw[:, 0].max() + 0.3
    y_min, y_max = x_raw[:, 1].min() - 0.3, x_raw[:, 1].max() + 0.3
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500),
    )
    grid_raw = np.c_[xx.ravel(), yy.ravel()]
    grid_std = standardize_transform(grid_raw, mean, std)
    zz = model.predict(grid_std).reshape(xx.shape)

    # Higher-contrast color palette
    bg_colors = ["#dbeafe", "#ffedd5", "#fef3c7"]
    pt_colors = ["#2563eb", "#ea580c", "#ca8a04"]

    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], colors=bg_colors, alpha=0.8)

    for class_id, name in enumerate(CLASS_NAMES):
        mask = y == class_id
        plt.scatter(
            x_raw[mask, 0],
            x_raw[mask, 1],
            c=pt_colors[class_id],
            edgecolors="#222222",
            linewidths=1.5,
            s=70,
            label=name,
        )

    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(alpha=0.15)
    plt.tight_layout()

    plt.savefig(OUT_FIG, dpi=180)
    print(f"Saved figure to: {OUT_FIG}")
    plt.show()


if __name__ == "__main__":
    main()
