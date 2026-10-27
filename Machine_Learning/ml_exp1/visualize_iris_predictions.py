"""
加载鸢尾花手写模型权重，生成样本分类结果分布图。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = Path("/Users/previous_rium/Desktop/PY/ml_exp1/dataset/iris")
IRIS_CSV = DATA_ROOT / "train" / "iris.csv"
MODEL_CKPT = Path("/Users/previous_rium/Desktop/PY/ml_exp1/checkpoints/Iris.npz")
OUT_FIG = Path("/Users/previous_rium/Desktop/PY/ml_exp1/iris_prediction_distribution.png")

FEATURE_COLS = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
CLASS_MAP = {"setosa": 0, "versicolor": 1, "virginica": 2}


def standardize_fit_transform(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def predict_multiclass(x, w, b):
    logits = x @ w + b
    return np.argmax(logits, axis=1)


def main():
    if not MODEL_CKPT.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint: {MODEL_CKPT}. Run trainer3.py first."
        )

    df = pd.read_csv(IRIS_CSV, encoding="utf-8")
    x = df[FEATURE_COLS].values.astype(float)
    y_true = df["Species"].map(CLASS_MAP).values.astype(int)

    # 与 trainer3 一致：先做标准化（这里在全数据上拟合，仅用于展示分布）
    x = standardize_fit_transform(x)

    state = np.load(MODEL_CKPT, allow_pickle=True)
    w = state["w"]
    b = state["b"]

    y_pred = predict_multiclass(x, w, b)

    true_counts = np.bincount(y_true, minlength=3)
    pred_counts = np.bincount(y_pred, minlength=3)

    print("True class counts:", dict(zip(CLASS_NAMES, true_counts.tolist())))
    print("Pred class counts:", dict(zip(CLASS_NAMES, pred_counts.tolist())))

    x_pos = np.arange(len(CLASS_NAMES))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x_pos - width / 2, true_counts, width=width, label="True", color="#4C72B0")
    plt.bar(x_pos + width / 2, pred_counts, width=width, label="Pred", color="#55A868")
    plt.xticks(x_pos, CLASS_NAMES)
    plt.ylabel("Sample Count")
    plt.title("Iris Class Distribution: True vs Predicted")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(OUT_FIG, dpi=160)
    print(f"Saved figure to: {OUT_FIG}")
    plt.show()


if __name__ == "__main__":
    main()
