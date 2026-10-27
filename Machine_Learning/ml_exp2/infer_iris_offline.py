"""
加载 train_bp.py 保存的 Iris 权重 (checkpoints/Iris.npz)，在对应验证折上评估，
并绘制二维特征切片上的决策区域图；散点默认展示全部 150 条样本（用训练折 mean/std 标准化）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "datasets" / "iris" / "train" / "iris.csv"
CHECKPOINTS = ROOT / "checkpoints"
WEIGHTS_DEFAULT = CHECKPOINTS / "Iris.npz"

SPECIES_NAMES = ["setosa", "versicolor", "virginica"]
N_CLASSES = 3

_MPL_FONT_CONFIGURED = False


def configure_matplotlib_font() -> None:
    global _MPL_FONT_CONFIGURED
    if _MPL_FONT_CONFIGURED:
        return
    _MPL_FONT_CONFIGURED = True
    mpl.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    mpl.rcParams["axes.unicode_minus"] = False


def softmax(z: np.ndarray) -> np.ndarray:
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(x: np.ndarray, w1, b1, w2, b2) -> np.ndarray:
    z1 = x @ w1 + b1
    a1 = np.maximum(0.0, z1)
    return softmax(a1 @ w2 + b2)


def predict(x: np.ndarray, w1, b1, w2, b2) -> np.ndarray:
    return np.argmax(forward(x, w1, b1, w2, b2), axis=1)


def standardize_fit_transform(x: np.ndarray):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def standardize_transform(x: np.ndarray, mean, std):
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    return (x - mean) / std_safe


def indices(y, n_splits=10, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    chunks = {}
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        chunks[c] = np.array_split(idx, n_splits)

    for fold in range(n_splits):
        val_idx = np.concatenate([chunks[c][fold] for c in classes])
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx, assume_unique=True)
        yield train_idx, val_idx


def load_iris_fold():
    df = pd.read_csv(DATA_ROOT)
    feature_cols = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    x = df[feature_cols].to_numpy(dtype=float)
    y = (
        df["Species"]
        .map({"setosa": 0, "versicolor": 1, "virginica": 2})
        .to_numpy(dtype=int)
    )
    train_idx, val_idx = next(indices(y, n_splits=10, seed=42))
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_train_s, mean, std = standardize_fit_transform(x_train)
    x_val_s = standardize_transform(x_val, mean, std)
    # 全表用训练折的 mean/std 标准化，便于在图上画齐所有样本
    x_all_std = standardize_transform(x, mean, std)
    return x_val_s, y_val, x_all_std, y, feature_cols


def plot_decision_slice(
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    feature_cols: list[str],
    w1,
    b1,
    w2,
    b2,
    sample_idx: int,
    feat_i: int,
    feat_j: int,
    grid_points: int,
    out_path: Path,
    show_all_points: bool,
    slice_mode: str,
):
    """
    slice_mode:
      mean — 未画出的特征固定在「训练折标准化后的均值」= 全 0，与散点的统计中心一致，避免错位感。
      anchor — 未画出的特征固定在验证集 anchor 样本（旧行为）。
    """
    n = x_val.shape[1]
    if not (0 <= feat_i < n and 0 <= feat_j < n):
        raise ValueError(f"feat index out of range 0..{n - 1}")
    if feat_i == feat_j:
        raise ValueError("feat_i and feat_j must differ")

    if slice_mode == "mean":
        slice_base = np.zeros((1, n), dtype=float)
    elif slice_mode == "anchor":
        slice_base = x_val[sample_idx : sample_idx + 1].copy()
    else:
        raise ValueError("slice_mode must be 'mean' or 'anchor'")

    xs = x_plot if show_all_points else x_val
    u_min, u_max = float(xs[:, feat_i].min()), float(xs[:, feat_i].max())
    v_min, v_max = float(xs[:, feat_j].min()), float(xs[:, feat_j].max())
    pad_u = 0.08 * (u_max - u_min + 1e-6)
    pad_v = 0.08 * (v_max - v_min + 1e-6)
    u = np.linspace(u_min - pad_u, u_max + pad_u, grid_points)
    v = np.linspace(v_min - pad_v, v_max + pad_v, grid_points)
    uu, vv = np.meshgrid(u, v)

    grid_x = np.tile(slice_base, (grid_points * grid_points, 1))
    grid_x[:, feat_i] = uu.ravel()
    grid_x[:, feat_j] = vv.ravel()
    z = predict(grid_x, w1, b1, w2, b2).reshape(grid_points, grid_points)

    configure_matplotlib_font()
    level_edges = np.arange(N_CLASSES + 1) - 0.5

    plt.figure(figsize=(8, 6))
    cf = plt.contourf(
        uu, vv, z, levels=level_edges, cmap="tab10", alpha=0.72, extend="both"
    )
    cb = plt.colorbar(cf, ticks=np.arange(N_CLASSES))
    cb.ax.set_yticklabels(SPECIES_NAMES)
    cb.set_label("Predicted species")

    if show_all_points:
        plt.scatter(
            x_plot[:, feat_i],
            x_plot[:, feat_j],
            c=y_plot,
            cmap="tab10",
            vmin=-0.5,
            vmax=N_CLASSES - 0.5,
            s=38,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
            label=f"all samples (n={len(y_plot)}, true class)",
        )

    ax_i = float(x_val[sample_idx, feat_i])
    ax_j = float(x_val[sample_idx, feat_j])
    plt.scatter(
        ax_i,
        ax_j,
        c="red",
        s=220,
        marker="*",
        edgecolors="k",
        linewidths=0.8,
        zorder=6,
        label=f"val anchor #{sample_idx} ({SPECIES_NAMES[int(y_val[sample_idx])]})",
    )
    plt.xlabel(f"{feature_cols[feat_i]} (standardized)")
    plt.ylabel(f"{feature_cols[feat_j]} (standardized)")
    if slice_mode == "mean":
        plt.title(f"Iris · mean slice · val #{sample_idx}")
    else:
        plt.title(f"Iris · anchor slice · val #{sample_idx}")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Iris BP offline eval + decision slice")
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS_DEFAULT,
        help="npz from train_bp.py (w1,b1,w2,b2)",
    )
    parser.add_argument("--sample", type=int, default=0, help="anchor index in val set")
    parser.add_argument(
        "--feat1",
        type=int,
        default=2,
        help="x-axis feature index (default 2=Petal.Length)",
    )
    parser.add_argument(
        "--feat2",
        type=int,
        default=3,
        help="y-axis feature index (default 3=Petal.Width)",
    )
    parser.add_argument("--grid", type=int, default=120)
    parser.add_argument(
        "--no-val-points",
        action="store_true",
        help="do not scatter sample points on the plot",
    )
    parser.add_argument(
        "--slice",
        choices=["mean", "anchor"],
        default="mean",
        help="mean: fix non-plot dims to 0 (train std space); anchor: fix to val sample",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CHECKPOINTS / "iris_decision_slice.png",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(
            f"Missing weights: {args.weights}\nRun train_bp.py first to save Iris.npz."
        )

    d = np.load(args.weights, allow_pickle=False)
    w1, b1, w2, b2 = d["w1"], d["b1"], d["w2"], d["b2"]

    x_val, y_val, x_all_std, y_all, feature_cols = load_iris_fold()
    y_pred = predict(x_val, w1, b1, w2, b2)
    y_names = [SPECIES_NAMES[i] for i in y_val]
    pred_names = [SPECIES_NAMES[i] for i in y_pred]

    print(f"Weights: {args.weights}")
    print(f"Val size: {len(y_val)}")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}\n")
    print(
        classification_report(
            y_names, pred_names, labels=SPECIES_NAMES, digits=4, zero_division=0
        )
    )

    if not args.no_val_points:
        print(
            "\n[关于决策图与散点]\n"
            "  背景：仅两维变化，其余维由 --slice 固定（mean=训练标准化后各维 0）。\n"
            "  散点：每个样本仍是完整 4 维，只投影两维到底图，与切片不必一致属正常现象。\n"
            "  单样本切片：--slice anchor\n"
        )

    si = int(np.clip(args.sample, 0, len(y_val) - 1))
    plot_decision_slice(
        x_val,
        y_val,
        x_all_std,
        y_all,
        feature_cols,
        w1,
        b1,
        w2,
        b2,
        sample_idx=si,
        feat_i=args.feat1,
        feat_j=args.feat2,
        grid_points=args.grid,
        out_path=args.out,
        show_all_points=not args.no_val_points,
        slice_mode=args.slice,
    )


if __name__ == "__main__":
    main()
