"""
离线加载手写 BP 权重，在验证集上评估，并画二维决策切片图。

默认 --slice mean：未画出的特征固定在「训练集 StandardScaler 之后」的均值（各维为 0），
与散点叠在一起时比「锁在某一验证样本上」更不易产生错位感。
若需旧行为，使用 --slice anchor。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_ROOT = Path(__file__).resolve().parent / "datasets" / "wine"
TRAIN_CSV = DATA_ROOT / "winequality-red.csv"
CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"

_MPL_FONT_CONFIGURED = False


def configure_matplotlib_font() -> None:
    """尽量使用系统中文字体，减少图中中文缺字警告（macOS 常见为 PingFang SC）。"""
    global _MPL_FONT_CONFIGURED
    if _MPL_FONT_CONFIGURED:
        return
    _MPL_FONT_CONFIGURED = True
    mpl.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    mpl.rcParams["axes.unicode_minus"] = False


def softmax(z: np.ndarray) -> np.ndarray:
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward_wine(x: np.ndarray, w1, b1, w2, b2) -> np.ndarray:
    z1 = x @ w1 + b1
    a1 = np.maximum(0.0, z1)
    return softmax(a1 @ w2 + b2)


def forward_wine2(x: np.ndarray, w1, b1, w2, b2, w3, b3) -> np.ndarray:
    z1 = x @ w1 + b1
    a1 = np.maximum(0.0, z1)
    z2 = a1 @ w2 + b2
    a2 = np.maximum(0.0, z2)
    return softmax(a2 @ w3 + b3)


def load_weights(path: Path) -> tuple[dict, str]:
    """根据 npz 中是否含 w3/b3 自动区分 wine（单隐层）与 wine2（双隐层）。"""
    data = np.load(path, allow_pickle=False)
    names = set(data.files)
    if "w3" in names and "b3" in names:
        variant = "wine2"
        weights = {
            "w1": data["w1"],
            "b1": data["b1"],
            "w2": data["w2"],
            "b2": data["b2"],
            "w3": data["w3"],
            "b3": data["b3"],
        }
    elif {"w1", "b1", "w2", "b2"}.issubset(names):
        variant = "wine"
        weights = {
            "w1": data["w1"],
            "b1": data["b1"],
            "w2": data["w2"],
            "b2": data["b2"],
        }
    else:
        raise ValueError(
            f"无法识别权重格式: {path}，期望键 w1,b1,w2,b2 或再加 w3,b3。实际: {sorted(names)}"
        )
    return weights, variant


def predict(x: np.ndarray, weights: dict, variant: str) -> np.ndarray:
    if variant == "wine":
        return np.argmax(forward_wine(x, **weights), axis=1)
    return np.argmax(
        forward_wine2(
            x,
            weights["w1"],
            weights["b1"],
            weights["w2"],
            weights["b2"],
            weights["w3"],
            weights["b3"],
        ),
        axis=1,
    )


def prepare_data(feature_cols: list[str], random_state: int = 42):
    df = pd.read_csv(TRAIN_CSV)
    x = df[feature_cols].to_numpy(dtype=np.float64)
    y_raw = df["quality"].to_numpy(dtype=np.int64)
    classes = np.sort(np.unique(y_raw))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_all = scaler.transform(x)
    return x_val, y_val, classes, idx_to_class, scaler, feature_cols, x_all, y


def plot_decision_slice(
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_plot: np.ndarray,
    y_plot: np.ndarray,
    classes: np.ndarray,
    idx_to_class: dict,
    feature_cols: list[str],
    weights: dict,
    variant: str,
    sample_idx: int,
    feat_i: int,
    feat_j: int,
    grid_points: int,
    out_path: Path,
    show_all_points: bool = True,
    slice_mode: str = "mean",
):
    """
    slice_mode mean: 未画出的维度固定在训练集 StandardScaler 后的均值（全 0），
    与「典型」标准化样本一致，减轻与全样本散点叠在一起的错位感。
    anchor: 其余维度固定在验证集 anchor 样本（旧行为）。
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
    pad_u = 0.05 * (u_max - u_min + 1e-6)
    pad_v = 0.05 * (v_max - v_min + 1e-6)
    u = np.linspace(u_min - pad_u, u_max + pad_u, grid_points)
    v = np.linspace(v_min - pad_v, v_max + pad_v, grid_points)
    uu, vv = np.meshgrid(u, v)

    grid_x = np.tile(slice_base, (grid_points * grid_points, 1))
    grid_x[:, feat_i] = uu.ravel()
    grid_x[:, feat_j] = vv.ravel()

    z = predict(grid_x, weights, variant).reshape(grid_points, grid_points)

    configure_matplotlib_font()

    n_cls = len(classes)
    level_edges = np.arange(n_cls + 1) - 0.5

    plt.figure(figsize=(8, 6))
    cf = plt.contourf(
        uu, vv, z, levels=level_edges, cmap="tab10", alpha=0.75, extend="both"
    )
    cb = plt.colorbar(cf, ticks=np.arange(n_cls))
    cb.ax.set_yticklabels([str(int(c)) for c in classes])
    # 使用英文标签，避免 DejaVu Sans 缺中文字形导致大量 UserWarning
    cb.set_label("Predicted class (quality)")

    if show_all_points:
        plt.scatter(
            x_plot[:, feat_i],
            x_plot[:, feat_j],
            c=y_plot,
            cmap="tab10",
            vmin=-0.5,
            vmax=n_cls - 0.5,
            s=18,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
            label=f"all samples (n={len(y_plot)}, true quality)",
        )

    y_true = y_val[sample_idx]
    ax_i = float(x_val[sample_idx, feat_i])
    ax_j = float(x_val[sample_idx, feat_j])
    plt.scatter(
        ax_i,
        ax_j,
        c="red",
        s=200,
        marker="*",
        edgecolors="k",
        linewidths=0.8,
        zorder=6,
        label=(
            f"val anchor #{sample_idx} (true quality={idx_to_class[int(y_true)]})"
        ),
    )
    plt.xlabel(f"{feature_cols[feat_i]} (standardized)")
    plt.ylabel(f"{feature_cols[feat_j]} (standardized)")
    if slice_mode == "mean":
        plt.title(f"Wine ({variant}) · mean slice · val #{sample_idx}")
    else:
        plt.title(f"Wine ({variant}) · anchor slice · val #{sample_idx}")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"已保存划分图: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="红酒模型离线测试 + 2D 决策切片图")
    parser.add_argument(
        "--variant",
        choices=["wine", "wine2"],
        default=None,
        help="与 train_wine / train_wine2 对应；默认根据权重 npz 自动判断",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="npz 权重路径，默认 checkpoints/wine_best.npz 或 wine2_best.npz",
    )
    parser.add_argument("--sample", type=int, default=0, help="验证集中用于切片的样本下标")
    parser.add_argument("--feat1", type=int, default=0, help="横轴特征列索引")
    parser.add_argument("--feat2", type=int, default=1, help="纵轴特征列索引")
    parser.add_argument("--grid", type=int, default=120, help="网格密度")
    parser.add_argument(
        "--no-val-points",
        action="store_true",
        help="不绘制样本散点（默认绘制全表经训练集 scaler 后的所有点）",
    )
    parser.add_argument(
        "--slice",
        choices=["mean", "anchor"],
        default="mean",
        help="mean: 背景在非画图维度取训练标准化均值0；anchor: 固定在验证锚点样本",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CHECKPOINTS / "wine_decision_slice.png",
        help="输出图像路径",
    )
    args = parser.parse_args()

    weights_path = args.weights or (CHECKPOINTS / "wine2_best.npz")
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"未找到权重文件: {weights_path}\n请先运行 train_wine.py 或 train_wine2.py 训练并保存。"
        )

    weights, variant = load_weights(weights_path)
    if args.variant is not None and args.variant != variant:
        raise ValueError(
            f"权重文件对应模型为 {variant}，与 --variant {args.variant} 不一致，请去掉 --variant 或改正。"
        )

    if variant == "wine":
        feature_cols = [c for c in pd.read_csv(TRAIN_CSV, nrows=0).columns if c != "quality"]
    else:
        feature_cols = [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ]

    print(f"已根据权重自动选择模型类型: {variant}")
    x_val, y_val, classes, idx_to_class, _, cols, x_all, y_all = prepare_data(
        feature_cols
    )

    y_pred = predict(x_val, weights, variant)
    y_val_label = np.array([idx_to_class[int(i)] for i in y_val])
    y_pred_label = np.array([idx_to_class[int(i)] for i in y_pred])
    acc = accuracy_score(y_val, y_pred)

    print(f"权重: {weights_path}")
    print(f"验证集样本数: {len(y_val)}")
    print(f"Accuracy: {acc:.4f}\n")
    print(classification_report(y_val_label, y_pred_label, labels=classes, digits=4, zero_division=0))

    if not args.no_val_points:
        print(
            "\n[关于决策图与散点]\n"
            "  背景色：只在两个坐标轴特征上变化；其余特征按 --slice 固定（mean=训练标准化后各维 0）。\n"
            "  散点：每个样本仍用其完整的 11 维，只把其中两维投到平面上；其余维与背景切片不必相同，\n"
            "        因此点与色块不必一一对应，这是高维分类在 2D 上的正常表现。\n"
            "  若仍想对比「单条样本的切片」，请使用：--slice anchor\n"
        )

    si = int(np.clip(args.sample, 0, len(y_val) - 1))
    plot_decision_slice(
        x_val,
        y_val,
        x_all,
        y_all,
        classes,
        idx_to_class,
        cols,
        weights,
        variant,
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
