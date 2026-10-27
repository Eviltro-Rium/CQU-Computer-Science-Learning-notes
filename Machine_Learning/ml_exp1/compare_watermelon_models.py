from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path("/Users/previous_rium/Desktop/PY/ml_exp1/dataset/watermelon")
TRAIN_30A_CSV = DATA_ROOT / "train" / "3.0a.csv"
TRAIN_30_CSV = DATA_ROOT / "train" / "3.0.csv"
VAL_CSV = DATA_ROOT / "val" / "table_4.2.csv"
TEST_CSV = DATA_ROOT / "test" / "4.0.csv"

CKPT_DIR = Path("/Users/previous_rium/Desktop/PY/ml_exp1/checkpoints")
MODEL1_CKPT = CKPT_DIR / "watermelon1.npz"  # trainer.py (2 特征)
MODEL2_CKPT = CKPT_DIR / "watermelon2.npz"  # trainer2.py (全特征 one-hot + 数值)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _predict_binary(params: dict, x_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = sigmoid(x_np @ params["w"] + params["b"]).ravel()
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def prepare_for_model1():
    feature_cols = ["密度", "含糖率"]
    df_train = pd.read_csv(TRAIN_30A_CSV, encoding="utf-8")
    df_val = pd.read_csv(VAL_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, header=None, names=feature_cols, encoding="utf-8")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(df_train[feature_cols].values)
    x_val = scaler.transform(df_val[feature_cols].values)
    x_test = scaler.transform(df_test[feature_cols].values)
    y_val = df_val["好坏"].map({"好瓜": 1, "坏瓜": 0}).values.astype(int)
    _ = x_train  # 仅用于拟合 scaler
    return x_val, y_val, x_test


def prepare_for_model2():
    cat_cols = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]
    num_cols = ["密度", "含糖率"]

    df_train = pd.read_csv(TRAIN_30_CSV, encoding="utf-8")
    df_val = pd.read_csv(VAL_CSV, encoding="utf-8")
    df_test = pd.read_csv(TEST_CSV, header=None, names=num_cols, encoding="utf-8")

    x_train_cat = pd.get_dummies(df_train[cat_cols], dtype=float)
    x_val_cat = pd.get_dummies(df_val[cat_cols], dtype=float)
    x_train_cat, x_val_cat = x_train_cat.align(
        x_val_cat, join="left", axis=1, fill_value=0.0
    )

    # 测试集缺少离散特征，按全 0 one-hot 填充以便输入维度对齐
    x_test_cat = pd.DataFrame(0.0, index=df_test.index, columns=x_train_cat.columns)

    scaler = StandardScaler()
    x_train_num = scaler.fit_transform(df_train[num_cols].values)
    x_val_num = scaler.transform(df_val[num_cols].values)
    x_test_num = scaler.transform(df_test[num_cols].values)

    x_train_df = pd.concat(
        [x_train_cat.reset_index(drop=True), pd.DataFrame(x_train_num)], axis=1
    )
    x_val_df = pd.concat(
        [x_val_cat.reset_index(drop=True), pd.DataFrame(x_val_num)], axis=1
    )
    x_test_df = pd.concat(
        [x_test_cat.reset_index(drop=True), pd.DataFrame(x_test_num)], axis=1
    )

    y_val = df_val["好坏"].map({"好瓜": 1, "坏瓜": 0}).values.astype(int)
    _ = x_train_df  # 仅用于保持与训练时一致的预处理流程
    return x_val_df.values, y_val, x_test_df.values


def load_model(ckpt_path: Path) -> dict:
    state = np.load(ckpt_path, allow_pickle=True)
    return {"w": state["w"], "b": state["b"]}


def main():
    missing = [p for p in [MODEL1_CKPT, MODEL2_CKPT] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoint files: {[str(p) for p in missing]}. "
            "Run trainer.py and trainer2.py first."
        )

    x1_val, y_val, x1_test = prepare_for_model1()
    x2_val, y2_val, x2_test = prepare_for_model2()

    if not np.array_equal(y_val, y2_val):
        raise ValueError("Validation labels from the two preprocessing pipelines do not match.")

    model1 = load_model(MODEL1_CKPT)
    model2 = load_model(MODEL2_CKPT)

    val_pred1, val_prob1 = _predict_binary(model1, x1_val)
    val_pred2, val_prob2 = _predict_binary(model2, x2_val)
    test_pred1, test_prob1 = _predict_binary(model1, x1_test)
    test_pred2, test_prob2 = _predict_binary(model2, x2_test)

    val_acc1 = accuracy_score(y_val, val_pred1)
    val_acc2 = accuracy_score(y_val, val_pred2)
    val_auc1 = roc_auc_score(y_val, val_prob1)
    val_auc2 = roc_auc_score(y_val, val_prob2)
    val_pr_auc1 = average_precision_score(y_val, val_prob1)
    val_pr_auc2 = average_precision_score(y_val, val_prob2)
    val_f1_1 = f1_score(y_val, val_pred1)
    val_f1_2 = f1_score(y_val, val_pred2)
    val_prec1 = precision_score(y_val, val_pred1)
    val_prec2 = precision_score(y_val, val_pred2)
    val_rec1 = recall_score(y_val, val_pred1)
    val_rec2 = recall_score(y_val, val_pred2)

    print("=== Validation Comparison (with labels) ===")
    print(f"Model1 (3.0a, 2 features) Validation Accuracy: {val_acc1:.4f}")
    print(f"Model2 (3.0, full features) Validation Accuracy: {val_acc2:.4f}")
    print(f"Model1 Validation AUC: {val_auc1:.4f}")
    print(f"Model2 Validation AUC: {val_auc2:.4f}")
    print(f"Model1 Validation PR-AUC: {val_pr_auc1:.4f}")
    print(f"Model2 Validation PR-AUC: {val_pr_auc2:.4f}")
    print(f"Model1 Validation F1: {val_f1_1:.4f}")
    print(f"Model2 Validation F1: {val_f1_2:.4f}")
    print(f"Model1 Validation Precision: {val_prec1:.4f}")
    print(f"Model2 Validation Precision: {val_prec2:.4f}")
    print(f"Model1 Validation Recall: {val_rec1:.4f}")
    print(f"Model2 Validation Recall: {val_rec2:.4f}")
    print("=== Test-set Visualization Stats (no labels) ===")
    print(f"Model1 Test Positive Ratio: {test_pred1.mean():.4f}")
    print(f"Model2 Test Positive Ratio: {test_pred2.mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(["Model1", "Model2"], [val_acc1, val_acc2], color=["#4C72B0", "#55A868"])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Validation Accuracy")
    axes[0].set_ylabel("Accuracy")

    axes[1].bar(
        ["Model1", "Model2"],
        [val_auc1, val_auc2],
        color=["#4C72B0", "#55A868"],
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Validation ROC-AUC")
    axes[1].set_ylabel("AUC")

    fig.suptitle("Watermelon Model Comparison")
    fig.tight_layout()

    out_path = Path("/Users/previous_rium/Desktop/PY/ml_exp1/watermelon_model_compare.png")
    fig.savefig(out_path, dpi=160)
    print(f"Saved comparison figure: {out_path}")

    # 额外输出测试集可视化（概率分布 + 预测类别占比）
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0.0, 1.0, 11)
    axes2[0].hist(
        test_prob1, bins=bins, alpha=0.6, label="Model1", color="#4C72B0", edgecolor="white"
    )
    axes2[0].hist(
        test_prob2, bins=bins, alpha=0.6, label="Model2", color="#55A868", edgecolor="white"
    )
    axes2[0].set_title("Test Probability Distribution")
    axes2[0].set_xlabel("P(good melon)")
    axes2[0].set_ylabel("Count")
    axes2[0].legend()

    test_pos_ratio1 = float(test_pred1.mean())
    test_pos_ratio2 = float(test_pred2.mean())
    axes2[1].bar(
        ["Model1", "Model2"],
        [test_pos_ratio1, test_pos_ratio2],
        color=["#4C72B0", "#55A868"],
    )
    axes2[1].set_ylim(0.0, 1.05)
    axes2[1].set_title("Test Predicted Positive Ratio")
    axes2[1].set_ylabel("Ratio")

    fig2.suptitle("Watermelon Test Set Visualization (No Labels)")
    fig2.tight_layout()

    out_path2 = Path("/Users/previous_rium/Desktop/PY/ml_exp1/watermelon_test_visualization.png")
    fig2.savefig(out_path2, dpi=160)
    print(f"Saved test visualization figure: {out_path2}")
    plt.show()


if __name__ == "__main__":
    main()
