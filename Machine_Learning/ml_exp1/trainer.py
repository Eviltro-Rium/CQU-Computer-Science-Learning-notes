"""
使用只含数据属性的样本（3.0a.csv）西瓜分类学习器
"""

from pathlib import Path
import numpy as np
import pandas as pd


DATA_ROOT = Path("/Users/previous_rium/Desktop/PY/ml_exp1/dataset/watermelon")
TRAIN_CSV = DATA_ROOT / "train" / "3.0a.csv"
VAL_CSV = DATA_ROOT / "val" / "table_4.2.csv"
TEST_CSV = DATA_ROOT / "test" / "4.0.csv"

# 标准化
def standardize_fit_transform(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std

def standardize_transform(x, mean, std):
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    return (x - mean) / std_safe

# sigmoid 激活函数
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return acc, precision, recall, f1


class LogisticRegressionScratch:
    def __init__(self, n_features, lr=0.5, epochs=200):
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def fit(self, x, y):
        y = y.astype(float)
        best_loss = float("inf")
        best_epoch = -1
        best_w = self.w.copy()
        best_b = self.b
        for epoch in range(self.epochs):
            logits = x @ self.w + self.b
            probs = sigmoid(logits)
            eps = 1e-12
            loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
            self.losses.append(loss)

            grad_w = (x.T @ (probs - y)) / len(y)
            grad_b = np.mean(probs - y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
                best_w = self.w.copy()
                best_b = self.b

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"epoch: {epoch + 1}, loss: {loss:.6f}")

        self.w = best_w
        self.b = best_b
        return best_loss, best_epoch

    def predict_proba(self, x):
        return sigmoid(x @ self.w + self.b)

    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) >= threshold).astype(int)


def main():
    df_train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    df_val = pd.read_csv(VAL_CSV, encoding="utf-8")

    feature_cols = ["密度", "含糖率"]
    x_train = df_train[feature_cols].values.astype(float)
    y_train = df_train["好瓜"].values.astype(int)

    x_val = df_val[feature_cols].values.astype(float)
    y_val = df_val["好坏"].map({"好瓜": 1, "坏瓜": 0}).values.astype(int)

    df_test = pd.read_csv(TEST_CSV, header=None, names=feature_cols, encoding="utf-8")
    x_test = df_test[feature_cols].values.astype(float)

    x_train, mean, std = standardize_fit_transform(x_train)
    x_val = standardize_transform(x_val, mean, std)
    x_test = standardize_transform(x_test, mean, std)

    model = LogisticRegressionScratch(n_features=x_train.shape[1], lr=0.5, epochs=200)
    best_loss, best_epoch = model.fit(x_train, y_train)

    save_path = Path("checkpoints") / "watermelon1.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, w=model.w, b=model.b, mean=mean, std=std)
    print(f"最小训练损失: {best_loss:.6f} (epoch {best_epoch})")
    print(f"最佳权重已保存到: {save_path}")

    y_val_pred = model.predict(x_val)
    acc, precision, recall, f1 = binary_metrics(y_val, y_val_pred)
    print(f"验证集准确率: {acc:.4f}")
    print(f"验证集 Precision: {precision:.4f}")
    print(f"验证集 Recall: {recall:.4f}")
    print(f"验证集 F1: {f1:.4f}")

    probs = model.predict_proba(x_test)
    y_pred = (probs >= 0.5).astype(int)
    print("测试集样本数:", len(y_pred), "（4.0.csv 无标签，仅输出预测）")
    print("预测类别 (0=坏瓜, 1=好瓜):", y_pred)
    print("预测概率 P(好瓜):", probs)
    print("系数 (标准化后特征):", dict(zip(feature_cols, model.w)))
    print("截距:", model.b)


if __name__ == "__main__":
    main()
