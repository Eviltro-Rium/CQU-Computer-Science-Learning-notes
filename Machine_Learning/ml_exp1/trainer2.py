"""
使用含全部数据属性的样本（3.0.csv）西瓜分类学习器
离散 one-hot 部分用 0 填充后再预测
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("/Users/previous_rium/Desktop/PY/ml_exp1/dataset/watermelon")
TRAIN_CSV = DATA_ROOT / "train" / "3.0.csv"
VAL_CSV = DATA_ROOT / "val" / "table_4.2.csv"
TEST_CSV = DATA_ROOT / "test" / "4.0.csv"


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def standardize_fit_transform(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def standardize_transform(x, mean, std):
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    return (x - mean) / std_safe


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
    def __init__(self, n_features, lr=0.05, epochs=200):
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

            if (epoch + 1) % 50 == 0 or epoch == 0:
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

    cat_cols = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]
    num_cols = ["密度", "含糖率"]
    df_test = pd.read_csv(TEST_CSV, header=None, names=num_cols, encoding="utf-8")

    y_train = df_train["好瓜"].map({"是": 1, "否": 0}).values.astype(int)
    y_val = df_val["好坏"].map({"好瓜": 1, "坏瓜": 0}).values.astype(int)

    # 独热编码
    X_train_cat = pd.get_dummies(df_train[cat_cols], dtype=float)
    X_val_cat = pd.get_dummies(df_val[cat_cols], dtype=float)
    X_train_cat, X_val_cat = X_train_cat.align(
        X_val_cat, join="left", axis=1, fill_value=0.0
    )

    x_train_num, mean, std = standardize_fit_transform(df_train[num_cols].values.astype(float))
    x_val_num = standardize_transform(df_val[num_cols].values.astype(float), mean, std)
    x_test_num = standardize_transform(df_test[num_cols].values.astype(float), mean, std)

    X_train_df = pd.concat(
        [X_train_cat.reset_index(drop=True), pd.DataFrame(x_train_num)], axis=1
    )
    X_val_df = pd.concat(
        [X_val_cat.reset_index(drop=True), pd.DataFrame(x_val_num)], axis=1
    )
    # 4.0.csv 仅有数值特征，没有离散特征；离散 one-hot 部分用 0 填充保持维度一致
    X_test_cat = pd.DataFrame(0.0, index=df_test.index, columns=X_train_cat.columns)
    X_test_df = pd.concat(
        [X_test_cat.reset_index(drop=True), pd.DataFrame(x_test_num)], axis=1
    )

    X_train = X_train_df.values
    X_val = X_val_df.values
    X_test = X_test_df.values
    model = LogisticRegressionScratch(n_features=X_train.shape[1], lr=0.05, epochs=200)
    best_loss, best_epoch = model.fit(X_train, y_train)

    save_path = Path("checkpoints") / "watermelon2.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, w=model.w, b=model.b, mean=mean, std=std, columns=X_train_df.columns.values)
    print(f"最小训练损失: {best_loss:.6f} (epoch {best_epoch})")
    print(f"最佳权重已保存到: {save_path}")

    y_val_pred = model.predict(X_val)
    acc, precision, recall, f1 = binary_metrics(y_val, y_val_pred)
    print(f"验证集准确率: {acc:.4f}")
    print(f"验证集 Precision: {precision:.4f}")
    print(f"验证集 Recall: {recall:.4f}")
    print(f"验证集 F1: {f1:.4f}")

    test_probs = model.predict_proba(X_test)
    y_test_pred = (test_probs >= 0.5).astype(int)
    print("测试集样本数:", len(y_test_pred))
    print("测试集预测类别 (0=坏瓜, 1=好瓜):", y_test_pred)
    print("测试集预测概率 P(好瓜):", test_probs)

    print("参数维度:", len(model.w))
    print("截距:", model.b)



if __name__ == "__main__":
    main()
