"""
鸢尾花三分类学习器
十折交叉验证，无测试
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_ROOT = Path("/Users/previous_rium/Desktop/PY/ml_exp1/dataset/iris")
TRAIN_CSV = DATA_ROOT / "train" / "iris.csv"


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


def stratified_kfold_indices(y, n_splits=10, random_state=42):
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    classes = np.unique(y)
    per_class_chunks = {}
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        per_class_chunks[c] = np.array_split(idx, n_splits)

    for fold in range(n_splits):
        val_idx = np.concatenate([per_class_chunks[c][fold] for c in classes])
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx, assume_unique=False)
        yield train_idx, val_idx


def multiclass_metrics(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    acc = np.trace(cm) / np.sum(cm)
    precisions = []
    recalls = []
    f1s = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return acc, float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


class SoftmaxRegressionScratch:
    def __init__(self, n_features, n_classes, lr=0.05, epochs=200):
        self.w = np.zeros((n_features, n_classes), dtype=float)
        self.b = np.zeros((1, n_classes), dtype=float)
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def fit(self, x, y):
        y_oh = one_hot(y, self.b.shape[1])
        best_loss = float("inf")
        best_epoch = -1
        best_w = self.w.copy()
        best_b = self.b.copy()

        for epoch in range(self.epochs):
            logits = x @ self.w + self.b
            probs = softmax(logits)
            eps = 1e-12
            loss = -np.mean(np.sum(y_oh * np.log(probs + eps), axis=1))
            self.losses.append(loss)

            grad_logits = (probs - y_oh) / len(y)
            grad_w = x.T @ grad_logits
            grad_b = np.sum(grad_logits, axis=0, keepdims=True)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
                best_w = self.w.copy()
                best_b = self.b.copy()

        self.w = best_w
        self.b = best_b
        return best_loss, best_epoch

    def predict(self, x):
        logits = x @ self.w + self.b
        return np.argmax(logits, axis=1)


def train_one_fold(x_train, y_train, x_val, y_val, epochs=200, lr=0.05):
    model = SoftmaxRegressionScratch(
        n_features=x_train.shape[1], n_classes=3, lr=lr, epochs=epochs
    )
    best_loss, best_epoch = model.fit(x_train, y_train)

    save_path = Path("checkpoints") / "Iris.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, w=model.w, b=model.b)

    y_pred = model.predict(x_val)
    acc, precision, recall, f1 = multiclass_metrics(y_val, y_pred, n_classes=3)
    return acc, precision, recall, f1, model.losses, best_loss, best_epoch


def main():
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8")

    cols = df[["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]].values.astype(float)
    labels = df["Species"].map({"setosa": 0, "versicolor": 1, "virginica": 2}).values.astype(int)

    fold_accs = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(
        stratified_kfold_indices(labels, n_splits=10, random_state=42), start=1
    ):
        X_train, X_val = cols[train_idx], cols[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        X_train, mean, std = standardize_fit_transform(X_train)
        X_val = standardize_transform(X_val, mean, std)

        acc, precision, recall, f1, train_losses, best_loss, best_epoch = train_one_fold(
            X_train, y_train, X_val, y_val, epochs=200, lr=0.05
        )
        fold_accs.append(acc)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)
        fold_losses.append(train_losses)
        print(
            f"Fold {fold:02d} | Acc: {acc:.4f} | "
            f"P(macro): {precision:.4f} | R(macro): {recall:.4f} | "
            f"F1(macro): {f1:.4f} | best_loss: {best_loss:.6f} (epoch {best_epoch})"
        )

    print("-" * 40)
    print("10-fold accuracies:", [round(a, 4) for a in fold_accs])
    print("10-fold precisions(macro):", [round(a, 4) for a in fold_precisions])
    print("10-fold recalls(macro):", [round(a, 4) for a in fold_recalls])
    print("10-fold F1s(macro):", [round(a, 4) for a in fold_f1s])
    print(f"Mean accuracy: {np.mean(fold_accs):.4f}")
    print(f"Std accuracy : {np.std(fold_accs):.4f}")
    print(f"Mean precision(macro): {np.mean(fold_precisions):.4f}")
    print(f"Mean recall(macro)   : {np.mean(fold_recalls):.4f}")
    print(f"Mean F1(macro)       : {np.mean(fold_f1s):.4f}")

    # 绘制 10 折平均训练损失曲线
    mean_loss_curve = np.mean(np.array(fold_losses), axis=0)
    epochs = np.arange(1, len(mean_loss_curve) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean_loss_curve, color="#1f77b4", linewidth=2)
    plt.title("Iris 10-Fold Mean Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = Path("/Users/previous_rium/Desktop/PY/ml_exp1/iris_mean_loss_curve.png")
    plt.savefig(out_path, dpi=160)
    print(f"Mean loss curve saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()