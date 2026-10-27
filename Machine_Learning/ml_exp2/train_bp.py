from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_ROOT = Path("/Users/previous_rium/Desktop/PY/ml_exp2/datasets/iris")
TRAIN_CSV = DATA_ROOT / "train" / "iris.csv"

# 高斯标准化
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

# 独热编码
def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out

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

class model:
    def __init__(self, n_features, n_classes, hidden_dim=16, lr=0.05, epochs=200, seed=42):
        # BP 神经网络: input -> hidden(ReLU) -> softmax
        rng = np.random.RandomState(seed)
        self.w1 = rng.randn(n_features, hidden_dim) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros((1, hidden_dim), dtype=float)
        self.w2 = rng.randn(hidden_dim, n_classes) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, n_classes), dtype=float)
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def fit(self, x, y):
        n_classes = self.b2.shape[1]
        y_oh = one_hot(y, n_classes)
        best_loss = float("inf")
        best_epoch = -1
        best_w1 = self.w1.copy()
        best_b1 = self.b1.copy()
        best_w2 = self.w2.copy()
        best_b2 = self.b2.copy()

        for epoch in range(self.epochs):
            # forward
            z1 = x @ self.w1 + self.b1
            a1 = np.maximum(0.0, z1)  # ReLU
            logits = a1 @ self.w2 + self.b2
            probs = softmax(logits)
            eps = 1e-12
            loss = -np.mean(np.sum(y_oh * np.log(probs + eps), axis=1))
            self.losses.append(loss)

            # backward
            grad_logits = (probs - y_oh) / len(y)
            grad_w2 = a1.T @ grad_logits
            grad_b2 = np.sum(grad_logits, axis=0, keepdims=True)

            grad_a1 = grad_logits @ self.w2.T
            grad_z1 = grad_a1 * (z1 > 0).astype(float)
            grad_w1 = x.T @ grad_z1
            grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

            # update
            self.w2 -= self.lr * grad_w2
            self.b2 -= self.lr * grad_b2
            self.w1 -= self.lr * grad_w1
            self.b1 -= self.lr * grad_b1

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
                best_w1 = self.w1.copy()
                best_b1 = self.b1.copy()
                best_w2 = self.w2.copy()
                best_b2 = self.b2.copy()

        self.w1 = best_w1
        self.b1 = best_b1
        self.w2 = best_w2
        self.b2 = best_b2
        return best_loss, best_epoch

    def predict(self, x):
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(0.0, z1)
        logits = a1 @ self.w2 + self.b2
        return np.argmax(logits, axis=1)


def train_one_fold(x_train, y_train, x_val, y_val, epochs=200, lr=0.05, seed=42):
    clf = model(
        n_features=x_train.shape[1], n_classes=3, lr=lr, epochs=epochs, seed=seed
    )
    best_loss, best_epoch = clf.fit(x_train, y_train)

    save_path = Path("checkpoints") / "Iris.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, w1=clf.w1, b1=clf.b1, w2=clf.w2, b2=clf.b2)

    y_pred = clf.predict(x_val)
    acc, precision, recall, f1 = multiclass_metrics(y_val, y_pred, n_classes=3)
    return acc, precision, recall, f1, clf.losses, best_loss, best_epoch


def plot_loss_curve(losses, save_path):
    epochs = np.arange(1, len(losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BP Network Training Loss (Iris)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    np.random.seed(42)

    df = pd.read_csv(TRAIN_CSV)
    feature_cols = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    target_col = "Species"

    x = df[feature_cols].to_numpy(dtype=float)
    y = (
        df[target_col]
        .map({"setosa": 0, "versicolor": 1, "virginica": 2})
        .to_numpy(dtype=int)
    )

    # 使用分层 10 折中的第一折进行训练与验证
    train_idx, val_idx = next(indices(y, n_splits=10, seed=42))
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    x_train, mean, std = standardize_fit_transform(x_train)
    x_val = standardize_transform(x_val, mean, std)

    acc, precision, recall, f1, losses, best_loss, best_epoch = train_one_fold(
        x_train, y_train, x_val, y_val, epochs=200, lr=0.05, seed=42
    )

    curve_path = Path("checkpoints") / "Iris_loss_curve.png"
    plot_loss_curve(losses, curve_path)

    print("Validation results:")
    print(
        f"acc={acc:.4f}, precision(macro)={precision:.4f}, "
        f"recall(macro)={recall:.4f}, f1(macro)={f1:.4f}"
    )
    print(
        f"best_epoch={best_epoch}, best_train_loss={best_loss:.4f}, "
        f"loss_curve={curve_path}"
    )


if __name__ == "__main__":
    main()