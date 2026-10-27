from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_ROOT = Path("/Users/previous_rium/Desktop/PY/ml_exp2/datasets/wine")
TRAIN_CSV = DATA_ROOT / "winequality-red.csv"

def softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


class BPMLP2:
    # 两隐层 BP: input -> 64(ReLU) -> 16(ReLU) -> softmax
    def __init__(self, input_dim, num_classes, lr=5e-4, epochs=200, seed=42):
        rng = np.random.RandomState(seed)
        self.w1 = rng.randn(input_dim, 64) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, 64), dtype=float)
        self.w2 = rng.randn(64, 16) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros((1, 16), dtype=float)
        self.w3 = rng.randn(16, num_classes) * np.sqrt(2.0 / 16)
        self.b3 = np.zeros((1, num_classes), dtype=float)
        self.lr = lr
        self.epochs = epochs

    def fit(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        class_weights=None,
        batch_size=64,
        patience=30,
        sample_probs=None,
    ):
        y_train_oh = one_hot(y_train, self.b3.shape[1])
        y_val_oh = one_hot(y_val, self.b3.shape[1])
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_epoch = -1
        no_improve_epochs = 0
        best_params = (
            self.w1.copy(),
            self.b1.copy(),
            self.w2.copy(),
            self.b2.copy(),
            self.w3.copy(),
            self.b3.copy(),
        )

        for epoch in range(self.epochs):
            if sample_probs is None:
                idx = np.random.permutation(len(y_train))
            else:
                idx = np.random.choice(
                    len(y_train), size=len(y_train), replace=True, p=sample_probs
                )

            epoch_loss_sum = 0.0
            eps = 1e-12
            for start in range(0, len(y_train), batch_size):
                batch_idx = idx[start : start + batch_size]
                xb = x_train[batch_idx]
                yb = y_train[batch_idx]
                yb_oh = y_train_oh[batch_idx]

                z1 = xb @ self.w1 + self.b1
                a1 = np.maximum(0.0, z1)
                z2 = a1 @ self.w2 + self.b2
                a2 = np.maximum(0.0, z2)
                logits = a2 @ self.w3 + self.b3
                probs = softmax(logits)

                sample_ce = -np.sum(yb_oh * np.log(probs + eps), axis=1)
                if class_weights is not None:
                    sample_ce = sample_ce * class_weights[yb]
                batch_loss = float(np.mean(sample_ce))
                epoch_loss_sum += batch_loss * len(batch_idx)

                grad_logits = (probs - yb_oh) / len(batch_idx)
                if class_weights is not None:
                    grad_logits = grad_logits * class_weights[yb][:, None]
                grad_w3 = a2.T @ grad_logits
                grad_b3 = np.sum(grad_logits, axis=0, keepdims=True)
                grad_a2 = grad_logits @ self.w3.T
                grad_z2 = grad_a2 * (z2 > 0).astype(float)
                grad_w2 = a1.T @ grad_z2
                grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
                grad_a1 = grad_z2 @ self.w2.T
                grad_z1 = grad_a1 * (z1 > 0).astype(float)
                grad_w1 = xb.T @ grad_z1
                grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

                self.w3 -= self.lr * grad_w3
                self.b3 -= self.lr * grad_b3
                self.w2 -= self.lr * grad_w2
                self.b2 -= self.lr * grad_b2
                self.w1 -= self.lr * grad_w1
                self.b1 -= self.lr * grad_b1

            train_loss = epoch_loss_sum / len(y_train)
            train_losses.append(float(train_loss))

            val_probs = self.predict_proba(x_val)
            val_loss = float(-np.mean(np.sum(y_val_oh * np.log(val_probs + eps), axis=1)))
            val_losses.append(val_loss)

            if (epoch + 1) % 20 == 0:
                train_probs = self.predict_proba(x_train)
                train_pred = np.argmax(train_probs, axis=1)
                val_pred = np.argmax(val_probs, axis=1)
                train_acc = np.mean(train_pred == y_train)
                val_acc = np.mean(val_pred == y_val)
                print(
                    f"Epoch {epoch + 1:03d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} "
                    f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                no_improve_epochs = 0
                best_params = (
                    self.w1.copy(),
                    self.b1.copy(),
                    self.w2.copy(),
                    self.b2.copy(),
                    self.w3.copy(),
                    self.b3.copy(),
                )
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}, no val_loss improvement for {patience} epochs."
                )
                break

        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = best_params
        return best_val_loss, best_epoch, train_losses, val_losses

    def predict_proba(self, x):
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(0.0, z2)
        logits = a2 @ self.w3 + self.b3
        return softmax(logits)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)


def save_loss_curve(train_losses, val_losses, save_path: Path):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Wine2 Training Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_once(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    epochs,
    lr=5e-3,
    batch_size=64,
    patience=30,
):
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    class_inv = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = np.power(class_inv, 0.5)
    class_weights = class_weights / class_weights.mean()
    class_weights = np.clip(class_weights, 0.7, 2.5)
    sample_weights = class_weights[y_train]
    sample_probs = sample_weights / sample_weights.sum()

    model = BPMLP2(
        input_dim=X_train.shape[1], num_classes=num_classes, lr=lr, epochs=epochs, seed=42
    )

    save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "wine2_best.npz"
    best_val_loss, best_epoch, train_losses, val_losses = model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        class_weights=class_weights,
        batch_size=batch_size,
        patience=patience,
        sample_probs=sample_probs,
    )
    np.savez(
        save_path,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        w3=model.w3,
        b3=model.b3,
    )
    save_loss_curve(train_losses, val_losses, save_dir / "wine2_loss_curve.png")

    final_val_pred = model.predict(X_val)

    val_acc = accuracy_score(y_val, final_val_pred)
    print(
        f"best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}, val_acc={val_acc:.4f}"
    )
    return val_acc, final_val_pred

def main():
    np.random.seed(42)

    df = pd.read_csv(TRAIN_CSV)
    feature_cols = ["fixed acidity", "volatile acidity",
                    "citric acid", "residual sugar",
                    "chlorides", "free sulfur dioxide",
                    "total sulfur dioxide", "density",
                    "pH", "sulphates", "alcohol"]
    target_col = "quality"

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df[target_col].to_numpy(dtype=np.int64)
    classes = np.sort(np.unique(y_raw))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    val_acc, val_pred = train_once(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes=len(classes),
        epochs=300,
        lr=5e-3,
        batch_size=64,
        patience=30,
    )

    y_val_label = np.array([idx_to_class[i] for i in y_val])
    y_pred_label = np.array([idx_to_class[i] for i in val_pred])

    print("\nValidation results:")
    print(f"Accuracy: {val_acc:.4f}")
    print("\nValidation classification report:")
    print(
        classification_report(
            y_val_label, y_pred_label, labels=classes, digits=4, zero_division=0
        )
    )


if __name__ == "__main__":
    main()

