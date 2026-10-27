import csv
from pathlib import Path

import config2 as cfg
import numpy as np
import tree_viz


def load_data(path, encoding="utf-8"):
    X, y = [], []
    with open(path, newline="", encoding=encoding) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)
        is_california = header[-1].strip() == "ocean_proximity"
        feat_slice = slice(0, -2) if is_california else slice(0, -1)
        label_col = -2 if is_california else -1

        for row in reader:
            if not row or len(row) < 2:
                continue
            if len(row) != n_cols:
                continue
            feats = row[feat_slice]
            lab = row[label_col].strip()
            if not lab or any(not str(v).strip() for v in feats):
                continue
            try:
                x_row = [float(v) for v in feats]
                y_val = float(lab)
            except ValueError:
                continue
            X.append(x_row)
            y.append(y_val)

    return header, X, y


def data_split(X, y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(X)
    perm = rng.permutation(n)

    n_val = max(1, min(int(round(n * val_ratio)), n - 1))
    val_mask = np.zeros(n, dtype=bool)
    val_mask[perm[:n_val]] = True

    X_train, X_val = X[~val_mask], X[val_mask]
    y_train, y_val = y[~val_mask], y[val_mask]
    return X_train, X_val, y_train, y_val


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-15:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


class Regressor:
    """CART 回归树：分裂准则为加权 MSE（等价于子集方差加权），叶子为 y 的均值。"""

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.root = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        n, d = X.shape
        var_y = float(np.var(y, ddof=0)) if n > 0 else 0.0

        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or n <= 1
            or var_y < 1e-12
        ):
            return {"type": "leaf", "value": float(np.mean(y))}

        best_cost = np.inf
        best = None

        for feat in range(d):
            order = np.argsort(X[:, feat])
            x_feat = X[order, feat]
            y_sorted = y[order]

            for j in range(n - 1):
                if x_feat[j] == x_feat[j + 1]: continue

                t = (x_feat[j] + x_feat[j + 1]) / 2.0
                left_y = y_sorted[: j + 1]
                right_y = y_sorted[j + 1 :]
                cost = (len(left_y) / n) * np.var(left_y, ddof=0) + (
                    len(right_y) / n
                ) * np.var(right_y, ddof=0)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best = (feat, t)

        if best is None or best_cost >= var_y - 1e-12:
            return {"type": "leaf", "value": float(np.mean(y))}

        feat, thres = best
        left_mask = X[:, feat] <= thres
        right_mask = ~left_mask
        return {
            "type": "internal",
            "feature": feat,
            "threshold": thres,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[right_mask], y[right_mask], depth + 1),
        }

    def _predict_one(self, x, node):
        if node["type"] == "leaf":
            return float(node["value"])
        j, t = node["feature"], node["threshold"]
        if x[j] <= t:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.root) for row in X], dtype=float)

    def print_tree(self, feature_names=None):
        if self.root is None:
            print("(empty tree)")
            return
        tree_viz.print_tree_pretty(
            self.root,
            feature_names=feature_names,
            id2str=None,
            regression=True,
            title="Regression tree",
        )


if __name__ == "__main__":
    data_dir = cfg.DATASET_PATH
    data_encoding = cfg.CSV_ENCODING
    val_ratio = cfg.VAL_RATIO
    seed = cfg.SEED

    header, X, y = load_data(str(data_dir), encoding=data_encoding)
    X_train, X_val, y_train, y_val = data_split(X, y, val_ratio=val_ratio, seed=seed)

    is_ca = header[-1].strip() == "ocean_proximity"
    feat_names = header[:-2] if is_ca else header[:-1]
    if len(feat_names) != X_train.shape[1]:
        feat_names = feat_names[: X_train.shape[1]]

    print(
        f"Dataset: {data_dir.name}, n_train={len(y_train)}, n_val={len(y_val)}, "
        f"n_features={X_train.shape[1]}\n"
    )

    model = Regressor(
        max_depth=cfg.MAX_DEPTH, min_samples_split=cfg.MIN_SAMPLES_SPLIT
    )
    model.fit(X_train, y_train)

    if cfg.PRINT_TREE:
        model.print_tree(feature_names=feat_names)

    png_path = getattr(cfg, "TREE_PNG_PATH", None)
    if png_path:
        p = Path(png_path) if not isinstance(png_path, Path) else png_path
        if tree_viz.save_tree_png_matplotlib(
            model.root,
            p,
            feature_names=feat_names,
            regression=True,
            title="Regression tree",
        ):
            print(f"Tree PNG saved: {p.resolve()}")
        else:
            print("Skip PNG: install matplotlib (`pip install matplotlib`).")

    dot_path = getattr(cfg, "TREE_DOT_PATH", None)
    if dot_path:
        d = tree_viz.resolve_dot_path(dot_path, png_path)
        tree_viz.save_tree_dot(
            model.root,
            d,
            feature_names=feat_names,
            regression=True,
            graph_name="train2_tree",
        )
        print(f"Tree DOT saved: {d.resolve()} (e.g. dot -Tpng {d.name} -o tree.png)")

    y_pred = model.predict(X_val)
    print(f"MAE:  {mae(y_val, y_pred):.4f}")
    print(f"RMSE: {rmse(y_val, y_pred):.4f}")
    print(f"R2:   {r2_score(y_val, y_pred):.4f}")
