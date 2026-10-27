import csv
from pathlib import Path
import config as cfg
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split


def _stratified_train_val_indices(y, val_ratio, seed):
    """
    分层划分下标：尽量使训练集、验证集都包含各标签（每类至少 2 个样本时才可实现）。
    若某类全局仅 1 条，该类只能全部进训练集，并返回 skipped 供提示。
    """
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    idx = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)

    try:
        train_idx, val_idx = train_test_split(
            idx,
            test_size=val_ratio,
            stratify=y,
            random_state=seed,
            shuffle=True,
        )
        return train_idx, val_idx, []
    except ValueError:
        pass

    train_list, val_list = [], []
    skipped = []
    for c in np.unique(y):
        ix = np.flatnonzero(y == c)
        rng.shuffle(ix)
        n_c = len(ix)
        if n_c < 2:
            train_list.extend(ix.tolist())
            skipped.append(int(c))
            continue
        n_val_c = max(1, int(round(n_c * val_ratio)))
        n_val_c = min(n_val_c, n_c - 1)
        val_list.extend(ix[:n_val_c].tolist())
        train_list.extend(ix[n_val_c:].tolist())

    if not val_list and n > 0:
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        return train_idx, val_idx, skipped

    return np.array(train_list, dtype=np.int64), np.array(val_list, dtype=np.int64), skipped

def load_data(path, newline='', encoding='utf-8'):
    X, y = [], []
    with open(path, newline=newline, encoding=encoding) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)

        skip_index_col = not header[0].strip()
        feat_slice = slice(1, -1) if skip_index_col else slice(0, -1)

        for row in reader:
            if not row or len(row) < 2:
                continue
            if len(row) != n_cols:
                continue
            X.append(row[feat_slice])
            y.append(row[-1].strip())

    return header, X, y

# 标签映射
def label_mapping(y):
    classes = sorted(set(y))
    # 返回字典
    str2id = {name: i for i, name in enumerate(classes)}
    id2str = {i: name for name, i in str2id.items()}
    return str2id, id2str, classes

def encode_labels(y, str2id):
    return [str2id[label] for label in y]

# 数据分割（分层：训练/验证尽量都含各类标签；每类至少 2 个样本才可分层）
def preprocess(X, y, val_ratio=0.1, seed=42):
    import warnings

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    train_idx, val_idx, skipped = _stratified_train_val_indices(y, val_ratio, seed)
    if skipped:
        warnings.warn(
            
            + str(skipped),
            UserWarning,
            stacklevel=2,
        )

    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    X_train, X_val = X[~val_mask], X[val_mask]
    y_train, y_val = y[~val_mask], y[val_mask]

    # 标准化
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma

    return X_train, X_val, y_train, y_val


class K_means:
    """k-means 聚类（k-means++ 初始化 + 欧氏距离）。"""

    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def _rng(self):
        return np.random.default_rng(self.random_state)

    def _init_centroids(self, X):
        rng = self._rng()
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=np.float64)
        idx = int(rng.integers(0, n_samples))
        centers[0] = X[idx]
        closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for j in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total <= 0:
                idx = int(rng.integers(0, n_samples))
            else:
                probs = closest_sq / total
                idx = int(rng.choice(n_samples, p=probs))
            centers[j] = X[idx]
            d = np.sum((X - centers[j]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, d)
        return centers

    @staticmethod
    def _assign_labels(X, centers):
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        return np.argmin(dist_sq, axis=1)

    @staticmethod
    def _inertia(X, labels, centers):
        c = centers[labels]
        return float(np.sum((X - c) ** 2))

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X 应为二维数组 (n_samples, n_features)")
        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError("样本数不能小于簇数 n_clusters")

        rng = self._rng()
        centers = self._init_centroids(X)
        labels = None

        for it in range(self.max_iter):
            labels = self._assign_labels(X, centers)
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    new_centers[k] = X[int(rng.integers(0, n_samples))]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift <= self.tol:
                self.n_iter_ = it + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = self._inertia(X, labels, centers)
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("请先调用 fit")
        X = np.asarray(X, dtype=np.float64)
        return self._assign_labels(X, self.cluster_centers_)


def _cluster_to_class_mapping(y_true, y_pred, n_clusters=None):
    """
    用匈牙利算法在列联表上最优匹配：簇 id -> 真实类别 id。
    若某簇在当前 y_pred 中未出现，则用 n_clusters 补全键。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_classes = np.unique(y_true)
    pred_clusters = np.unique(y_pred)
    M = contingency_matrix(y_true, y_pred).astype(np.float64)
    cost = M.max() - M
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[int(pred_clusters[c])] = int(true_classes[r])
    for p in pred_clusters:
        p = int(p)
        if p not in mapping:
            mask = y_pred == p
            if np.any(mask):
                mapping[p] = int(np.bincount(y_true[mask].astype(np.int64)).argmax())
            else:
                mapping[p] = int(true_classes[0])
    if n_clusters is not None:
        for j in range(int(n_clusters)):
            if j not in mapping:
                mapping[j] = int(true_classes[j % len(true_classes)])
    return mapping


def _align_pred_labels(y_pred, cluster_to_class):
    return np.array([cluster_to_class[int(p)] for p in np.asarray(y_pred)], dtype=np.int64)


def evaluate_classification_from_clusters(
    y_true, y_pred_cluster, centers=None, n_clusters=None
):
    """
    将聚类簇与真实类对齐后，输出 Accuracy、weighted 的 Precision / Recall / F1。
    另返回 per_class：按真实类别 id 的逐类 precision / recall / f1 / support。
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_cluster = np.asarray(y_pred_cluster, dtype=np.int64)
    class_list = np.sort(np.unique(y_true))

    if n_clusters is None and centers is not None:
        n_clusters = centers.shape[0]

    cmap = _cluster_to_class_mapping(
        y_true, y_pred_cluster, n_clusters=n_clusters
    )
    y_pred_cls = _align_pred_labels(y_pred_cluster, cmap)

    acc = float(accuracy_score(y_true, y_pred_cls))
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred_cls, average="weighted", zero_division=0
    )

    p_c, r_c, f_c, sup_c = precision_recall_fscore_support(
        y_true,
        y_pred_cls,
        labels=class_list,
        average=None,
        zero_division=0,
    )
    per_class = [
        {
            "class": int(class_list[i]),
            "precision": float(p_c[i]),
            "recall": float(r_c[i]),
            "f1": float(f_c[i]),
            "support": int(sup_c[i]),
        }
        for i in range(len(class_list))
    ]

    return {
        "accuracy": acc,
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "cluster_to_class": cmap,
        "per_class": per_class,
    }


def _invert_cluster_to_class(cluster_to_class):
    """簇 id -> 类 id 转为 类 id -> [簇 id, ...]。"""
    inv = {}
    for cluster_id, class_id in cluster_to_class.items():
        inv.setdefault(int(class_id), []).append(int(cluster_id))
    for v in inv.values():
        v.sort()
    return inv


def _class_to_clusters_from_mapping(cluster_to_class):
    """cluster_to_class: 簇 id -> 类 id。返回 [(class_id, [cluster_id, ...]), ...] 按 class_id 排序。"""
    if not cluster_to_class:
        return []
    inv = _invert_cluster_to_class(cluster_to_class)
    return [(cid, inv[cid]) for cid in sorted(inv.keys())]


def print_hungarian_class_to_clusters_table(
    title, cluster_to_class, id2str=None, all_class_ids=None
):
    """匈牙利匹配结果：每个真实类对应哪些聚类编号（-1 显示为「噪声」）。

    all_class_ids: 若给定（如 sorted(str2id.values())），则列出全部类别；
    未被任何簇匹配到的类在「对应编号」列显示 —。
    """
    print(title)
    if not cluster_to_class:
        print("  (无)")
        return
    inv = _invert_cluster_to_class(cluster_to_class)
    if all_class_ids is not None:
        class_order = [int(c) for c in sorted(all_class_ids)]
    else:
        class_order = sorted(inv.keys())
    name_w = 22
    print(f"  {'class':>6} | {'类名(原始)':<{name_w}} | 对应编号")
    dash_len = 6 + 3 + name_w + 3 + 48
    print("  " + "-" * dash_len)
    for class_id in class_order:
        label = (
            str(id2str.get(class_id, class_id))[:name_w]
            if id2str is not None
            else str(class_id)
        )
        if class_id not in inv:
            clusters_str = "—"
        else:
            clusters_str = "，".join(
                "噪声" if int(c) == -1 else str(c) for c in inv[class_id]
            )
        print(f"  {class_id:>6} | {label:<{name_w}} | {clusters_str}")


def print_per_class_metrics_table(title, per_class, id2str=None):
    """按类别打印 precision / recall / f1 / support 表。"""
    print(title)
    name_w = 22
    row_fmt = (
        "  {:>6} | {:<"
        + str(name_w)
        + "} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>8}"
    )
    sep = "  " + "-" * (6 + 3 + name_w + 3 + 10 + 3 + 10 + 3 + 10 + 3 + 8)
    print(
        f"  {'class':>6} | {'name':<{name_w}} | {'precision':>10} | {'recall':>10} | {'f1':>10} | {'support':>8}"
    )
    print(sep)
    for row in per_class:
        cid = row["class"]
        name = (
            str(id2str.get(cid, cid))[:name_w]
            if id2str is not None
            else str(cid)
        )
        print(
            row_fmt.format(
                cid,
                name,
                row["precision"],
                row["recall"],
                row["f1"],
                row["support"],
            )
        )


if __name__ == "__main__":
    PATH = cfg.DATASET_PATH
    _, X, y = load_data(PATH)

    str2id, id2str, classes = label_mapping(y)
    y = encode_labels(y, str2id)

    X = np.asarray(X, dtype=np.float16)
    y = np.asarray(y, dtype=np.int64)

    X_train, X_val, y_train, y_val = preprocess(
        X, y, val_ratio=cfg.VAL_RATIO, seed=cfg.SEED
    )

    X_train64 = X_train.astype(np.float64)
    X_val64 = X_val.astype(np.float64)

    print(
        "实验种子: 划分 SEED =",
        cfg.SEED,
        "| k-means 初始化 KMEANS_INIT_SEED =",
        cfg.KMEANS_INIT_SEED,
    )

    km = K_means(
        n_clusters=cfg.PARA_K,
        max_iter=cfg.MAX_ITER,
        tol=cfg.TOL,
        random_state=cfg.KMEANS_INIT_SEED,
    )
    km.fit(X_train64)
    print("train inertia:", km.inertia_, "n_iter:", km.n_iter_)

    y_pred_train = km.labels_
    y_pred_val = km.predict(X_val64)

    print("验证集样本数:", len(y_val))
    print("验证集真实标签 y_val:", y_val)
    print("验证集聚类标签 y_pred:", y_pred_val)

    train_m = evaluate_classification_from_clusters(
        y_train, y_pred_train, centers=km.cluster_centers_
    )
    val_m = evaluate_classification_from_clusters(
        y_val, y_pred_val, centers=km.cluster_centers_
    )
    print("\n训练集（匈牙利匹配后）:")
    for k, v in train_m.items():
        if k in ("cluster_to_class", "per_class"):
            continue
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    all_ids = sorted(str2id.values())
    print_hungarian_class_to_clusters_table(
        "匈牙利匹配（类 -> 编号）", train_m["cluster_to_class"], id2str, all_class_ids=all_ids
    )
    print_per_class_metrics_table(
        "训练集 各类别指标:", train_m["per_class"], id2str
    )

    print("\n验证集（匈牙利匹配后）:")
    for k, v in val_m.items():
        if k in ("cluster_to_class", "per_class"):
            continue
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print_hungarian_class_to_clusters_table(
        "匈牙利匹配（类 -> 编号）", val_m["cluster_to_class"], id2str, all_class_ids=all_ids
    )
    print_per_class_metrics_table(
        "验证集 各类别指标:", val_m["per_class"], id2str
    )