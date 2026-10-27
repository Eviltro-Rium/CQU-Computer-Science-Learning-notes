import csv
import config1 as cfg
import numpy as np
from collections import Counter

# 数据加载：表头与特征列 slice 在循环外只算一次，循环内不再分支判断布局
def load_data(path, encoding="utf-8"):
    X, y = [], []
    with open(path, newline="", encoding=encoding) as f:
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

# 混淆矩阵：cm[i][j] = 真实类别为 labels[i]、预测为 labels[j] 的样本数
def confusion_matrix(y_true, y_pred, labels):
    idx = {int(c): k for k, c in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for a, b in zip(y_true, y_pred):
        cm[idx[int(a)]][idx[int(b)]] += 1
    return cm


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def classification_metrics(y_true, y_pred, labels):
    """
    多分类：返回每类 precision / recall / f1 / support，以及 macro / weighted 平均。
    labels 为类别 id 列表，顺序决定混淆矩阵行列顺序。
    """
    cm = confusion_matrix(y_true, y_pred, labels)
    n = len(labels)
    per_class = []
    precisions, recalls, f1s, supports = [], [], [], []

    for i in range(n):
        tp = cm[i][i]
        fp = sum(cm[k][i] for k in range(n) if k != i)
        fn = sum(cm[i][j] for j in range(n) if j != i)
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class.append(
            {
                "label": labels[i],
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
            }
        )
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

    total = sum(supports)
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    if total > 0:
        w_p = sum(p * s for p, s in zip(precisions, supports)) / total
        w_r = sum(r * s for r, s in zip(recalls, supports)) / total
        w_f1 = sum(f * s for f, s in zip(f1s, supports)) / total
    else:
        w_p = w_r = w_f1 = 0.0

    micro_f1 = accuracy_score(y_true, y_pred)

    return {
        "confusion_matrix": cm,
        "per_class": per_class,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": float(w_p),
        "weighted_recall": float(w_r),
        "weighted_f1": float(w_f1),
        "micro_f1": micro_f1,
    }


def print_classification_report(metrics, id2str=None):
    """打印混淆矩阵与分类报告。"""
    cm = metrics["confusion_matrix"]
    labels = [row["label"] for row in metrics["per_class"]]

    def name(c):
        if id2str is not None:
            try:
                return str(id2str.get(int(c), c))
            except (TypeError, ValueError):
                return str(c)
        return str(c)

    print("Confusion matrix (rows=true, cols=pred):")
    header = " " * 8 + "".join(f"{name(c):>12}" for c in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"{name(labels[i]):>8}" + "".join(f"{v:>12}" for v in row))
    print()

    print(f"{'class':>16}  {'precision':>10}  {'recall':>10}  {'f1':>10}  {'support':>8}")
    for row in metrics["per_class"]:
        c = row["label"]
        print(
            f"{name(c):>16}  {row['precision']:>10.4f}  {row['recall']:>10.4f}  "
            f"{row['f1']:>10.4f}  {row['support']:>8}"
        )
    print(
        f"{'macro avg':>16}  {metrics['macro_precision']:>10.4f}  "
        f"{metrics['macro_recall']:>10.4f}  {metrics['macro_f1']:>10.4f}  "
        f"{sum(r['support'] for r in metrics['per_class']):>8}"
    )
    print(
        f"{'weighted avg':>16}  {metrics['weighted_precision']:>10.4f}  "
        f"{metrics['weighted_recall']:>10.4f}  {metrics['weighted_f1']:>10.4f}  "
        f"{sum(r['support'] for r in metrics['per_class']):>8}"
    )
    print(
        f"{'micro F1 (=acc)':>16}  {'':>10}  {'':>10}  {metrics['micro_f1']:>10.4f}  "
        f"{sum(r['support'] for r in metrics['per_class']):>8}"
    )


# 数据分割
def data_split(X, y, val_ratio=0.2,seed=42):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(X)
    perm = rng.permutation(n)

    n_val = max(1, int(n * val_ratio))
    val_mask = np.zeros(n, dtype=bool)
    val_mask[perm[:n_val]] = True

    X_train, X_val = X[~val_mask], X[val_mask]
    y_train, y_val = y[~val_mask], y[val_mask]
    return X_train, X_val, y_train, y_val

# 基尼指数
def JiNiTaiMei(y):
    n = len(y)
    if n == 0:
        return 0.0

    c = Counter(y)
    return 1.0 - sum((cnt/n)**2 for cnt in c.values())

# 找主类
def majority_class(y):
    return Counter(y).most_common(1)[0][0]

"""CART分类器"""
class Classifier:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

        
    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = int)
        self.root = self._build(X,y,depth=0)

    def _build(self, X, y, depth):
        n, d = X.shape
        # 停止分裂：太深、样本太少、或已纯一类
        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or len(set(y)) == 1
        ):
            return {"type": "leaf", "value": majority_class(y)}
            # 用字典表示节点

        best_gain = 0.0
        base = JiNiTaiMei(y)
        best = None  # 最优分裂保存为 (特征列下标, 阈值)

        for feat in range(d):
            # 对特征进行排序
            order = np.argsort(X[:, feat])
            x_feat = X[order, feat]
            y_sorted = y[order]
            # 对特征进行分裂
            for j in range(n - 1):
                if x_feat[j] == x_feat[j + 1]:
                    continue

                t = (x_feat[j] + x_feat[j + 1]) / 2.0
                left_y = y_sorted[: j + 1]
                right_y = y_sorted[j + 1 :]
                g = (len(left_y) / n) * JiNiTaiMei(left_y) + \
                    (len(right_y) / n) * JiNiTaiMei(right_y)
                gain = base - g
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (feat, t)

        if best is None or best_gain <= 1e-12:
            return {"type": "leaf", "value": majority_class(y)}

        feat, t = best
        left_mask = X[:, feat] <= t
        right_mask = ~left_mask
        left_child = self._build(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build(X[right_mask], y[right_mask], depth + 1)
        return {
            "type": "internal",
            "feature": feat,
            "threshold": t,
            "left": left_child,
            "right": right_child,
        }

    def _predict_one(self, x, node):
        if node["type"] == "leaf":
            return node["value"]
        j, t = node["feature"], node["threshold"]
        if x[j] <= t:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.root) for row in X])

    def print_tree(self, feature_names=None, id2str=None):
        """递归打印决策树；feature_names 为与列下标对齐的特征名列表，id2str 用于把类别 id 还原成原名。"""
        if self.root is None:
            print("(tree is empty — call fit first)")
            return

        def fmt_class(v):
            if id2str is not None:
                try:
                    return str(id2str.get(int(v), v))
                except (TypeError, ValueError):
                    return str(v)
            return str(v)

        def walk(node, indent):
            pad = "  " * indent
            if node["type"] == "leaf":
                print(f"{pad}=> class = {fmt_class(node['value'])}")
                return
            j, t = node["feature"], node["threshold"]
            if feature_names is not None and 0 <= j < len(feature_names):
                fname = str(feature_names[j])
            else:
                fname = f"x[{j}]"
            print(f"{pad}{fname} <= {t:.6g} ?")
            print(f"{pad}  +-- yes")
            walk(node["left"], indent + 1)
            print(f"{pad}  +-- no")
            walk(node["right"], indent + 1)

        walk(self.root, 0)


if __name__ == "__main__":
    data_dir = cfg.DATASET_PATH
    data_encoding = cfg.CSV_ENCODING

    header, X, y = load_data(
        str(data_dir), encoding=data_encoding
    )
    str2id, id2str, classes = label_mapping(y)
    y = encode_labels(y, str2id)

    val_ratio = cfg.VAL_RATIO
    SEED = cfg.SEED
    X_train, X_val, y_train, y_val = data_split(X, y, val_ratio=val_ratio, seed=SEED)


    X_train = np.array(X_train, dtype=float)
    X_val = np.array(X_val, dtype=float)
    y_train = np.array(y_train, dtype=int)
    y_val = np.array(y_val, dtype=int)

    max_depth = cfg.MAX_DEPTH
    min_samples_split = cfg.MIN_SAMPLES_SPLIT

    model = Classifier(max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)

    if header[0].strip() == "":
        feat_names = header[1:-1]
    else:
        feat_names = header[:-1]
    if len(feat_names) != X_train.shape[1]:
        feat_names = feat_names[: X_train.shape[1]]

    if cfg.PRINT_TREE:
        print("Decision tree:")
        model.print_tree(feature_names=feat_names, id2str=id2str)
        print()

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    label_ids = list(range(len(classes)))
    m = classification_metrics(y_val, y_pred, label_ids)
    print()
    print_classification_report(m, id2str=id2str)