"""
基于 DBSCAN 的密度聚类实验脚本。
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

import config as cfg
from train_Kmeans import (
    encode_labels,
    evaluate_classification_from_clusters,
    label_mapping,
    load_data,
    preprocess,
    print_hungarian_class_to_clusters_table,
    print_per_class_metrics_table,
)


def suggest_eps_from_kdistance(X, min_samples, quantile=0.85):
    """
    用「到第 min_samples 个近邻的距离」的分位数，粗略估计可用的 eps 尺度（仅作调参提示）。
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 3:
        return None
    k = min(max(2, int(min_samples)), n - 1)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    dist, _ = nn.kneighbors(X)
    kdist = dist[:, -1]
    return float(np.quantile(kdist, quantile))


def _silhouette_on_non_noise(X, labels):
    """非噪声点上计算轮廓系数；不可算时返回 -inf。"""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    m = labels >= 0
    if np.count_nonzero(m) < 3:
        return float("-inf")
    lu = np.unique(labels[m])
    if len(lu) < 2:
        return float("-inf")
    try:
        return float(silhouette_score(X[m], labels[m], metric="euclidean"))
    except ValueError:
        return float("-inf")


def auto_eps_dbscan(
    X,
    min_samples,
    n_target_clusters,
    max_extra=6,
    min_non_noise_frac=0.35,
    sil_noise_trade=2.2,
):
    """
    沿 k-距离分位扫描 eps，收集簇数 ∈ [n_target, n_target+max_extra] 的候选。

    优先在「非噪声比例 ≥ min_non_noise_frac」的候选中，最大化
        adjusted = silhouette(非噪声) − sil_noise_trade × 噪声比例
    以避免极小 eps 带来极高轮廓、但训练点几乎全为噪声的无效解。

    若无满足比例约束的候选，则放宽为全体候选上最大化 adjusted；仍无则退回簇数最大的 eps。
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    k = min(max(2, int(min_samples)), n - 1)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    dist, _ = nn.kneighbors(X)
    kdist = dist[:, -1]
    if not np.any(np.isfinite(kdist)) or np.max(kdist) <= 0:
        return 0.1, 0

    best_eps, best_nc = float(np.quantile(kdist, 0.4)), -1
    candidates = []

    for q in np.linspace(0.95, 0.04, 46):
        eps = float(np.quantile(kdist, q))
        if eps <= 1e-15:
            continue
        lab = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(
            X
        )
        nc = len(set(lab.tolist())) - (1 if -1 in lab else 0)
        if nc == 0:
            continue
        if n_target_clusters <= nc <= n_target_clusters + max_extra:
            candidates.append((eps, lab, nc))
        if nc > best_nc:
            best_eps, best_nc = eps, nc

    if not candidates:
        return best_eps, best_nc

    def score_tuple(eps, lab, nc):
        sil = _silhouette_on_non_noise(X, lab)
        noise_frac = float(np.mean(lab == -1))
        nn_frac = 1.0 - noise_frac
        adjusted = sil - sil_noise_trade * noise_frac
        return sil, noise_frac, nn_frac, adjusted

    def pick_best(pool):
        best_t = None
        best_adj = None
        for item in pool:
            eps, lab, nc = item
            sil, noise_frac, nn_frac, adjusted = score_tuple(eps, lab, nc)
            if best_adj is None or adjusted > best_adj:
                best_adj = adjusted
                best_t = (eps, lab, nc, sil, noise_frac, nn_frac)
        return best_t

    min_nn = max(3, int(np.ceil(min_non_noise_frac * n)))
    decent = [
        c
        for c in candidates
        if int(np.sum(c[1] >= 0)) >= min_nn
    ]
    pool = decent if decent else candidates
    best_t = pick_best(pool)
    if best_t is None:
        return best_eps, best_nc
    eps, lab, nc, sil, noise_frac, nn_frac = best_t
    return eps, nc


def dbscan_predict_from_train(
    X_train,
    labels_train,
    X_query,
    eps,
    metric="euclidean",
    max_neighbors=32,
):
    """
    对每个查询点：在训练集上取至多 max_neighbors 个近邻，
    在距离 <= eps 且训练标签非噪声的点中做簇标签多数票；若无则 -1。
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_query = np.asarray(X_query, dtype=np.float64)
    labels_train = np.asarray(labels_train, dtype=np.int64)
    n_train = X_train.shape[0]
    n_q = X_query.shape[0]
    k = min(int(max_neighbors), n_train)
    if k < 1:
        return np.full(n_q, -1, dtype=np.int64)

    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X_train)
    dist, idx = nn.kneighbors(X_query)
    out = np.full(n_q, -1, dtype=np.int64)

    for i in range(n_q):
        in_ball = (dist[i] <= eps) & (labels_train[idx[i]] >= 0)
        if not np.any(in_ball):
            continue
        labs = labels_train[idx[i][in_ball]]
        uniq, counts = np.unique(labs, return_counts=True)
        out[i] = int(uniq[np.argmax(counts)])

    return out


def _dbscan_cluster_count(labels):
    labels = np.asarray(labels, dtype=np.int64)
    return len(set(labels.tolist())) - (1 if -1 in labels else 0)


def _align_with_noise_as_wrong(y_pred_cluster, cluster_to_class):
    """将非噪声簇映射为类别；噪声或未知簇保留为 -1，后续按错误计入。"""
    y_pred_cluster = np.asarray(y_pred_cluster, dtype=np.int64)
    y_pred_cls = np.full_like(y_pred_cluster, -1)
    for i, p in enumerate(y_pred_cluster):
        if int(p) in cluster_to_class:
            y_pred_cls[i] = int(cluster_to_class[int(p)])
    return y_pred_cls


def _classification_metrics(y_true, y_pred_cls, all_class_ids):
    """按给定类别集合计算 weighted 与逐类指标；-1 预测会被自然计为错误。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_cls = np.asarray(y_pred_cls, dtype=np.int64)
    labels = np.asarray(all_class_ids, dtype=np.int64)

    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true,
        y_pred_cls,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    p_c, r_c, f_c, sup_c = precision_recall_fscore_support(
        y_true,
        y_pred_cls,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class = [
        {
            "class": int(labels[i]),
            "precision": float(p_c[i]),
            "recall": float(r_c[i]),
            "f1": float(f_c[i]),
            "support": int(sup_c[i]),
        }
        for i in range(len(labels))
    ]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred_cls)),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "per_class": per_class,
    }


def tune_dbscan_with_validation(
    X_train,
    X_val,
    y_val,
    all_class_ids,
    min_samples_list,
    n_target_clusters,
    max_extra=6,
    max_val_noise_frac=0.15,
    max_train_noise_frac=0.35,
    max_neighbors=32,
):
    """
    用验证集标签做参数选择，目标是最大化「全样本 weighted F1（噪声按错误计入）」。

    该函数用于课程实验中的对照调参；若要保持严格无监督，请关闭
    config.DBSCAN_TUNE_WITH_VAL_LABELS。
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.int64)

    best, best_key = None, None
    fallback, fallback_key = None, None

    for min_samples in min_samples_list:
        min_samples = int(min_samples)
        if min_samples < 2:
            continue

        n = X_train.shape[0]
        k = min(max(2, min_samples), n - 1)
        if k < 2:
            continue

        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(X_train)
        dist, _ = nn.kneighbors(X_train)
        kdist = dist[:, -1]

        for q in np.linspace(0.95, 0.04, 46):
            eps = float(np.quantile(kdist, q))
            if eps <= 1e-15:
                continue

            y_pred_train = DBSCAN(
                eps=eps, min_samples=min_samples, metric="euclidean"
            ).fit_predict(X_train)
            n_clusters = _dbscan_cluster_count(y_pred_train)
            if not (n_target_clusters <= n_clusters <= n_target_clusters + max_extra):
                continue

            y_pred_val = dbscan_predict_from_train(
                X_train,
                y_pred_train,
                X_val,
                eps=eps,
                metric="euclidean",
                max_neighbors=max_neighbors,
            )
            non_noise = y_pred_val >= 0
            if not np.any(non_noise):
                continue

            subset_m = evaluate_classification_from_clusters(
                y_val[non_noise],
                y_pred_val[non_noise],
                centers=None,
                n_clusters=None,
            )
            y_pred_full_cls = _align_with_noise_as_wrong(
                y_pred_val, subset_m["cluster_to_class"]
            )
            full_m = _classification_metrics(y_val, y_pred_full_cls, all_class_ids)

            train_noise = float(np.mean(y_pred_train == -1))
            val_noise = float(np.mean(y_pred_val == -1))
            sil = _silhouette_on_non_noise(X_train, y_pred_train)

            key = (
                full_m["f1_weighted"],
                full_m["accuracy"],
                -val_noise,
                -train_noise,
                -abs(n_clusters - n_target_clusters),
                sil,
            )
            item = {
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "train_noise": train_noise,
                "val_noise": val_noise,
                "silhouette": sil,
                "metrics": full_m,
            }

            if fallback_key is None or key > fallback_key:
                fallback, fallback_key = item, key

            if train_noise > max_train_noise_frac or val_noise > max_val_noise_frac:
                continue
            if best_key is None or key > best_key:
                best, best_key = item, key

    return best if best is not None else fallback


def evaluate_and_print_dbscan_split(title, y_true, y_pred, id2str, all_ids):
    """统一输出 DBSCAN 的非噪声子集指标与全样本（噪声计错）指标。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    non_noise = y_pred >= 0
    n_total = len(y_true)
    n_non_noise = int(np.count_nonzero(non_noise))
    coverage = n_non_noise / n_total if n_total else 0.0

    print(f"\n{title}:")
    print(f"  非噪声覆盖率: {coverage:.6f} ({n_non_noise}/{n_total})")

    if n_non_noise == 0:
        print("  全部为噪声，跳过匈牙利与分类指标。")
        return

    subset_m = evaluate_classification_from_clusters(
        y_true[non_noise],
        y_pred[non_noise],
        centers=None,
        n_clusters=None,
    )
    print("  非噪声子集指标:")
    for k, v in subset_m.items():
        if k in ("cluster_to_class", "per_class"):
            continue
        print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

    y_pred_full_cls = _align_with_noise_as_wrong(
        y_pred, subset_m["cluster_to_class"]
    )
    full_m = _classification_metrics(y_true, y_pred_full_cls, all_ids)
    print("  全样本指标（噪声按错误计入）:")
    for k, v in full_m.items():
        if k == "per_class":
            continue
        print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

    print_hungarian_class_to_clusters_table(
        "匈牙利匹配（类 -> 编号）",
        subset_m["cluster_to_class"],
        id2str,
        all_class_ids=all_ids,
    )
    print_per_class_metrics_table(
        "各类别指标（全样本，噪声按错误计入）:", full_m["per_class"], id2str
    )


if __name__ == "__main__":
    PATH = cfg.DATASET_PATH
    _, X, y = load_data(PATH)

    str2id, id2str, _classes = label_mapping(y)
    y = encode_labels(y, str2id)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    X_train, X_val, y_train, y_val = preprocess(
        X, y, val_ratio=cfg.VAL_RATIO, seed=cfg.SEED
    )
    X_train64 = np.asarray(X_train, dtype=np.float64)
    X_val64 = np.asarray(X_val, dtype=np.float64)
    all_ids = sorted(str2id.values())

    min_samples = int(cfg.DBSCAN_MIN_SAMPLES)
    eps_auto = bool(getattr(cfg, "DBSCAN_EPS_AUTO", True))

    if bool(getattr(cfg, "DBSCAN_TUNE_WITH_VAL_LABELS", False)):
        max_extra = int(getattr(cfg, "DBSCAN_AUTO_MAX_EXTRA", 6))
        min_samples_list = getattr(
            cfg, "DBSCAN_TUNE_MIN_SAMPLES_LIST", [min_samples]
        )
        max_val_noise = float(getattr(cfg, "DBSCAN_TUNE_MAX_VAL_NOISE_FRAC", 0.15))
        max_train_noise = float(
            getattr(cfg, "DBSCAN_TUNE_MAX_TRAIN_NOISE_FRAC", 0.35)
        )
        val_nn = int(getattr(cfg, "DBSCAN_VAL_NEIGHBORS", 32))
        tuned = tune_dbscan_with_validation(
            X_train64,
            X_val64,
            y_val,
            all_ids,
            min_samples_list=min_samples_list,
            n_target_clusters=len(str2id),
            max_extra=max_extra,
            max_val_noise_frac=max_val_noise,
            max_train_noise_frac=max_train_noise,
            max_neighbors=val_nn,
        )
        if tuned is None:
            raise RuntimeError("DBSCAN 验证集调参没有找到可用参数，请放宽噪声比例或候选范围。")
        eps = float(tuned["eps"])
        min_samples = int(tuned["min_samples"])
        print(
            f"DBSCAN eps={eps:.4f}（验证集调参：全样本F1最优，噪声按错误计入）, "
            f"簇数={tuned['n_clusters']}, min_samples={min_samples}, "
            f"训练噪声率={tuned['train_noise']:.2%}, 验证噪声率={tuned['val_noise']:.2%} | SEED={cfg.SEED}"
        )
    elif eps_auto:
        max_extra = int(getattr(cfg, "DBSCAN_AUTO_MAX_EXTRA", 6))
        min_nn_f = float(getattr(cfg, "DBSCAN_AUTO_MIN_NON_NOISE_FRAC", 0.35))
        trade = float(getattr(cfg, "DBSCAN_SIL_NOISE_TRADE", 2.2))
        eps, nc_probe = auto_eps_dbscan(
            X_train64,
            min_samples,
            n_target_clusters=len(str2id),
            max_extra=max_extra,
            min_non_noise_frac=min_nn_f,
            sil_noise_trade=trade,
        )
        print(
            f"DBSCAN eps={eps:.4f}（自动：簇数∈[{len(str2id)},{len(str2id)+max_extra}]，"
            f"最大化 轮廓−{trade}×噪声率；非噪声≥{min_nn_f:.0%} 优先）, "
            f"簇数={nc_probe}, min_samples={min_samples} | SEED={cfg.SEED}"
        )
    else:
        eps = float(cfg.DBSCAN_EPS)
        print(
            f"DBSCAN eps={eps}, min_samples={min_samples} | 划分 SEED={cfg.SEED}"
        )

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    y_pred_train = db.fit_predict(X_train64)
    val_nn = int(getattr(cfg, "DBSCAN_VAL_NEIGHBORS", 32))
    y_pred_val = dbscan_predict_from_train(
        X_train64,
        y_pred_train,
        X_val64,
        eps=eps,
        metric="euclidean",
        max_neighbors=val_nn,
    )

    labels_set = set(y_pred_train.tolist())
    n_clusters_est = len(labels_set) - (1 if -1 in labels_set else 0)
    n_noise_tr = int(np.sum(y_pred_train == -1))
    n_noise_val = int(np.sum(y_pred_val == -1))

    print(
        "估计簇数(不含噪声):",
        n_clusters_est,
        "| 核心点数量:",
        len(db.core_sample_indices_),
        "| 训练噪声:",
        n_noise_tr,
        "/",
        len(y_train),
        "| 验证噪声(ε-球内多数票):",
        n_noise_val,
        "/",
        len(y_val),
    )

    sil_tr = _silhouette_on_non_noise(X_train64, y_pred_train)
    if sil_tr > float("-inf"):
        print(f"训练集(非噪声)轮廓系数: {sil_tr:.4f}")

    if n_clusters_est > 0 and n_clusters_est < len(str2id):
        print(
            f"说明: DBSCAN 当前簇数({n_clusters_est})少于标签类别数({len(str2id)})，"
            "多类会共享簇或无法在表中出现对应编号；细分结构可尝试略减小 eps（簇通常变多）。"
        )

    if n_clusters_est == 0:
        q_eps = suggest_eps_from_kdistance(X_train64, min_samples, quantile=0.85)
        if q_eps is not None:
            print(
                "提示: 当前参数下无核心点（全噪声）。可尝试增大 config.DBSCAN_EPS，"
                "或减小 DBSCAN_MIN_SAMPLES。参考 k-距离分位(约 85%) eps ≈",
                f"{q_eps:.3f}",
            )
        else:
            print(
                "提示: 当前参数下无核心点（全噪声）。请增大 DBSCAN_EPS 或减小 DBSCAN_MIN_SAMPLES。"
            )

    print("验证集样本数:", len(y_val))
    print("验证集真实标签 y_val:", y_val)
    print("验证集聚类标签 y_pred:", y_pred_val)

    evaluate_and_print_dbscan_split(
        "训练集（匈牙利匹配后）", y_train, y_pred_train, id2str, all_ids
    )
    evaluate_and_print_dbscan_split(
        "验证集（匈牙利匹配后）", y_val, y_pred_val, id2str, all_ids
    )
