from pathlib import Path

_ROOT = Path(__file__).resolve().parent

DATASET_PATH = _ROOT / "datasets" / "iris.csv"
# 簇个数 k；iris 三分类时常取 3
PARA_K = 9
# 训练/验证划分等（与 k-means 初始化无关，便于固定划分做对比）
SEED = 42
# k-means++ 初始中心、空簇重抽等；改此值可在同一划分下对比不同初始化
KMEANS_INIT_SEED = 514
VAL_RATIO = 0.1
MAX_ITER = 300
TOL = 1e-4

# DBSCAN（密度聚类）；特征已标准化，eps 为欧氏半径。
# DBSCAN_EPS_AUTO=True：在簇数合格候选上最大化 (轮廓系数 − SIL_NOISE_TRADE×噪声比例)，避免「高轮廓但几乎全噪声」。
DBSCAN_EPS_AUTO = True
DBSCAN_EPS = 0.55
DBSCAN_MIN_SAMPLES = 4
DBSCAN_AUTO_MAX_EXTRA = 6
DBSCAN_VAL_NEIGHBORS = 32
# 自动选 eps：非噪声样本至少占训练集比例；噪声惩罚系数（越大越排斥高噪声解）
DBSCAN_AUTO_MIN_NON_NOISE_FRAC = 0.35
DBSCAN_SIL_NOISE_TRADE = 2.2
# 为实验对比优化指标：使用验证集标签扫描 eps/min_samples（纯无监督实验可改为 False）
DBSCAN_TUNE_WITH_VAL_LABELS = True
DBSCAN_TUNE_MIN_SAMPLES_LIST = [3, 4, 5, 6, 7, 8]
DBSCAN_TUNE_MAX_VAL_NOISE_FRAC = 0.15
DBSCAN_TUNE_MAX_TRAIN_NOISE_FRAC = 0.35