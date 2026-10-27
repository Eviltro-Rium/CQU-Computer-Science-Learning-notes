"""供 train1.py 使用的数据集路径与 CSV 编码。"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

DATASET_PATH = _ROOT / "datasets" / "iris" / "iris.csv"
CSV_ENCODING = "utf-8"
MAX_DEPTH = 3
MIN_SAMPLES_SPLIT = 2
PRINT_TREE = False
# 保存树图：设为 None 不生成；可设为 _ROOT / "output" / "train1_tree.png"
TREE_PNG_PATH = _ROOT / "output" / "train1_tree.png"
# DOT 必须与 PNG 使用不同文件名，否则会覆盖 PNG 导致「图片损坏」
TREE_DOT_PATH = _ROOT / "output" / "train1_tree.dot"
SEED = 114
VAL_RATIO = 0.2