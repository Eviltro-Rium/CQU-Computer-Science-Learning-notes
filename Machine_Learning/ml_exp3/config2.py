from pathlib import Path

_ROOT = Path(__file__).resolve().parent

DATASET_PATH = _ROOT / "datasets" / "Boston_house" / "house_data.csv"
CSV_ENCODING = "utf-8"
MAX_DEPTH = 5
MIN_SAMPLES_SPLIT = 2
PRINT_TREE = False
TREE_PNG_PATH =  _ROOT / "output" / "train2_tree.png"
TREE_DOT_PATH = None
SEED = 47
VAL_RATIO = 0.2