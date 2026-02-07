from pathlib import Path
DATA_DIR =  Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

COCO_IMG_DIR = RAW_DATA_DIR / "coco" / "val2017"
COCO_ANN_FILE = RAW_DATA_DIR / "coco" / "annotations" / "instances_val2017.json"

EASY_SOURCE_DIR = RAW_DATA_DIR / "easy"
MEDIUM_SOURCE_DIR = RAW_DATA_DIR / "medium"
HARD_SOURCE_DIR = RAW_DATA_DIR / "hard"

EASY_OUTPUT_DIR = PROCESSED_DATA_DIR / "easy"
MEDIUM_OUTPUT_DIR = PROCESSED_DATA_DIR / "medium"
HARD_OUTPUT_DIR = PROCESSED_DATA_DIR / "hard"

COLOR_MAP = {
    'red': 0,
    'orange': 15,
    'yellow': 30,
    'lime': 45,
    'green': 60,
    'cyan': 90,
    'blue': 120,
    'purple': 140,
    'pink': 160,
    'magenta': 170
}

# ID文件路径
EASY_IDS_DIR = RAW_DATA_DIR / "easy"
MEDIUM_IDS_DIR = RAW_DATA_DIR / "medium"
HARD_IDS_DIR = RAW_DATA_DIR / "hard"


