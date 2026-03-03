from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CLUSTER_DIR = DATA_DIR / "cluster"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CLUSTER_IMG = PROJECT_ROOT / "llm"

SMS_CLEAN_PATH = PROCESSED_DIR / "sms_clean.csv"
