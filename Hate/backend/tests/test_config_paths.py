import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config


HATE_DIR = Path(__file__).resolve().parents[2]
assert Path(Config.MODEL_PATH) == HATE_DIR / "backend" / "models" / "best_model.keras"
assert Path(Config.TOKENIZER_PATH) == HATE_DIR / "tokenizer.pkl"
assert Path(Config.LABELS_PATH) == HATE_DIR / "labels.pkl"
