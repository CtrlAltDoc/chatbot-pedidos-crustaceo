"""Configuración central: rutas y umbrales en un solo lugar."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
LOGS_DIR = ROOT / "logs"

INTENT_MODEL_PATH = MODELS_DIR / "intent_pipeline.joblib"
VISION_HEAD_PATH = MODELS_DIR / "vision_head.npz"

# Umbrales del agente (por debajo se pide aclaración en lugar de adivinar)
INTENT_THRESHOLD = float(os.getenv("INTENT_THRESHOLD", 0.40))
VISION_THRESHOLD = float(os.getenv("VISION_THRESHOLD", 0.90))  # 0.60 dejaba pasar un perro como cangreburger (79 %)

# Límites de entrada (manejo de errores)
MAX_TEXT_CHARS = 500
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}

API_HOST = "127.0.0.1"
API_PORT = int(os.getenv("API_PORT", 5000))


def load_menu() -> dict:
    with open(DATA_DIR / "menu.json", encoding="utf-8") as f:
        return json.load(f)
