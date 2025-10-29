
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent

# /media/amritesh/bytesviewhdd1/ai_stuff/ai_monk_lab/data/raw/vehicle-detection.v3i.yolov11"

# RAW DATA 
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "data/raw/vehicle_detection"))
DATA_YAML_PATH = Path(os.getenv("DATA_YAML_PATH", DATASET_ROOT / "data.yaml"))


# TRAINING PARAMS
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", 640))
YOLO_EPOCHS = int(os.getenv("YOLO_EPOCHS", 100))
YOLO_BATCH = int(os.getenv("YOLO_BATCH", 16))
YOLO_WORKERS = int(os.getenv("YOLO_WORKERS", 4))


# INFERENCE CONFIGURATION
CLASSES = {
    "bus": 0,
    "car": 1,
    "microbus": 2,
    "motorbike": 3,
    "pickup-van": 4,
    "truck": 5,
}
ID_TO_NAME = {v: k for k, v in CLASSES.items()}

MODEL_WEIGHTS_PATH = Path(os.getenv("MODEL_WEIGHTS_PATH", f"{BASE_DIR}/runs/detect/vehicle_detector/weights/best.pt"))
OUTPUT_DIR = Path("inference_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LOGGER SETUP

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "model_training.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),          # Console output
        logging.FileHandler(LOG_FILE),    # File output
    ],
)

LOGGER = logging.getLogger("model_training_logger")