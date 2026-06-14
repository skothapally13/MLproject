import logging
import os
from datetime import datetime


# ---- Date and Time ----
DATE = datetime.now().strftime("%m_%d_%Y")
TIME = datetime.now().strftime("%H_%M_%S")


# ---- Folder structure ----
LOGS_ROOT = "logs"
LOG_DIR = os.path.join(LOGS_ROOT, DATE)

os.makedirs(LOG_DIR, exist_ok=True)


# ---- Each run creates unique log file ----
LOG_FILE = f"run_{TIME}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)


# ---- Logging configuration ----
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] [ %(lineno)d ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
