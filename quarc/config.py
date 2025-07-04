import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

STAGE1_DIR = PROCESSED_DATA_DIR / "stage1"
STAGE2_DIR = PROCESSED_DATA_DIR / "stage2"
STAGE3_DIR = PROCESSED_DATA_DIR / "stage3"
STAGE4_DIR = PROCESSED_DATA_DIR / "stage4"

MODELS_DIR = PROJ_ROOT / "models"
LOG_DIR = PROJ_ROOT / "logs"

# Pistachio's density path
DEFAULT_RXN_CLASS_PATH = DATA_DIR / "Pistachio Reaction Types.csv"
DEFAULT_DENSITY_PATH = DATA_DIR / "densities_clean_new.tsv"

# Notifications
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
SLACK_USER_ID = os.getenv("SLACK_USER_ID")


def send_slack_notification(message):
    data = {"text": f"<@{SLACK_USER_ID}> {message}"}

    requests.post(
        SLACK_WEBHOOK_URL,
        data=json.dumps(data),
        headers={"Content-Type": "application/json"},
    )