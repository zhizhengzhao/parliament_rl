"""Parliament configuration."""

from pathlib import Path

HOST = "0.0.0.0"
PORT = 8080
ADMIN_KEY = "sp_admin_parliament"
LOG_SUMMARY_MAX_LEN = 2000
DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
