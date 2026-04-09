"""Parliament configuration."""

import os

HOST = "0.0.0.0"
PORT = 8080
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ADMIN_KEY = "sp_admin_parliament"
LOG_SUMMARY_MAX_LEN = 2000
