# ============================================
# Logging Configuration for NetworkSecurity Project
# ============================================

import logging       # Standard Python logging module
import os            # For creating and managing directories
from datetime import datetime  # To timestamp the log file names

# -------------------------------------------------
# 1️⃣ Generate a unique log file name using timestamp
# Example: "10_22_2025_14_30_45.log"
# -------------------------------------------------
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# -------------------------------------------------
# 2️⃣ Create a 'logs' directory path inside the current working directory
# Each log file will be stored inside the 'logs' folder.
# -------------------------------------------------
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# -------------------------------------------------
# 3️⃣ Create the directory if it doesn’t already exist
# (prevents errors when writing logs for the first time)
# -------------------------------------------------
os.makedirs(logs_path, exist_ok=True)

# -------------------------------------------------
# 4️⃣ Full path for the actual log file to be created
# -------------------------------------------------
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# -------------------------------------------------
# 5️⃣ Configure the logging system
# - filename: path to the log file
# - format: structure of each log entry
# - level: minimum severity of messages to record
# -------------------------------------------------
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
