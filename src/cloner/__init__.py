# cloner.py
import logging
import os

# Define the log directory and file path
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Configure logger
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

logging.basicConfig(
    level=logging.DEBUG,  # Ensure the logger captures all levels of messages
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()  # StreamHandler to show logs in the console
    ]
)

# Create logger object
logger = logging.getLogger("cloner")
logger.propagate = True 
