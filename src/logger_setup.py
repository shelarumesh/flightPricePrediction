import logging
import sys
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os


def setup_logger(name: str, log_file: str, level=logging.INFO) -> Logger:
    """Function to set up a logger with both file and console handlers.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level: Logging level.

    Returns:
        Logger: Configured logger instance.
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_file).parent
    os.makedirs(log_dir, exist_ok=True)

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
    console_handler = logging.StreamHandler(sys.stdout)

    # Set levels for handlers
    file_handler.setLevel(level)
    console_handler.setLevel(level)

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger