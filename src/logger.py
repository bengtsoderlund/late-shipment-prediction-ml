"""
logger.py

Provides a reusable logger instance for each module in the project.
Ensures consistent formatting and avoids duplicate log handlers.

Used in: All modules (e.g., load_data.py, train_model.py, run_pipeline.py)
"""

import logging
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Creates or retrieves a logger instance with a standardized format.

    Parameters:
        name (str): Typically __name__ from the calling module

    Returns:
        logging.Logger: Configured logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # ─────────────────────────────────────────────
        # Create logs/ folder if needed
        # ─────────────────────────────────────────────
        logs_dir = Path(__file__).resolve().parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # ─────────────────────────────────────────────
        # File handler
        # ─────────────────────────────────────────────
        file_handler = logging.FileHandler(logs_dir / "pipeline.log", encoding='utf-8')
        file_format = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # ─────────────────────────────────────────────
        # Console handler
        # ─────────────────────────────────────────────
        stream_handler = logging.StreamHandler()
        stream_format = logging.Formatter("[%(levelname)s] %(message)s")
        stream_handler.setFormatter(stream_format)
        logger.addHandler(stream_handler)

        # ─────────────────────────────────────────────
        # Prevent double logging
        # ─────────────────────────────────────────────
        logger.propagate = False

    return logger
