"""
load_data.py

This module loads the raw shipment data from a CSV file.

It logs:
- The file path being loaded
- Whether the file was successfully read

Used in: run_pipeline.py (Step 1)
"""

import pandas as pd
from src.logger import get_logger


logger = get_logger(__name__)


def load_raw_data(filepath):
    """
    Loads raw shipment data from a CSV file.
    
    Parameters:
        filepath (Path or str): Path to the CSV file.

    Returns:
        pd.DataFrame: The raw DataFrame.
        
    Raises:
        Exception: If the file cannot be read.
    """
    
    logger.info(f"Loading data from: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding="latin1")
        logger.info("Raw data successfully loaded")
        return df
    
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}", exc_info=True)
        raise