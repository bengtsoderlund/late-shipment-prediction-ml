"""
clean_data.py

This module cleans the raw shipment data.

It logs:
- The steps taken during data cleaning
- Whether cleaning was successful or failed

Used in: run_pipeline.py (Step 2)
"""

import pandas as pd
from src.logger import get_logger


logger = get_logger(__name__)


def clean_raw_data(df):
    """
    Clean raw shipment data
    
    Parameters:
        df (pd.DataFrame): The raw DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
        
    Raises:
        Exception: If the data cannot be cleaned.
    """
    
    try:
        logger.debug(("Renaming columns to snake_case for consistency, readability, "
                      "and compatibility with code conventions and API schemas."))
        rename_dict = {col: col.replace(" ","_").replace("(", "").replace(")", "").lower() for col in df.columns}
        df.rename(columns=rename_dict, inplace=True)
        
        logger.debug("Removing invalid product category names...")
        df = df[df['category_name'] != 'As Seen on  TV!']

        logger.debug("Filtering out incorrect customer state codes...")
        df = df[~df['customer_state'].isin(['91732', '95758'])]
    
        logger.debug("Stripping and fixing spaces in street names...")
        df['customer_street'] = df['customer_street'].str.strip().str.replace(r'\s+', ' ', regex=True)

        logger.debug("Filtering to only complete or closed orders...")
        df = df[df['order_status'].isin(['COMPLETE', 'CLOSED'])]
        
        logger.info("Raw data cleaned successfully")
        return df
    
    
    except Exception as e:
        logger.error(f"Failed to clean raw data: {e}", exc_info=True)
        raise
