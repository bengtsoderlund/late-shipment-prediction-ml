"""
feature_engineering.py

This module creates new features for the ML model.

It logs:
- The steps to create new features
- Whether feature engineering was successful or failed

Used in: run_pipeline.py (Step 3)
"""

import pandas as pd
from src.logger import get_logger


logger = get_logger(__name__)


def engineer_features(df):
    """
    Engineer new features
    
    Parameters:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        pd.DataFrame: The engineered DataFrame.
        
    Raises:
        Exception: If the feature engineering cannot be completed.
    """
    
    try:
        # ─────────────────────────────────────────────
        # Create target variables
        # ─────────────────────────────────────────────
        logger.debug("Creating target variables...")
        df['late'] = (df['days_for_shipping_real'] > df['days_for_shipment_scheduled']).astype(int)
        df['very_late'] = (df['days_for_shipping_real'] > (df['days_for_shipment_scheduled'] + 2)).astype(int)

        # ─────────────────────────────────────────────
        # Order-level aggregated features
        # ─────────────────────────────────────────────
        logger.debug("Calculating total order value...")
        df['order_value'] = df.groupby('order_id')['order_item_total'].transform('sum')

        logger.debug("Counting unique items per order...")
        df['unique_items_per_order'] = df.groupby('order_id')['order_item_id'].transform('nunique')

        logger.debug("Calculating number of units per order...")
        df['units_per_order'] = df.groupby('order_id')['order_item_quantity'].transform('sum')

        # ─────────────────────────────────────────────
        # Time-based features
        # ─────────────────────────────────────────────
        logger.debug("Extracting time-based features from order date...")
        df['order_date_dateorders'] = pd.to_datetime(df['order_date_dateorders'], format='%m/%d/%Y %H:%M')
        df['year'] = df['order_date_dateorders'].dt.year
        df['month'] = df['order_date_dateorders'].dt.month
        df['day'] = df['order_date_dateorders'].dt.day
        df['hour'] = df['order_date_dateorders'].dt.hour
        df['minute'] = df['order_date_dateorders'].dt.minute
        df['day_of_week'] = df['order_date_dateorders'].dt.weekday
        
        logger.info("Feature engineering completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Failed to engineer features: {e}", exc_info=True)
        raise