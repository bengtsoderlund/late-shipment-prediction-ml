"""
preprocess_features.py

This module accepts a cleaned, feature-engineered DataFrame and returns 
preprocessed training and test sets along with two target variables.

This module performs:
- Target variable extraction
- Feature selection
- Train-test splitting
- Numerical scaling
- Categorical encoding (one-hot and ordinal)
- Final feature matrix construction
- Optionally saves unprocessed data, preprocessed datasets, scaler, and encoders

Used in: run_pipeline.py (Step 4)

Global feature lists are defined at the top of this module to ensure consistency
between training (in this function) and inference (in the FastAPI router).
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from src.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Selected and grouped model input features
# ─────────────────────────────────────────────
# These features are used for training and inference.
# Defined globally to ensure consistency between the ML pipeline and FastAPI router.

NUMERICAL_FEATURES = [
    'order_item_quantity', 'order_item_total', 'product_price', 
    'year', 'month', 'day', 'order_value', 'unique_items_per_order', 
    'order_item_discount_rate', 'units_per_order', 'order_profit_per_order'
]

ONEHOT_FEATURES = ['type', 'customer_segment', 'shipping_mode']

LABEL_FEATURES = [
    'category_id', 'customer_country', 'customer_state', 'department_id', 
    'order_city', 'order_country', 'order_region', 'order_state'
]

# All model input features
ALL_INPUT_FEATURES = NUMERICAL_FEATURES + ONEHOT_FEATURES + LABEL_FEATURES


def preprocess_features(
        df, 
        save_to_disk, 
        unprocessed_path, 
        preprocessed_path,
        scaler_file,
        onehot_encoder_file,
        ordinal_encoder_file
):
    """
    Preprocesses features for training ML models to predict both 'late' and 'very_late'.

    Steps:
    - Create target variables for both models
    - Select input features used for prediction
    - Train-test split
    - Scale numerical features and save scaler
    - Encode categorical features (one-hot and label encoding)
    - Save encoders (optional)
    - Save unprocessed and preprocessed datasets (optional)

    Parameters:
        df (pd.DataFrame): The engineered DataFrame with targets already created
        save_to_disk (bool): If True, saves unprocessed and preprocessed datasets to disk

    Returns:
        Dictionary containing:
        - X_train (pd.DataFrame): Processed training features
        - X_test (pd.DataFrame): Processed test features
        - y_late_train (pd.Series): Binary target for late shipments (train set)
        - y_late_test (pd.Series): Binary target for late shipments (test set)
        - y_very_late_train (pd.Series): Binary target for very late shipments (train set)
        - y_very_late_test (pd.Series): Binary target for very late shipments (test set)
        
    Raises:
        Exception: If the preprocessing cannot be completed.
    """

    
    try:
        # ─────────────────────────────────────────────
        # Step 1: Extract target variables
        # ─────────────────────────────────────────────
        logger.debug("Defining target variables: 'late' and 'very_late'")
        y_late = df["late"]
        y_very_late = df["very_late"]
        
        # ─────────────────────────────────────────────
        # Step 2: Select predictive features
        # ─────────────────────────────────────────────
        logger.debug("Selecting input features for model training...")

        X = df[ALL_INPUT_FEATURES]
        
        # ─────────────────────────────────────────────
        # Step 3: Save unprocessed datasets (optional)
        # ─────────────────────────────────────────────
        if save_to_disk:
            logger.info(f"Saving unprocessed data to: {unprocessed_path}")
            logger.debug(("Purpose: to provide a clean, pre-feature-engineered baseline for schema definition, "
                          "API input validation, and reproducibility."))
            
            unprocessed_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(X, unprocessed_path / "X_unprocessed.pkl")
        
        # ─────────────────────────────────────────────
        # Step 4: Train/test split
        # ─────────────────────────────────────────────
        logger.debug("Performing train/test split...")
        X_train, X_test, y_late_train, y_late_test = train_test_split(X, y_late, test_size=0.25, random_state=42)
        y_very_late_train = y_very_late.loc[y_late_train.index]
        y_very_late_test = y_very_late.loc[y_late_test.index]
        logger.debug(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    

        # ─────────────────────────────────────────────
        # Step 5: Initialize scalers/encoders
        # ─────────────────────────────────────────────
        logger.debug("Initializing scalers/encoders (numerical, one-hot, label)...")
        scaler = RobustScaler()
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        

        # ─────────────────────────────────────────────
        # Step 6: Transform numerical features and save scaler
        # ─────────────────────────────────────────────
        logger.debug("Scaling numerical features and saving fitted scaler...")
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train[NUMERICAL_FEATURES]),
            columns=NUMERICAL_FEATURES,
            index=X_train.index
        )
    
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test[NUMERICAL_FEATURES]),
            columns=NUMERICAL_FEATURES,
            index=X_test.index
        )
        
        if save_to_disk and scaler_file:
            scaler_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_file)
            logger.info(f"Saved RobustScaler to: {scaler_file}")

        # ─────────────────────────────────────────────
        # Step 7: One-hot encode categorical features and save encoder
        # ─────────────────────────────────────────────
        logger.debug("One-hot encoding low-cardinality categorical features and saving encoder...")
        X_train_onehot = pd.DataFrame(
            onehot_encoder.fit_transform(X_train[ONEHOT_FEATURES]),
            columns=onehot_encoder.get_feature_names_out(ONEHOT_FEATURES),
            index=X_train.index
        )
        X_test_onehot = pd.DataFrame(
            onehot_encoder.transform(X_test[ONEHOT_FEATURES]),
            columns=onehot_encoder.get_feature_names_out(ONEHOT_FEATURES),
            index=X_test.index
        )

        if save_to_disk and onehot_encoder_file:
            onehot_encoder_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(onehot_encoder, onehot_encoder_file)
            logger.info(f"Saved OneHotEncoder to: {onehot_encoder_file}")

        # ─────────────────────────────────────────────
        # Step 8: Label encode high-cardinality features and save encoder
        # ─────────────────────────────────────────────
        logger.debug("Label encoding high-cardinality categorical features and saving encoder...")
        X_train_label = pd.DataFrame(
            ordinal_encoder.fit_transform(X_train[LABEL_FEATURES]),
            columns=LABEL_FEATURES,
            index=X_train.index
        )
    
        X_test_label = pd.DataFrame(
            ordinal_encoder.transform(X_test[LABEL_FEATURES]),
            columns=LABEL_FEATURES,
            index=X_test.index
        )
        
        if save_to_disk and ordinal_encoder_file:
            ordinal_encoder_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(ordinal_encoder, ordinal_encoder_file)
            logger.info(f"Saved OrdinalEncoder to: {ordinal_encoder_file}")

        # ─────────────────────────────────────────────
        # Step 9: Concatenate all processed features
        # ─────────────────────────────────────────────
        logger.debug("Concatenating all features...")
        X_train_processed = pd.concat([X_train_scaled, X_train_onehot, X_train_label], axis=1)
        X_test_processed = pd.concat([X_test_scaled, X_test_onehot, X_test_label], axis=1)
        logger.debug(f"Final training feature shape: {X_train_processed.shape}")
        logger.debug(f"Final test feature shape: {X_test_processed.shape}")
        
        # ─────────────────────────────────────────────
        # Step 10: Save preprocessed datasets (optional)
        # ─────────────────────────────────────────────
        if save_to_disk:
            logger.info(f"Saving preprocessed data to: {preprocessed_path}")
            logger.debug("Purpose: to enable reuse during model tuning or evaluation without rerunning preprocessing.")
            
            preprocessed_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(X_train_processed, preprocessed_path / "X_train.pkl")
            joblib.dump(X_test_processed, preprocessed_path / "X_test.pkl")
            joblib.dump(y_late_train, preprocessed_path / "y_late_train.pkl")
            joblib.dump(y_late_test, preprocessed_path / "y_late_test.pkl")
            joblib.dump(y_very_late_train, preprocessed_path / "y_very_late_train.pkl")
            joblib.dump(y_very_late_test, preprocessed_path / "y_very_late_test.pkl")
            
        # ─────────────────────────────────────────────
        # Step 11: Final logging and return
        # ─────────────────────────────────────────────
        logger.info("Preprocessing completed successfully")
        
        return {
            "X_train": X_train_processed,
            "X_test": X_test_processed,
            "y_late_train": y_late_train,
            "y_late_test": y_late_test,
            "y_very_late_train": y_very_late_train,
            "y_very_late_test": y_very_late_test
        }
    
    except Exception as e:
        logger.error(f"Failed to preprocess features: {e}", exc_info=True)
        raise
