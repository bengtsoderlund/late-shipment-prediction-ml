"""
train_late_model.py

This module trains a Random Forest model to predict late shipments using 
preprocessed training and test datasets.

This module performs:
- Model definition with predefined hyperparameters
- Model fitting on training data
- Accuracy and classification report logging
- Feature importance extraction and logging
- Model saving to disk using joblib

Used in: run_pipeline.py (Step 5)
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
from src.logger import get_logger


logger = get_logger(__name__)


def train_late_model(X_train, y_train, X_test, y_test, model_file):
    """
    Trains a Random Forest model to predict late shipments using processed feature and target data.

    Steps:
    - Define a Random Forest classifier using pre-tuned hyperparameters
    - Fit the model on the training data
    - Evaluate the model using accuracy and a classification report
    - Log top feature importances
    - Save the trained model to disk using joblib

    Parameters:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target for late shipments
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target for late shipments
        model_file (str or Path): File path where the trained model should be saved

    Returns:
        None

    Raises:
        Exception: If the model training, evaluation, or saving process fails
    """

    try:
        # ─────────────────────────────────────────────
        # Define and fit model with best-known hyperparameters
        # ─────────────────────────────────────────────
        rf = RandomForestClassifier(
            random_state=42,
            bootstrap=True,
            criterion='entropy',
            max_depth=35,
            max_features=0.4,
            min_samples_leaf=6,
            min_samples_split=15,
            n_estimators=567
            
        )

        logger.info("Training Random Forest model with best parameters...")
        rf.fit(X_train, y_train)
        
        # ─────────────────────────────────────────────
        # Evaluate model
        # ─────────────────────────────────────────────
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        logger.info(f"Training Accuracy: {train_acc:.4f}")
        logger.info(f"Test Accuracy:     {test_acc:.4f}")
        logger.debug("Classification Report (test set):\n" + classification_report(y_test, y_test_pred))

        # ─────────────────────────────────────────────
        # Save model to specified path
        # ─────────────────────────────────────────────
        model_file = Path(model_file)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, model_file)
        logger.info(f"Model saved to: {model_file}")

        # ─────────────────────────────────────────────
        # Log top 10 feature importances
        # ─────────────────────────────────────────────
        importances = rf.feature_importances_
        feature_names = X_train.columns
        sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

        logger.info("Top 10 Feature Importances:")
        for feature, importance in sorted_features[:10]:
            logger.info(f"{feature}: {importance:.4f}")

        logger.info("Late model training completed successfully.\n")
        
    except Exception as e:
        logger.error(f"Failed to train late shipment model: {e}", exc_info=True)
        raise
