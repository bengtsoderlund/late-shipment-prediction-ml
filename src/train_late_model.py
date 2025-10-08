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
- MLflow tracking of parameters, metrics, and serialized model artifacts

Used in: run_pipeline.py and retrain_pipeline.py (Step 5)
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import datetime
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from src.logger import get_logger


logger = get_logger(__name__)


def train_late_model(X_train, y_train, X_test, y_test, model_file, mlruns_path):
    """
    Trains a Random Forest model to predict late shipments using processed feature and target data.

    Steps:
    - Define a Random Forest classifier using pre-tuned hyperparameters
    - Fit the model on the training data
    - Evaluate the model using accuracy and a classification report
    - Log top feature importances
    - Save the trained model to disk using joblib
    - Track parameters, metrics, and model artifacts with MLflow

    Parameters:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target for late shipments
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target for late shipments
        model_file (str or Path): File path where the trained model should be saved
        mlruns_path (str or Path): Directory for MLflow experiment tracking data (runs, metrics, and artifacts)

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

        # ─────────────────────────────────────────────
        # MLflow tracking: log params, metrics, and model
        # ─────────────────────────────────────────────
        logger.info("Starting MLflow tracking for late shipment model...")
        mlruns_path = Path(mlruns_path).resolve()
        mlflow.set_tracking_uri(mlruns_path.as_uri())  # yields "file:///.../mlruns" correctly on all OSes
        logger.info(f"Tracking URI set to {mlruns_path.as_uri()}")

        mlflow.set_experiment("Late Shipment Training")
        logger.info("Experiment 'Late Shipment Training' activated.")
        
        run_name = f"late_rf_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting new MLflow run: {run_name}")

        with mlflow.start_run(run_name=run_name):
            logger.info("Logging run metadata and parameters...")
            mlflow.set_tag("stage", "late_model_training")
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("primary_metric", "accuracy")
            mlflow.set_tag("data_version", "v2024-11-27")

            # Log the actual params used by the trained estimator
            mlflow.log_params(rf.get_params(deep=False))

            logger.info("Logging performance metrics...")
            mlflow.log_metric("train_accuracy", float(train_acc))
            mlflow.log_metric("test_accuracy", float(test_acc))
            
            logger.info("Create signature and input example...")
            signature = infer_signature(X_test, y_test_pred)
            input_example = X_test.head(5)

            logger.info("Saving model artifact to MLflow...")
            mlflow.sklearn.log_model(
                rf,
                artifact_path="late_model",
                signature=signature,
                input_example=input_example
            )

        logger.info("Late model training completed successfully.\n")
        
    except Exception as e:
        logger.error(f"Failed to train late shipment model: {e}", exc_info=True)
        raise
