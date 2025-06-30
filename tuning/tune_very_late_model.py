"""
tune_very_late_model.py

Tunes a Random Forest model to predict very late shipments using RandomizedSearchCV.
Falls back to preprocessing if preprocessed data is not found.
Logs all parameters, metrics, and the trained model to MLflow.
"""

import sys
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score
from scipy.stats import randint
import datetime


# ─────────────────────────────────────────────
# Step 1: Set Configurable Run Parameters
# ─────────────────────────────────────────────

# --- MLflow experiment tracking ---
experiment_name = "Very Late Shipment Tuning"
base_run_name = "rf_very_late_v6"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name =f"{base_run_name}_{timestamp}"
threshold = 0.3

mlflow_tags = {
    "model_type": "RandomForest",
    "dataset_version": "v1",
    "stage": "very_late_model_tuning",
    "primary_metric": "recall"
}

# --- Hyperparameupdate ter search space ---
param_dist = {
    'n_estimators': randint(375, 450),
    'max_depth': [20, 25, 30],
    'min_samples_split': randint(8, 14),
    'min_samples_leaf': randint(4, 7),
    'max_features': [0.6, 0.7],
    'bootstrap': [True],
    'criterion': ['entropy']
}



# --- RandomizedSearchCV settings ---
search_settings = {
    "n_iter": 200,
    "cv": 5,
    "scoring": "recall",
    "verbose": 0,
    "random_state": 42,
    "n_jobs": -1
}


# ─────────────────────────────────────────────
# Step 2: Configure Python Path for Imports
# ─────────────────────────────────────────────
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path().resolve()

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Step 3: Define Project Paths
# ─────────────────────────────────────────────
raw_data_file = project_root / "data" / "raw" / "shipments_raw.csv"
preprocessed_data_dir = project_root / "data" / "preprocessed"
preprocessed_data_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Step 4: Define the Tuning Function
# ─────────────────────────────────────────────
def run_tuning(X_train, y_train, X_test, y_test):
    """
    Tunes a Random Forest classifier using RandomizedSearchCV and logs the results to MLflow.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        X_test (DataFrame): Test features.
        y_test (Series): Test labels.

    Behavior:
        - Sets the MLflow experiment and run name.
        - Logs run-level tags and hyperparameters.
        - Trains a model with hyperparameter tuning using the provided search space.
        - Calculates relevant performance metrics (e.g., accuracy, recall, F1).
        - Logs metrics and model artifacts to MLflow.
    """
    
    mlruns_path = (project_root / "mlruns").as_posix()
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        logger.info(f"Starting run: {run_name}")

        
        # ─────────────────────────────────────────────
        # Step 4.1: Set MLflow Tags
        # ─────────────────────────────────────────────
        for k, v in mlflow_tags.items():
            mlflow.set_tag(k, v)

        # ─────────────────────────────────────────────
        # Step 4.2: Initialize and Tune the Model
        # ─────────────────────────────────────────────
        logger.debug("Initializing model and tuning with RandomizedSearchCV...")
        rf = RandomForestClassifier(random_state=42)

        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            **search_settings
        )
        
        logger.debug("Fitting very late model...")
        random_search.fit(X_train, y_train)

        # ─────────────────────────────────────────────
        # Step 4.3: Log Parameters and Metrics to MLflow
        # ─────────────────────────────────────────────
        logger.debug("Calculating best parameters")
        best_rf = random_search.best_estimator_
        best_params = random_search.best_params_

        logger.debug("Logging best hyperparameters and custom threshold to MLflow...")
        mlflow.log_params(best_params)
        mlflow.log_param("decision_threshold", threshold)
        
        logger.debug("Get predicted probabilities for the positive class (very late deliveries)")
        y_train_prob = best_rf.predict_proba(X_train)[:, 1]
        y_test_prob = best_rf.predict_proba(X_test)[:, 1]
        
        logger.debug("Predicting on training and test sets using best model and custom threshold")
        y_train_pred = (y_train_prob >= threshold).astype(int)
        y_test_pred = (y_test_prob >= threshold).astype(int)

        logger.debug("Calculating recall, preicsion, and F1 scores")
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        logger.debug("Logging train and test recall, precision, and F1 to MLflow...")
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("test_recall", test_recall)
        
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("test_precision", test_precision)
        
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_f1", test_f1)
        
        logger.info("\n Model performance")
        logger.info(f"Training Recall: {train_recall}")
        logger.info(f"Test Recall: {test_recall}")
        logger.info(f"\n Best Parameters: \n{best_params}")
        
        # ─────────────────────────────────────────────
        # Step 4.4: Log the Final Model to MLflow
        # ─────────────────────────────────────────────
        logger.debug("Saving model artifact to MLflow...")
        mlflow.sklearn.log_model(best_rf, run_name)
        logger.info("Model logged to MLflow.")


# ─────────────────────────────────────────────
# Step 5: Run Model Tuning with Preprocessed Data
# ─────────────────────────────────────────────
def main():
    """
    Main entry point for the tuning script.

    Loads preprocessed data if available; otherwise runs full preprocessing.
    Then fits and logs a tuned Random Forest model using MLflow.
    """
    
    logger.info("Tuning Very Late Random Forest model with best parameters...")

    try:
        # ─────────────────────────────────────────────
        # Step 5.1: Load Preprocessed Data
        # ─────────────────────────────────────────────
        logger.debug("Loading saved train/test predictors and target variables...")
        X_train = joblib.load(preprocessed_data_dir / "X_train.pkl")
        y_train = joblib.load(preprocessed_data_dir / "y_very_late_train.pkl")
        X_test = joblib.load(preprocessed_data_dir / "X_test.pkl")
        y_test = joblib.load(preprocessed_data_dir / "y_very_late_test.pkl")
    
        run_tuning(X_train, y_train, X_test, y_test)
        
    except FileNotFoundError:
        # ─────────────────────────────────────────────
        # Step 5.2: Preprocess Data if Files Are Missing
        # ─────────────────────────────────────────────
        logger.debug("Preprocessed data not found. Re-running preprocessing steps...")
        from src.load_data import load_raw_data
        from src.clean_data import clean_raw_data
        from src.feature_engineering import engineer_features
        from src.preprocess_features import preprocess_features
    
        df = load_raw_data(raw_data_file)
        df = clean_raw_data(df)
        df = engineer_features(df)
        processed = preprocess_features(df)
    
        X_train = processed["X_train"]
        y_train = processed["y_very_late_train"]
        X_test = processed["X_test"]
        y_test = processed["y_very_late_test"]
    
        run_tuning(X_train, y_train, X_test, y_test)


# ─────────────────────────────────────────────
# Step 6: Run the Script
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()

