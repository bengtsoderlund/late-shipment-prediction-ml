"""
run_pipeline.py

Main script to run the machine learning pipeline for predicting late shipments.
Steps include loading raw data, cleaning, feature engineering, preprocessing, 
model training, and evaluation.
Logs progress and errors to both console and log file.
"""

from pathlib import Path
from src.logger import get_logger
from src.load_data import load_raw_data
from src.clean_data import clean_raw_data
from src.feature_engineering import engineer_features
from src.preprocess_features import preprocess_features
from src.train_late_model import train_late_model
from src.train_very_late_model import train_very_late_model
import time

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# CONFIG: Paths
# ─────────────────────────────────────────────
try:
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path().resolve()

raw_data_file = base_dir / "data" / "raw" / "shipments_raw.csv"
unprocessed_data_dir = base_dir / "data" / "unprocessed"
preprocessed_data_dir = base_dir / "data" / "preprocessed"
late_model_file = base_dir / "models" / "late_model.pkl"
very_late_model_file = base_dir / "models" / "very_late_model.pkl"
scaler_file = base_dir / "models" / "scaler.pkl"
onehot_encoder_file = base_dir / "models" / "onehot_encoder.pkl"
ordinal_encoder_file = base_dir / "models" / "ordinal_encoder.pkl"
mlruns_path = base_dir / "mlruns"

# Create all local directories that must exist
required_dirs = [
    raw_data_file.parent,        # "data/raw"
    unprocessed_data_dir,
    preprocessed_data_dir,
    late_model_file.parent,      # "models"
    mlruns_path,
]

for d in required_dirs:
    d.mkdir(parents=True, exist_ok=True)


def main():
    """
    Runs the full ML pipeline for predicting late and very late shipments.
    Logs start and end times, loads raw data, and calls downstream processing steps.
    """
    
    start = time.time()
    logger.info("Starting pipeline...\n")
    
    try:
        # ─────────────────────────────────────────────
        # Step 1: Load raw data
        # ─────────────────────────────────────────────
        logger.info("- Step 1: Load raw data")
        df = load_raw_data(raw_data_file)
        
        logger.debug(f"Data shape: {df.shape}")
        logger.debug(f"Preview:\n{df.head()}")
        
        # ─────────────────────────────────────────────
        # Step 2: Clean raw data
        # ─────────────────────────────────────────────
        logger.info("- Step 2: Clean raw data")
        df = clean_raw_data(df)
        logger.debug(f"Cleaned data shape: {df.shape}")
        logger.debug(f"Preview:\n{df.head()}")
        
        # ─────────────────────────────────────────────
        # Step 3: Feature engineering
        # ─────────────────────────────────────────────
        logger.info("- Step 3: Feature engineering")
        df = engineer_features(df)
        logger.debug(f"Engineered data shape: {df.shape}")
        logger.debug(f"Preview:\n{df.head()}")
        
        # ─────────────────────────────────────────────
        # Step 4: Preprocess features
        # ─────────────────────────────────────────────
        logger.info("- Step 4: Preprocess features")
        processed_data = preprocess_features(
            df, 
            save_to_disk=True,
            unprocessed_path=unprocessed_data_dir,
            preprocessed_path=preprocessed_data_dir,
            scaler_file=scaler_file,
            onehot_encoder_file=onehot_encoder_file,
            ordinal_encoder_file=ordinal_encoder_file
        )
        
        # ─────────────────────────────────────────────
        # Step 5: Train late model
        # ─────────────────────────────────────────────
        logger.info("- Step 5: Train late model")
        train_late_model(
            X_train=processed_data["X_train"],
            y_train=processed_data["y_late_train"],
            X_test=processed_data["X_test"],
            y_test=processed_data["y_late_test"],
            model_file=late_model_file,
            mlruns_path=mlruns_path
        )
        
        # ─────────────────────────────────────────────
        # Step 6: Train very late model
        # ─────────────────────────────────────────────
        logger.info("- Step 6: Train very late model")
        train_very_late_model(
            X_train=processed_data["X_train"],
            y_train=processed_data["y_very_late_train"],
            X_test=processed_data["X_test"],
            y_test=processed_data["y_very_late_test"],
            model_file=very_late_model_file,
            mlruns_path=mlruns_path
        )
        
        duration = time.time() - start
        minutes = int(duration // 60)
        seconds = duration % 60
        if minutes:
            logger.info(f"\n Pipeline completed in {minutes} min {seconds:.2f} sec")
        else:
            logger.info(f"\n Pipeline completed in {seconds:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()

