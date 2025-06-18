"""
predict_late.py

FastAPI router for predicting late shipments.

This endpoint:
- Accepts shipment features as input (validated via Pydantic).
- Loads saved scaler and encoders to preprocess input.
- Loads trained 'late' shipment model.
- Returns a binary prediction indicating whether the shipment is late.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from api.shipment_schema import ShipmentFeatures
from src.preprocess_features import NUMERICAL_FEATURES, ONEHOT_FEATURES, LABEL_FEATURES
import pandas as pd
import joblib
from src.logger import get_logger

def load_artifact(file, name):
    try:
        return joblib.load(file)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"{name} not found. Please run the pipeline first to generate required model files."
        )


logger = get_logger(__name__)

base_dir = Path(__file__).resolve().parent.parent
scaler_file = base_dir / "models" / "scaler.pkl"
onehot_encoder_file = base_dir / "models" / "onehot_encoder.pkl"
ordinal_encoder_file = base_dir / "models" / "ordinal_encoder.pkl"
late_model_file = base_dir / "models" / "late_model.pkl"

router = APIRouter()

@router.post("/predict_late/", tags=["prediction"])
async def predict_late(shipment_features: ShipmentFeatures):
    logger.info("Received request to /predict_late endpoint")

    data_dict = shipment_features.model_dump()
    logger.debug(f"Raw input data: {data_dict}")
    X_unprocessed = pd.DataFrame([data_dict])
    logger.debug(f"DataFrame constructed: {X_unprocessed}")
    
    # ─────────────────────────────────────────────
    # Feature preprocessing and transformation
    # ─────────────────────────────────────────────
    X_num = X_unprocessed[NUMERICAL_FEATURES]
    X_onehot = X_unprocessed[ONEHOT_FEATURES]
    X_label = X_unprocessed[LABEL_FEATURES]
    
    scaler = load_artifact(scaler_file, "scaler")
    logger.info("Scaler loaded successfully")
    onehot_encoder = load_artifact(onehot_encoder_file, "onehot_encoder")
    logger.info("OneHot encoder loaded successfully")
    ordinal_encoder = load_artifact(ordinal_encoder_file, "ordinal_encoder")
    logger.info("Ordinal encoder loaded successfully")
        
    X_num_scaled = scaler.transform(X_num)
    X_onehot_encoded = onehot_encoder.transform(X_onehot)
    X_label_encoded = ordinal_encoder.transform(X_label)
    
    X_num_scaled = pd.DataFrame(X_num_scaled, columns=NUMERICAL_FEATURES, index=X_unprocessed.index)
    X_onehot_encoded = pd.DataFrame(
        X_onehot_encoded,
        columns=onehot_encoder.get_feature_names_out(ONEHOT_FEATURES),
        index=X_unprocessed.index
    )
    X_label_encoded = pd.DataFrame(X_label_encoded, columns=LABEL_FEATURES, index=X_unprocessed.index)
    
    X_processed = pd.concat([X_num_scaled, X_onehot_encoded, X_label_encoded], axis=1)
    logger.debug(f"X_processed shape: {X_processed.shape}")
   
    # ─────────────────────────────────────────────
    # Load trained model and generate prediction
    # ─────────────────────────────────────────────
    late_model = load_artifact(late_model_file, "late_model")
    logger.info("Late model loaded successfully")
    is_late = late_model.predict(X_processed)[0]
    
    logger.info(f"Prediction generated: {is_late}")
    return {"late_prediction": int(is_late)}