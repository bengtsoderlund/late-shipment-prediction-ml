"""
predict_very_late.py — Router for very-late shipment prediction (AWS/ECS)

What it does
- Accepts shipment features (Pydantic schema) and returns a binary prediction.
- Uses artifacts retrieved from `request.app.state.*` (loaded once at app startup from S3).

Request
- POST /predict_very_late/
- Body: ShipmentFeatures (see shipment_schema.py)

Response
- JSON: {"very_late_prediction": 0 or 1}

Notes
- No local file I/O: all scaler/encoders/model objects are provided via app.state.
- Keep this router unprefixed so the final path is exactly /predict_very_late/.
"""

import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from api.shipment_schema import ShipmentFeatures
from src.preprocess_features import NUMERICAL_FEATURES, ONEHOT_FEATURES, LABEL_FEATURES

logger = logging.getLogger("app.predict_very_late")
router = APIRouter()

@router.post("/predict_very_late/", tags=["prediction"])
async def predict_very_late(shipment_features: ShipmentFeatures, request: Request):
    logger.info("Received request to /predict_very_late/")

    # Construct single-row DataFrame from validated payload
    data_dict = shipment_features.model_dump()
    logger.debug(f"Raw input data: {data_dict}")
    X_unprocessed = pd.DataFrame([data_dict])
    logger.debug(f"DataFrame constructed: {X_unprocessed}")
    
    # ─────────────────────────────────────────────
    # Retrieve artifacts from app.state
    # ─────────────────────────────────────────────
    try:
        scaler = request.app.state.scaler
        onehot = request.app.state.onehot
        ordinal = request.app.state.ordinal
        very_late_model = request.app.state.very_late_model
        logger.info("Retrieved artifacts from app.state: scaler, onehot, ordinal, very_late_model")
        
    except AttributeError:
        logger.exception("Artifacts missing on app.state")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model artifacts not available on the server. Check startup logs."
        )
    
    
    # ─────────────────────────────────────────────
    # Split by feature groups
    # ─────────────────────────────────────────────
    try:
        X_num = X_unprocessed[NUMERICAL_FEATURES]
        X_onehot = X_unprocessed[ONEHOT_FEATURES]
        X_label = X_unprocessed[LABEL_FEATURES]
        
    except KeyError as e:
        expected = NUMERICAL_FEATURES + ONEHOT_FEATURES + LABEL_FEATURES
        missing = []
        for feature in expected:
            if feature not in X_unprocessed.columns:
                missing.append(feature)
                
        logger.warning(f"Missing features in payload: {missing}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required feature(s): {missing}"
        ) from e
        
   
    # ─────────────────────────────────────────────
    # Transform
    # ─────────────────────────────────────────────
    try:
        X_num_scaled = scaler.transform(X_num)
        X_onehot_encoded = onehot.transform(X_onehot)
        X_label_encoded = ordinal.transform(X_label)
        
    except Exception as e:
        logger.exception(f"Preprocessing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Preprocessing failed: {e}"
        )
            
    # ─────────────────────────────────────────────
    # Rebuild DataFrames with column names
    # ─────────────────────────────────────────────
    X_num_scaled = pd.DataFrame(X_num_scaled, columns=NUMERICAL_FEATURES, index=X_unprocessed.index)
    X_onehot_encoded = pd.DataFrame(
        X_onehot_encoded,
        columns=onehot.get_feature_names_out(ONEHOT_FEATURES),
        index=X_unprocessed.index
    )
    X_label_encoded = pd.DataFrame(X_label_encoded, columns=LABEL_FEATURES, index=X_unprocessed.index)
        
    # ─────────────────────────────────────────────
    # Concatenate processed features
    # ─────────────────────────────────────────────
    X_processed = pd.concat([X_num_scaled, X_onehot_encoded, X_label_encoded], axis=1)
    logger.info(f"Preprocessing complete. X_processed shape: {X_processed.shape}")
        
    # ─────────────────────────────────────────────
    # Predict 
    # ─────────────────────────────────────────────
    try:
        is_very_late = very_late_model.predict(X_processed)[0]
    except Exception:
        logger.exception("Model prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model prediction error."
        )
        
    logger.info(f"Prediction generated: {int(is_very_late)}")
    return {"very_late_prediction": int(is_very_late)}