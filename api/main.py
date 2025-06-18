"""
main.py

Entry point for the FastAPI app.

Includes routers for:
- Landing page (`/`)
- Health check (`/ping`)
- Very late shipment prediction (`/predict_very_late/`)

Note:
- The "late shipment" prediction endpoint (`/predict_late/`) is currently disabled 
  due to memory constraints on Render’s free tier. It can be re-enabled by 
  upgrading to a paid plan with higher memory capacity.
"""

from fastapi import FastAPI
from routers import landing, ping, predict_late, predict_very_late

# Create app with metadata
app = FastAPI(
    title="Shipment Delay Prediction API",
    description="A FastAPI application for predicting late and very late deliveries based on order and shipment data.",
    version="1.0.0"
)

# Include routers
app.include_router(landing.router)
app.include_router(ping.router)
# app.include_router(predict_late.router) # Temporarily disabled due to model size exceeding Render's free tier limits
app.include_router(predict_very_late.router)