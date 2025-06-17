"""
main.py

Entry point for the FastAPI app.

Includes routers for:
- Health check (`/ping`)
- Late shipment prediction (`/predict_late/`)
- Very late shipment prediction (`/predict_very_late/`)
"""

from fastapi import FastAPI
from routers import ping, predict_late, predict_very_late

# Create app with metadata
app = FastAPI(
    title="Shipment Delay Prediction API",
    description="A FastAPI application for predicting late and very late deliveries based on order and shipment data.",
    version="1.0.0"
)

# Include routers
app.include_router(ping.router)
app.include_router(predict_late.router)
app.include_router(predict_very_late.router)