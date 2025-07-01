"""
main.py

Entry point for the FastAPI app.

Includes routers for:
- Landing page (`/`)
- Health check (`/ping`)
- Very late shipment prediction (`/predict_very_late/`)
- Late shipment prediction (`/predict_late/`) — conditionally included for local use only

Note:
- The "late shipment" prediction endpoint (`/predict_late/`) is excluded on Render due to memory limitations.
  It is only registered if the `late_model.pkl` file exists, which allows local use without causing issues in deployment.
"""

import os
from fastapi import FastAPI
from routers import landing, ping, predict_late, predict_very_late

# Create FastAPI app with metadata
app = FastAPI(
    title="Shipment Delay Prediction API",
    description="A FastAPI application for predicting late and very late deliveries based on order and shipment data.",
    version="1.0.0"
)

# Always available routes
app.include_router(landing.router)          # Root landing page
app.include_router(ping.router)             # Health check endpoint
app.include_router(predict_very_late.router)  # Very late prediction (3+ day delay)

# Conditionally register the /predict_late route if the model file exists
# This avoids deploying it to environments like Render where the file is too large
if os.path.exists("models/late_model.pkl"):
    from routers import predict_late
    app.include_router(predict_late.router)