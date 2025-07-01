"""
main.py

Entry point for the FastAPI app.

Includes routers for:
- Landing page (`/`)
- Health check (`/ping`)
- Very late shipment prediction (`/predict_very_late/`)

Note:
- The "late shipment" prediction endpoint (`/predict_late/`) is excluded from deployment
  due to memory limitations on platforms like Render. The route remains available locally
  and can be enabled by uncommenting the final line below.
"""

from fastapi import FastAPI
from routers import landing, ping, predict_late, predict_very_late

# Create FastAPI app with metadata
app = FastAPI(
    title="Shipment Delay Prediction API",
    description="A FastAPI application for predicting late and very late deliveries based on order and shipment data.",
    version="1.0.0"
)

# Register available routes
app.include_router(landing.router)              # Root landing page
app.include_router(ping.router)                 # Health check endpoint
app.include_router(predict_very_late.router)    # Very late prediction (3+ day delay)

# Optional: Enable this route for local use only (commented out due to deployment limits)
# app.include_router(predict_late.router)       # Late prediction (1+ day delay)
