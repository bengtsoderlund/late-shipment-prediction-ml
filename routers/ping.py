"""
ping.py

Provides a lightweight health check endpoint (`/ping`) 
to confirm the API is running and responsive.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/ping", tags=["health"]) # tag organizes endpoint in interactive docs
def ping():
    return {"status": "ok"}
