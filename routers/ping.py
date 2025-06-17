from fastapi import APIRouter

# Create router object (needed to include ping.py into main.py)
router = APIRouter()

# Define a simple GET endpoint at /ping 
@router.get("/ping", tags=["health"]) # tag organizes endpoint in interactive docs
def ping():
    return {"status": "ok"}
