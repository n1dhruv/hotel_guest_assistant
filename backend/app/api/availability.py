from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.tools import check_availability

router = APIRouter(prefix="/api/availability", tags=["availability"])

class DirectAvailabilityRequest(BaseModel):
    checkIn: str = Field(description="Check-in date in YYYY-MM-DD format")
    checkOut: str = Field(description="Check-out date in YYYY-MM-DD format")
    adults: int = Field(default=2, ge=1, le=10, description="Number of adults")

@router.post("")
def check_availability_endpoint(req: DirectAvailabilityRequest) -> dict[str, Any]:
    """
    Direct deterministic endpoint for the room availability booking widget.
    """
    return check_availability(checkIn=req.checkIn, checkOut=req.checkOut, adults=req.adults)
