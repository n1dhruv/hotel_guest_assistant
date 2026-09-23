import json
from datetime import datetime, date
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
HOTEL_DATA_FILE = BASE_DIR / "data" / "hotel_data.json"

class AvailabilityInput(BaseModel):
    checkIn: str = Field(description="Check-in date in YYYY-MM-DD format (e.g. '2026-10-15')")
    checkOut: str = Field(description="Check-out date in YYYY-MM-DD format (e.g. '2026-10-18')")
    adults: int = Field(default=2, ge=1, le=10, description="Number of adult guests (minimum 1)")

def parse_date(date_str: str) -> date | None:
    """Parses date string supporting ISO YYYY-MM-DD and standard variations."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None

def check_availability(checkIn: str, checkOut: str, adults: int = 2) -> dict[str, Any]:
    """
    Deterministic mock tool to check room availability for given dates and guest count.
    
    Architecture rule:
    - Business rules, room filtering, date validation, and pricing arithmetic are kept
      100% deterministic inside this function, avoiding any LLM calculation hallucinations.
    """
    cin = parse_date(checkIn)
    cout = parse_date(checkOut)

    # 1. Validation checks
    if not cin:
        return {
            "available": False,
            "error": "invalid_check_in",
            "message": f"Invalid check-in date '{checkIn}'. Please specify a valid date in YYYY-MM-DD format."
        }

    if not cout:
        return {
            "available": False,
            "error": "invalid_check_out",
            "message": f"Invalid check-out date '{checkOut}'. Please specify a valid date in YYYY-MM-DD format."
        }

    if cout <= cin:
        return {
            "available": False,
            "error": "invalid_date_range",
            "message": f"Check-out date ({cout}) must be after check-in date ({cin})."
        }

    if adults < 1:
        return {
            "available": False,
            "error": "invalid_guest_count",
            "message": "At least 1 adult guest is required."
        }

    nights = (cout - cin).days

    # 2. Load ground-truth room inventory
    try:
        with open(HOTEL_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            rooms_data = data.get("rooms", [])
    except Exception:
        rooms_data = [
            {"id": "standard-queen", "type": "Standard Queen Room", "maxGuests": 2, "basePricePerNight": 4500},
            {"id": "deluxe-king", "type": "Deluxe King Room", "maxGuests": 3, "basePricePerNight": 6500},
            {"id": "family-suite", "type": "Family Executive Suite", "maxGuests": 4, "basePricePerNight": 9500},
            {"id": "presidential-suite", "type": "Maharaja Presidential Suite", "maxGuests": 5, "basePricePerNight": 18000},
        ]

    # 3. Capacity filtering (only rooms that can fit requested adult count)
    suitable_rooms = []
    for r in rooms_data:
        max_capacity = r.get("maxGuests", 2)
        if max_capacity >= adults:
            # Deterministic mock availability condition:
            # All rooms are available except Maharaja Presidential Suite on the 31st (monthly maintenance)
            is_maintenance = (cin.day == 31 and r.get("id") == "presidential-suite")
            
            price_per_night = r.get("basePricePerNight", 4500)
            suitable_rooms.append({
                "id": r.get("id"),
                "type": r.get("type"),
                "maxGuests": max_capacity,
                "bedType": r.get("bedType", ""),
                "description": r.get("description", ""),
                "pricePerNight": price_per_night,
                "totalPrice": price_per_night * nights,
                "available": not is_maintenance
            })

    # Available rooms
    available_rooms = [rm for rm in suitable_rooms if rm["available"]]

    if not available_rooms:
        if adults > 5:
            return {
                "available": False,
                "checkIn": cin.isoformat(),
                "checkOut": cout.isoformat(),
                "nights": nights,
                "adults": adults,
                "rooms": [],
                "message": f"Our largest single suite accommodates up to 5 guests. For a party of {adults}, please contact our concierge (+91 832 249 8000 / +91 98201 12345) to reserve connecting rooms."
            }
        return {
            "available": False,
            "checkIn": cin.isoformat(),
            "checkOut": cout.isoformat(),
            "nights": nights,
            "adults": adults,
            "rooms": [],
            "message": f"Sorry, no rooms matching {adults} guest(s) are available from {cin} to {cout}."
        }

    return {
        "available": True,
        "checkIn": cin.isoformat(),
        "checkOut": cout.isoformat(),
        "nights": nights,
        "adults": adults,
        "rooms": available_rooms,
        "message": f"Found {len(available_rooms)} room options available for {adults} guest(s) from {cin} to {cout} ({nights} night{'s' if nights > 1 else ''})."
    }
