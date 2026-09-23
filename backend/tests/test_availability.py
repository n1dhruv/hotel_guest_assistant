import pytest
from app.core.tools import check_availability, parse_date

def test_parse_date_valid():
    d1 = parse_date("2026-10-15")
    assert d1 is not None
    assert d1.year == 2026
    assert d1.month == 10
    assert d1.day == 15

    d2 = parse_date("15-10-2026")
    assert d2 is not None
    assert d2.day == 15

def test_parse_date_invalid():
    assert parse_date("not-a-date") is None
    assert parse_date("") is None

def test_check_availability_normal():
    result = check_availability(checkIn="2026-10-15", checkOut="2026-10-18", adults=2)
    assert result["available"] is True
    assert result["nights"] == 3
    assert result["adults"] == 2
    assert len(result["rooms"]) >= 1

    # Verify pricing formula: totalPrice == pricePerNight * nights
    for rm in result["rooms"]:
        assert rm["totalPrice"] == rm["pricePerNight"] * 3
        assert rm["maxGuests"] >= 2

def test_capacity_filtering_for_three_guests():
    result = check_availability(checkIn="2026-10-15", checkOut="2026-10-17", adults=3)
    assert result["available"] is True
    # Standard Queen Room only accommodates 2 guests, so it must NOT be in the results
    room_types = [r["type"] for r in result["rooms"]]
    assert "Standard Queen Room" not in room_types
    assert "Deluxe King Room" in room_types or "Family Executive Suite" in room_types

def test_invalid_date_range():
    # Checkout before checkin
    result = check_availability(checkIn="2026-10-20", checkOut="2026-10-18", adults=2)
    assert result["available"] is False
    assert result["error"] == "invalid_date_range"

def test_invalid_date_format():
    result = check_availability(checkIn="invalid-date", checkOut="2026-10-18", adults=2)
    assert result["available"] is False
    assert result["error"] == "invalid_check_in"

def test_zero_or_negative_guests():
    result = check_availability(checkIn="2026-10-15", checkOut="2026-10-18", adults=0)
    assert result["available"] is False
    assert result["error"] == "invalid_guest_count"

def test_exceeding_max_hotel_capacity():
    result = check_availability(checkIn="2026-10-15", checkOut="2026-10-18", adults=8)
    assert result["available"] is False
    assert "concierge" in result["message"].lower()

def test_deterministic_maintenance_date():
    # Day 31: Presidential Penthouse should not be available
    result = check_availability(checkIn="2026-10-31", checkOut="2026-11-02", adults=5)
    room_ids = [r["id"] for r in result["rooms"]]
    assert "presidential-suite" not in room_ids
