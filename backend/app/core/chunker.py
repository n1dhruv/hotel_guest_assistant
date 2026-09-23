"""
Semantic Chunker for Hotel Ground Truth Data.

Transforms raw hotel JSON into self-contained, semantically rich knowledge units
with structured metadata, domain keywords, and full contextual grounding.
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any


@dataclass
class SemanticChunk:
    chunk_id: str
    category: str  # 'property', 'amenity', 'room', 'policy', 'faq'
    topic: str
    title: str
    keywords: list[str]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = self.chunk_id  # compatibility alias
        return d

    @property
    def document_text(self) -> str:
        """Enriched text representation for embeddings and vector DB indexing."""
        kw_str = f" (Keywords: {', '.join(self.keywords)})" if self.keywords else ""
        return f"{self.title}: {self.content}{kw_str}"


class SemanticHotelChunker:
    """
    Parses hotel JSON data and chunks it into self-contained semantic units.
    Each chunk is enriched with property anchoring, normalized keywords,
    and structured metadata to ensure maximum retrieval recall and precision.
    """

    def __init__(self, data_path: Path | str):
        self.data_path = Path(data_path)

    def to_documents(self, embedding_model: Any = None, long_text_threshold: int = 600):
        """Converts hotel JSON directly to LangChain Document objects via json_to_documents."""
        from app.core.json_ingest import json_to_documents
        data = self.load_data()
        return json_to_documents(data, embedding_model=embedding_model, long_text_threshold=long_text_threshold)

    def load_data(self) -> dict[str, Any]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_chunks(self) -> list[SemanticChunk]:
        data = self.load_data()
        chunks: list[SemanticChunk] = []

        prop = data.get("property", {})
        prop_name = prop.get("name", "The Grand Azure Heritage Resort & Spa")
        prop_loc = "Candolim Beach, North Goa"

        # 1. Property Identity & Master Overview Chunk
        contact = prop.get("contact", {})
        prop_content = (
            f"{prop_name} is a luxury beachfront resort situated directly on {prop_loc} "
            f"({prop.get('address')}). {prop.get('tagline')}. "
            f"Standard check-in time is {prop.get('checkIn')} and check-out time is {prop.get('checkOut')}. "
            f"Front desk & Concierge contact: Phone {contact.get('phone')}, Mobile {contact.get('mobile')}, "
            f"Email: {contact.get('email')}."
        )
        chunks.append(
            SemanticChunk(
                chunk_id="property-overview",
                category="property",
                topic="property",
                title=f"About {prop_name}",
                keywords=[
                    "hotel", "resort", "overview", "location", "address", "candolim", "beach",
                    "goa", "contact", "phone", "email", "mobile", "front desk", "concierge"
                ],
                content=prop_content,
                metadata={
                    "section": "property",
                    "name": prop_name,
                    "checkIn": prop.get("checkIn"),
                    "checkOut": prop.get("checkOut"),
                    "phone": contact.get("phone"),
                    "email": contact.get("email"),
                }
            )
        )

        # 2. Amenity Semantic Chunks
        amenity_keyword_map = {
            "Oceanview Infinity Pool": [
                "pool", "swimming", "infinity pool", "swimming pool", "oceanview",
                "towels", "sun loungers", "temperature controlled", "beverages"
            ],
            "AyurVeda Spa & Fitness Center": [
                "spa", "gym", "fitness", "workout", "cardio", "massage", "ayurveda",
                "steam", "sauna", "yoga", "wellness", "health", "24/7 gym"
            ],
            "Complimentary Royal Buffet Breakfast": [
                "breakfast", "buffet", "morning meal", "food", "dining", "south indian",
                "north indian", "dosa", "idli", "continental", "vegetarian", "pure veg",
                "jain", "tea", "coffee", "complimentary breakfast"
            ],
            "High-Speed Wi-Fi": [
                "wifi", "wi-fi", "internet", "fiber", "speed", "500 mbps", "wireless", "broadband"
            ],
            "Valet & EV Parking": [
                "parking", "valet", "car", "vehicle", "ev", "electric vehicle",
                "charging", "charger", "tata power", "security"
            ],
            "Saffron Coastal & Spice Restaurant": [
                "restaurant", "dining", "food", "dinner", "lunch", "cuisine",
                "goan", "seafood", "thali", "tandoor", "vegetarian", "pure veg", "jain", "saffron"
            ]
        }

        amenity_topic_map = {
            "Oceanview Infinity Pool": "pool",
            "AyurVeda Spa & Fitness Center": "spa_gym",
            "Complimentary Royal Buffet Breakfast": "breakfast",
            "High-Speed Wi-Fi": "wifi",
            "Valet & EV Parking": "parking",
            "Saffron Coastal & Spice Restaurant": "dining"
        }

        for idx, a in enumerate(data.get("amenities", [])):
            name = a.get("name", "")
            hours = a.get("hours", "")
            desc = a.get("description", "")
            topic = amenity_topic_map.get(name, f"amenity_{idx}")
            kws = amenity_keyword_map.get(name, ["amenity", name.lower()])

            content = (
                f"{name} at {prop_name} ({prop_loc}). "
                f"Operating Hours: {hours}. Details: {desc}"
            )

            chunks.append(
                SemanticChunk(
                    chunk_id=f"amenity-{idx}-{name.lower().replace(' ', '-')[:20]}",
                    category="amenity",
                    topic=topic,
                    title=f"Amenity: {name}",
                    keywords=kws,
                    content=content,
                    metadata={"section": "amenities", "hours": hours, "amenity_name": name}
                )
            )

        # 3. Room Tier Semantic Chunks
        room_keyword_map = {
            "standard-queen": ["standard queen", "queen bed", "room", "cheapest room", "budget", "single room", "tariff"],
            "deluxe-king": ["deluxe king", "king bed", "ocean view room", "balcony", "bathtub", "soaking tub", "3 guests"],
            "family-suite": ["family executive suite", "suite", "two bedroom", "family room", "4 guests", "twin beds", "kids"],
            "presidential-suite": ["maharaja presidential suite", "penthouse", "luxury suite", "private pool", "plunge pool", "butler", "5 guests", "bar"]
        }

        for r in data.get("rooms", []):
            rid = r.get("id", "")
            rtype = r.get("type", "")
            max_g = r.get("maxGuests", 2)
            price = r.get("basePricePerNight", 0)
            bed = r.get("bedType", "")
            desc = r.get("description", "")
            kws = room_keyword_map.get(rid, ["room", rtype.lower(), f"{max_g} guests"])

            content = (
                f"{rtype} at {prop_name} (ID: {rid}). "
                f"Accommodates up to {max_g} adult guests. "
                f"Base tariff: ₹{price:,} per night (taxes extra). "
                f"Bedding setup: {bed}. Room features: {desc}"
            )

            chunks.append(
                SemanticChunk(
                    chunk_id=f"room-{rid}",
                    category="room",
                    topic=f"room_{rid}",
                    title=f"Room: {rtype}",
                    keywords=kws + ["tariff", "price", "rate", "capacity", "occupancy"],
                    content=content,
                    metadata={
                        "section": "rooms",
                        "room_id": rid,
                        "room_type": rtype,
                        "max_guests": max_g,
                        "base_price": price,
                        "bed_type": bed
                    }
                )
            )

        # 4. Resort Policies Semantic Chunks
        policy_meta = {
            "cancellation": {
                "title": "Policy: Cancellation & Refund",
                "topic": "cancellation",
                "keywords": ["cancellation", "cancel", "refund", "cancelling", "charge", "free cancellation", "24 hours", "fee"]
            },
            "checkInCheckOut": {
                "title": "Policy: Check-in and Check-out Timings",
                "topic": "checkincheckout",
                "keywords": ["checkin", "checkout", "check-in", "check-out", "timings", "timing", "early checkin", "late checkout", "arrival", "departure"]
            },
            "idVerification": {
                "title": "Policy: ID Verification & Documentation",
                "topic": "id_verification",
                "keywords": ["id", "identification", "aadhaar", "passport", "voter id", "driving license", "pan card", "government id", "documents", "verification"]
            },
            "pets": {
                "title": "Policy: Pet Policy",
                "topic": "pets",
                "keywords": ["pets", "pet", "dog", "dogs", "cat", "cats", "animal", "animals", "pet friendly", "service animal", "guide dog"]
            },
            "children": {
                "title": "Policy: Child Occupancy & Extra Bed",
                "topic": "children",
                "keywords": ["children", "child", "kids", "extra bed", "infant", "rollaway bed", "family", "age", "charges", "complimentary"]
            },
            "smoking": {
                "title": "Policy: Smoking Policy",
                "topic": "smoking",
                "keywords": ["smoking", "smoke", "cigarettes", "cigarette", "cigar", "smoke-free", "sanitation fee", "penalty", "smoking zone"]
            },
            "payment": {
                "title": "Policy: Payment & Invoicing",
                "topic": "payment",
                "keywords": ["payment", "pay", "upi", "credit card", "debit card", "visa", "mastercard", "rupay", "amex", "gst", "invoice", "billing"]
            }
        }

        policies = data.get("policies", {})
        for pol_key, pol_val in policies.items():
            pm = policy_meta.get(pol_key, {
                "title": f"Policy: {pol_key.capitalize()}",
                "topic": pol_key.lower(),
                "keywords": ["policy", pol_key.lower()]
            })

            content = f"{prop_name} {pm['title']}: {pol_val}"

            chunks.append(
                SemanticChunk(
                    chunk_id=f"policy-{pol_key}",
                    category="policy",
                    topic=pm["topic"],
                    title=pm["title"],
                    keywords=pm["keywords"],
                    content=content,
                    metadata={"section": "policies", "policy_key": pol_key}
                )
            )

        # 5. FAQ Semantic Chunks
        faq_topic_map = {
            "faq-1": ("checkincheckout", ["checkin", "checkout", "hours", "timings", "early checkin", "late checkout"]),
            "faq-2": ("pool", ["pool", "swimming", "infinity pool", "oceanview", "pool hours"]),
            "faq-3": ("breakfast", ["breakfast", "buffet", "morning meal", "pure veg", "jain", "timing"]),
            "faq-4": ("room_recommendation", ["three guests", "3 people", "triple sharing", "room suitability", "deluxe king", "family suite"]),
            "faq-5": ("cancellation", ["cancellation policy", "cancel booking", "refund rule", "24 hours"]),
            "faq-6": ("parking", ["parking", "valet", "ev charging", "car parking", "tata power"]),
            "faq-7": ("pets", ["pets allowed", "bring dog", "pet policy", "service animal"]),
            "faq-8": ("id_verification", ["id proof", "aadhaar", "pan card", "government id", "documents"]),
            "faq-9": ("dining", ["pure veg food", "jain food", "vegetarian restaurant", "saffron"]),
            "faq-10": ("highlights", ["highlights", "benefits", "what is special", "amenities summary", "why stay"]),
            "faq-11": ("amenities_summary", ["facilities list", "all amenities", "what do you offer", "resort facilities"])
        }

        for faq in data.get("faqs", []):
            fid = faq.get("id", "")
            topic_str = faq.get("topic") or faq.get("title", "")
            answer_text = faq.get("answer") or faq.get("a", "")
            topic_key, default_kws = faq_topic_map.get(fid, (fid, ["faq", topic_str.lower()]))

            content = (
                f"Regarding {topic_str} at {prop_name}: {answer_text}"
            )

            chunks.append(
                SemanticChunk(
                    chunk_id=fid,
                    category="faq",
                    topic=topic_key,
                    title=f"FAQ: {topic_str}",
                    keywords=default_kws + [topic_str.lower()],
                    content=content,
                    metadata={"section": "faqs", "faq_id": fid, "topic": topic_str}
                )
            )

        return chunks
