"""
Top-level runner / export for json_ingest.
Allows running `python json_ingest.py` directly from backend or repo root.
"""

import json
from pathlib import Path
import sys

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.json_ingest import json_to_documents, maybe_split, LONG_TEXT_THRESHOLD

__all__ = ["json_to_documents", "maybe_split", "LONG_TEXT_THRESHOLD"]

if __name__ == "__main__":
    data_path = BASE_DIR / "app" / "data" / "hotel_data.json"
    if not data_path.exists():
        data_path = BASE_DIR / "hotel_data.json"

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = json_to_documents(data)
    print(f"Generated {len(docs)} LangChain Documents from {data_path.name}:\n")
    for i, doc in enumerate(docs[:6], start=1):
        print(f"[{i}] Category: {doc.metadata.get('category')} | Metadata: {doc.metadata}")
        print(f"    Content: {doc.page_content}\n")
