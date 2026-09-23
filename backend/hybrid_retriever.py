"""
CLI Runner and Inspection Tool for Hybrid Search Engine (Vector + BM25 + RRF).
Allows running `python hybrid_retriever.py` directly from backend or repo root.

Examples:
    python hybrid_retriever.py "What is the WiFi speed?"
    python hybrid_retriever.py "Can I bring my golden retriever?" --breakdown
    python hybrid_retriever.py -q "Aadhaar card" --top-k 10 --breakdown
    python hybrid_retriever.py --stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.hybrid_retriever import (
    BM25Index,
    HybridRetriever,
    get_hybrid_retriever,
    reciprocal_rank_fusion,
)
from app.core.vector_store import get_embedder, get_vector_store

__all__ = [
    "HybridRetriever",
    "BM25Index",
    "reciprocal_rank_fusion",
    "get_hybrid_retriever",
]


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Search Engine (Dense Qdrant + Sparse BM25 + Reciprocal Rank Fusion) CLI"
    )
    parser.add_argument(
        "query_pos",
        nargs="?",
        default=None,
        help="Query to search (e.g. 'What is the WiFi speed?')",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="",
        help="Query string (alternative to positional argument)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Number of fused candidate chunks to return (default: 10)",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default=None,
        help="Filter results by category (property, amenity, room, policy, faq)",
    )
    parser.add_argument(
        "--no-hyde",
        action="store_true",
        help="Disable HyDE query expansion for dense channel",
    )
    parser.add_argument(
        "--breakdown",
        "-b",
        action="store_true",
        help="Display dual-channel ranking and scoring breakdown (Dense vs BM25)",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Display Hybrid Retriever index and operational statistics",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=None,
        help="Embedding provider for dense vector channel (auto, openrouter, nemotron, fastembed, deterministic)",
    )

    args = parser.parse_args()

    vs = None
    if args.provider:
        embedder = get_embedder(provider=args.provider)
        vs = get_vector_store(force_new=True, embedder=embedder)

    retriever = get_hybrid_retriever(
        vector_store=vs,
        force_new=bool(args.provider),
        use_hyde=not args.no_hyde,
    )

    if args.stats:
        print("\n================ Hybrid Search Pipeline Statistics ================")
        print(json.dumps(retriever.get_stats(), indent=2))
        print("===================================================================\n")
        if not args.query and not args.query_pos:
            return

    query_text = (args.query or args.query_pos or "").strip()
    if not query_text:
        print("No query provided. Usage: python hybrid_retriever.py 'What is the WiFi speed?'")
        return

    results = retriever.retrieve_candidates(
        query=query_text,
        top_k=args.top_k,
        category_filter=args.category,
        use_hyde=not args.no_hyde,
    )

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("\n------------------------------------------------------------")
    print(f"Hybrid Search Query: \"{query_text}\" (Top {len(results)} Chunks)")
    print(f"Channels: Dense Vector (HyDE={not args.no_hyde}) + Sparse BM25 | Fusion: RRF (k=60)")
    print("------------------------------------------------------------")

    if not results:
        print("No matching candidate chunks found.")
        return

    for rank, item in enumerate(results, 1):
        channels_str = " + ".join([c.upper() for c in item.get("channels", [])])
        print(f"\n[{rank}] RRF Score: {item['rrf_score']} | [{item['category'].upper()}] {item['title']}")
        print(f"    Topic: {item.get('topic', 'general')} | Channels: {channels_str}")

        if args.breakdown:
            d_rank = item.get("dense_rank")
            d_score = item.get("dense_score")
            b_rank = item.get("bm25_rank")
            b_score = item.get("bm25_score")
            d_str = f"Rank {d_rank} (Cosine: {d_score})" if d_rank else "Not in top 10"
            b_str = f"Rank {b_rank} (Score: {b_score})" if b_rank else "Not in top 10"
            print(f"    -> Dense Channel: {d_str}")
            print(f"    -> BM25 Channel:  {b_str}")

        content_preview = item.get("content", "").replace("\n", " ")
        if len(content_preview) > 130:
            content_preview = content_preview[:127] + "..."
        print(f"    Snippet: {content_preview}")

    print("\n------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
