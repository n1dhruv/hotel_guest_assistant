"""
CLI Runner and Inspection Tool for Reranking Pipeline.
Reranks Top 10 hybrid candidates into refined Top 3-5 chunks using
NVIDIA Llama Nemotron Rerank VL 1B V2 via OpenRouter.

Examples:
    python reranker.py "Can I bring my golden retriever?"
    python reranker.py "What are the pool hours?" --top-n 3 --breakdown
    python reranker.py -q "Aadhaar card verification" --top-n 4
    python reranker.py --stats
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

from app.core.hybrid_retriever import get_hybrid_retriever
from app.core.reranker import (
    BaseReranker,
    DeterministicLocalReranker,
    OpenRouterNemotronReranker,
    get_reranker,
)
from app.core.vector_store import get_embedder, get_vector_store

__all__ = [
    "BaseReranker",
    "DeterministicLocalReranker",
    "OpenRouterNemotronReranker",
    "get_reranker",
]


def main():
    parser = argparse.ArgumentParser(
        description="Reranker CLI (NVIDIA Llama Nemotron Rerank VL 1B V2 via OpenRouter)"
    )
    parser.add_argument(
        "query_pos",
        nargs="?",
        default=None,
        help="Query to search and rerank (e.g. 'What is the WiFi speed?')",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="",
        help="Query string (alternative to positional argument)",
    )
    parser.add_argument(
        "--top-n",
        "-n",
        type=int,
        default=3,
        help="Number of refined reranked chunks to return (default: 3)",
    )
    parser.add_argument(
        "--candidate-k",
        "-k",
        type=int,
        default=10,
        help="Number of initial hybrid candidates to retrieve before reranking (default: 10)",
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
        help="Disable HyDE query expansion for initial dense channel",
    )
    parser.add_argument(
        "--breakdown",
        "-b",
        action="store_true",
        help="Display rank transitions and score delta (Initial RRF Rank -> Reranked Rank)",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Display Reranker operational statistics",
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
        help="Embedding provider for vector channel (fastembed, auto, openrouter, nemotron, deterministic)",
    )

    args = parser.parse_args()

    reranker = get_reranker()

    if args.stats:
        print("\n================ Reranker Operational Statistics ================")
        print(json.dumps(reranker.get_stats(), indent=2))
        print("=================================================================\n")
        if not args.query and not args.query_pos:
            return

    query_text = (args.query or args.query_pos or "").strip()
    if not query_text:
        print("No query provided. Usage: python reranker.py 'Can I bring my pet?'")
        return

    vs = None
    if args.provider:
        embedder = get_embedder(provider=args.provider)
        vs = get_vector_store(force_new=True, embedder=embedder)

    hybrid_retriever = get_hybrid_retriever(
        vector_store=vs,
        force_new=bool(args.provider),
        use_hyde=not args.no_hyde,
    )

    # Stage 1: Extract initial hybrid candidates (Top 10)
    candidates = hybrid_retriever.retrieve_candidates(
        query=query_text,
        top_k=args.candidate_k,
        category_filter=args.category,
        use_hyde=not args.no_hyde,
    )

    # Stage 2: Cross-encoder rerank to Top N
    reranked = reranker.rerank(
        query=query_text,
        candidates=candidates,
        top_n=args.top_n,
    )

    if args.json:
        payload = {
            "query": query_text,
            "initial_candidates_count": len(candidates),
            "reranked_count": len(reranked),
            "results": reranked,
            "stats": reranker.get_stats(),
        }
        print(json.dumps(payload, indent=2))
        return

    print("\n" + "=" * 68)
    print(f"Query: \"{query_text}\"")
    print(f"Reranker Model: {reranker.model_name}")
    print(f"Stage 1 Candidates: {len(candidates)} -> Stage 2 Reranked: {len(reranked)}")
    print("=" * 68)

    if not reranked:
        print("   No matching chunks found.")
        return

    for item in reranked:
        rank = item.get("rerank_rank", "?")
        score = item.get("rerank_score", 0.0)
        category = item.get("category", "").upper()
        title = item.get("title", "")
        topic = item.get("topic", "")
        prior_rank = item.get("rrf_rank", "?")
        delta = item.get("rerank_delta", 0)
        delta_str = f"+{delta}" if delta > 0 else str(delta)

        print(f"\n[{rank}] Rerank Score: {score:.6f} | [{category}] {title}")
        print(f"    Topic: {topic} | Rank Shift: {prior_rank} -> {rank} ({delta_str})")
        if args.breakdown:
            channels = ", ".join(item.get("channels", []))
            print(f"    Initial Channels: {channels} | RRF Score: {item.get('rrf_score', 'N/A')}")
            if "dense_score" in item:
                print(f"    Dense Cosine: {item.get('dense_score')}")
            if "bm25_score" in item:
                print(f"    BM25 Score: {item.get('bm25_score')}")
        content_preview = item.get("content", "")[:130].replace("\n", " ")
        print(f"    Snippet: {content_preview}...")

    print("\n" + "-" * 68 + "\n")


if __name__ == "__main__":
    main()
