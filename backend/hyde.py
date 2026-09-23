"""
CLI Runner and Inspection Tool for HyDE (Hypothetical Document Embeddings).
Allows running `python hyde.py` directly from backend or repo root.

Examples:
    python hyde.py "Can I bring my golden retriever?"
    python hyde.py -q "What time can we check in?" --compare
    python hyde.py --stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.hyde import (
    HYDE_SYSTEM_PROMPT,
    HYDE_USER_PROMPT_TEMPLATE,
    HyDEGenerator,
    get_hyde_generator,
)

__all__ = [
    "HyDEGenerator",
    "get_hyde_generator",
    "HYDE_SYSTEM_PROMPT",
    "HYDE_USER_PROMPT_TEMPLATE",
]


def main():
    parser = argparse.ArgumentParser(
        description="HyDE (Hypothetical Document Embeddings) Query Pipeline CLI"
    )
    parser.add_argument(
        "query_pos",
        nargs="?",
        default=None,
        help="Guest question to expand with HyDE (e.g. 'Can I bring my golden retriever?')",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="",
        help="Guest question (alternative to positional argument)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="LiteLLM model to use (default: settings.LLM_MODEL)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=None,
        help="Timeout in seconds for generation (default: 1.5s)",
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare direct Qdrant vector retrieval vs HyDE-enhanced vector retrieval",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve during --compare (default: 3)",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Display HyDE generator configuration and runtime statistics",
    )

    args = parser.parse_args()

    generator = get_hyde_generator(
        model=args.model,
        timeout=args.timeout,
    )

    if args.stats:
        print("\n================ HyDE Pipeline Statistics ================")
        print(json.dumps(generator.get_stats(), indent=2))
        print("==========================================================\n")
        if not args.query and not args.query_pos:
            return

    query_text = (args.query or args.query_pos or "").strip()
    if not query_text:
        print("No query provided. Usage: python hyde.py 'Can I bring my golden retriever?'")
        return

    print("\n------------------------------------------------------------")
    print(f"Guest Query: \"{query_text}\"")
    print("------------------------------------------------------------")

    start_time = time.perf_counter()
    hypo_doc = generator.generate_hypothetical_document(query_text)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    is_fallback = hypo_doc.strip() == query_text.strip()
    print(f"\n[Generated Hypothetical Passage] (Latency: {latency_ms}ms, Fallback: {is_fallback}):")
    print(f"  \"{hypo_doc}\"\n")

    if args.compare:
        print("------------------------------------------------------------")
        print(f"Retrieval Comparison (Top {args.top_k} Chunks):")
        print("------------------------------------------------------------")
        try:
            from app.core.vector_store import get_vector_store

            store = get_vector_store()

            # 1. Direct Search with raw query
            raw_results = store.search(query_text, top_k=args.top_k, use_hyde=False)
            print("\n1. Direct Vector Search (Raw Query):")
            for i, r in enumerate(raw_results, 1):
                print(f"   [{i}] (Score: {r['score']}) {r['title']} [{r['category']}]")

            # 2. HyDE-Enhanced Search
            hyde_results = store.search(query_text, top_k=args.top_k, use_hyde=True)
            print("\n2. HyDE-Enhanced Vector Search:")
            for i, r in enumerate(hyde_results, 1):
                print(f"   [{i}] (Score: {r['score']}) {r['title']} [{r['category']}]")
            print()

        except Exception as e:
            print(f"Could not run vector retrieval comparison: {e}")


if __name__ == "__main__":
    main()
