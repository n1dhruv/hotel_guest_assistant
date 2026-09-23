"""
Top-level runner and migration CLI for Qdrant Vector Store.
Allows running `python vector_store.py` directly from backend or repo root.

Examples:
    python vector_store.py --stats
    python vector_store.py --reindex
    python vector_store.py --query "What time is check-in?"
    python vector_store.py --query "swimming pool" --top-k 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.vector_store import (
    BaseEmbedder,
    DeterministicLocalEmbedder,
    FastEmbedAdapter,
    GeminiEmbeddingAdapter,
    LiteLLMGeminiEmbeddingAdapter,
    NemotronEmbeddingAdapter,
    OpenAIEmbeddingAdapter,
    OpenRouterNemotronEmbeddingAdapter,
    QdrantVectorStore,
    chunk_id_to_uuid,
    get_embedder,
    get_vector_store,
)

__all__ = [
    "QdrantVectorStore",
    "get_vector_store",
    "get_embedder",
    "BaseEmbedder",
    "FastEmbedAdapter",
    "OpenRouterNemotronEmbeddingAdapter",
    "NemotronEmbeddingAdapter",
    "GeminiEmbeddingAdapter",
    "LiteLLMGeminiEmbeddingAdapter",
    "OpenAIEmbeddingAdapter",
    "DeterministicLocalEmbedder",
    "chunk_id_to_uuid",
]


def main():
    parser = argparse.ArgumentParser(description="Qdrant Vector Database Manager & Query CLI")
    parser.add_argument("--reindex", action="store_true", help="Force rebuild and reindex hotel knowledge base")
    parser.add_argument("--query", "-q", type=str, default="", help="Query to search against vector store")
    parser.add_argument("--top-k", "-k", type=int, default=4, help="Number of results to retrieve (default: 4)")
    parser.add_argument("--category", "-c", type=str, default=None, help="Filter by category (property, amenity, room, policy, faq)")
    parser.add_argument("--provider", "-p", type=str, default="auto", help="Embedding provider (auto, openrouter, nemotron, openai, fastembed, deterministic)")
    parser.add_argument("--in-memory", action="store_true", help="Run with ephemeral in-memory storage (:memory:)")
    parser.add_argument("--stats", action="store_true", help="Display vector store and collection statistics")
    parser.add_argument("--hyde", action="store_true", help="Use HyDE (Hypothetical Document Embeddings) for query expansion")

    args = parser.parse_args()

    print(f"🔧 Initializing Qdrant Vector Store [provider={args.provider}, in_memory={args.in_memory}]...")
    embedder = get_embedder(provider=args.provider)
    print(f"   Active Embedder: {embedder.provider_name} (dimension: {embedder.dimension})")

    store = QdrantVectorStore(
        embedder=embedder,
        in_memory=args.in_memory,
        force_reindex=args.reindex
    )

    if args.reindex:
        print("🔄 Reindexing hotel knowledge base into Qdrant...")
        count = store.index_from_file(force=True)
        print(f"✅ Successfully indexed {count} semantic chunks into '{store.collection_name}'.")

    point_count = store.get_point_count()

    if args.stats or (not args.query and not args.reindex):
        print("\n📊 --- Qdrant Vector Store Status ---")
        print(f"   Collection Name: {store.collection_name}")
        print(f"   Points in Collection: {point_count}")
        print(f"   Vector Dimension: {embedder.dimension}")
        print(f"   Embedding Provider: {embedder.provider_name}")
        print(f"   Ground Truth File: {store.data_path}")
        print("--------------------------------------\n")

    if args.query:
        print(f"\n🔍 Searching for: \"{args.query}\" (top_k={args.top_k}, category={args.category}, hyde={args.hyde})")
        results = store.search(
            query=args.query,
            top_k=args.top_k,
            category_filter=args.category,
            use_hyde=args.hyde
        )
        if not results:
            print("   No matching chunks found.")
        for idx, res in enumerate(results, start=1):
            print(f"\n[{idx}] Score: {res['score']:.4f} | [{res['category'].upper()}] {res['title']}")
            print(f"    Topic: {res.get('topic')} | ID: {res.get('id')}")
            print(f"    Content: {res['content'][:140]}...")

    store.close()


if __name__ == "__main__":
    main()
