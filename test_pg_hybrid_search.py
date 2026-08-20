from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder

from retrieval.pg_vector_store import PGVectorStore
from retrieval.pg_hybrid_search import PGHybridRetriever


def main():

    print("=" * 60)
    print("PGVECTOR + BM25 HYBRID RETRIEVAL TEST")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Load embedder
    # ----------------------------------------------------------

    print("\n[1] Loading embedding model...")

    embedder = Embedder(
        EmbeddingConfig(
            model_name="all-MiniLM-L6-v2"
        )
    )

    print("Embedding model loaded.")

    # ----------------------------------------------------------
    # 2. Connect to existing PostgreSQL vector store
    # ----------------------------------------------------------

    print("\n[2] Creating PGVectorStore...")

    store = PGVectorStore()

    print(
        "Total PostgreSQL chunks:",
        store.count()
    )

    # ----------------------------------------------------------
    # 3. Create hybrid retriever
    # ----------------------------------------------------------

    print("\n[3] Creating PGHybridRetriever...")

    retriever = PGHybridRetriever(
        vector_store=store,
        auto_refresh=False
    )

    print(
        "BM25 indexed chunks:",
        retriever.indexed_chunk_count
    )

    # ----------------------------------------------------------
    # 4. Test query
    # ----------------------------------------------------------

    query = "What is Python?"

    print("\n[4] Running hybrid search...")
    print("Query:", query)

    result = retriever.hybrid_search(
        query=query,
        embedder=embedder,
        top_k=5
    )

    # ----------------------------------------------------------
    # 5. Display results
    # ----------------------------------------------------------

    print("\n[5] RESULTS")

    print(
        "Retrieved chunks:",
        result.retrieved_count
    )

    for i, item in enumerate(
        result.retrieved_chunks,
        start=1
    ):

        print("\n" + "-" * 50)

        print("Result:", i)

        print(
            "Vector score:",
            item.vector_score
        )

        print(
            "BM25 score:",
            item.bm25_score
        )

        print(
            "Source:",
            item.chunk.metadata.get(
                "filename"
            )
        )

        print(
            "Document ID:",
            item.chunk.metadata.get(
                "document_id"
            )
        )

        print(
            "Chunk ID:",
            item.chunk.metadata.get(
                "chunk_id"
            )
        )

        print(
            "Chunk Index:",
            item.chunk.metadata.get(
                "chunk_index"
            )
        )

        print(
            "Content:",
            item.chunk.text[:300]
        )

    # ----------------------------------------------------------
    # 6. Test second query
    # ----------------------------------------------------------

    query = "Python readability and simplicity"

    print("\n" + "=" * 60)
    print("[6] SECOND QUERY")
    print("=" * 60)

    print("Query:", query)

    result = retriever.hybrid_search(
        query=query,
        embedder=embedder,
        top_k=5
    )

    print(
        "Retrieved chunks:",
        result.retrieved_count
    )

    for i, item in enumerate(
        result.retrieved_chunks,
        start=1
    ):

        print(
            f"{i}. "
            f"Vector={item.vector_score:.4f} "
            f"BM25={item.bm25_score:.4f} "
            f"ChunkID={item.chunk.metadata.get('chunk_id')}"
        )

    # ----------------------------------------------------------
    # 7. Final validation
    # ----------------------------------------------------------

    print("\n" + "=" * 60)

    if (
        store.count() > 0
        and retriever.indexed_chunk_count > 0
        and result.retrieved_count > 0
    ):
        print(
            "ALL PG HYBRID RETRIEVAL TESTS PASSED"
        )
    else:
        print(
            "PG HYBRID RETRIEVAL TEST FAILED"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()