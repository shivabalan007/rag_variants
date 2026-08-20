import uuid
import numpy as np

from ingestion.run_ingestion import ingest
from ingestion.base import Document
from chunking.semantic_chunker import semantic_chunk
from chunking.sliding_window_chunker import sliding_window_chunk
from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder

from retrieval.pg_vector_store import PGVectorStore


TEST_DOCUMENT_ID = f"pgvector-test-{uuid.uuid4()}"
TEST_FILENAME = "test2.txt"


def main():

    print("=" * 60)
    print("PGVECTOR STORE TEST")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Create embedder
    # ----------------------------------------------------------

    print("\n[1] Loading embedding model...")

    embedder = Embedder(
        EmbeddingConfig(
            model_name="all-MiniLM-L6-v2"
        )
    )

    print("Embedding model loaded.")

    # ----------------------------------------------------------
    # 2. Ingest test document
    # ----------------------------------------------------------

    print("\n[2] Ingesting document...")

    docs = ingest(
        "data/test2.txt"
    )

    print(f"Documents loaded: {len(docs)}")

    # ----------------------------------------------------------
    # 3. Chunk document
    # ----------------------------------------------------------

    print("\n[3] Creating chunks...")

    chunks = []

    for doc in docs:

        semantic_chunks = semantic_chunk(
            doc.text
        )

        for semantic_chunk_text in semantic_chunks:

            window_chunks = sliding_window_chunk(
                semantic_chunk_text,
                chunk_size=300,
                overlap=50
            )

            for chunk in window_chunks:

                chunks.append(
                    Document(
                        text=chunk,
                        metadata=doc.metadata
                    )
                )

    print(f"Total chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError(
            "No chunks were created."
        )

    # ----------------------------------------------------------
    # 4. Generate embeddings
    # ----------------------------------------------------------

    print("\n[4] Generating embeddings...")

    embeddings = embedder.embed_documents(
        chunks
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # ----------------------------------------------------------
    # 5. Create PGVectorStore
    # ----------------------------------------------------------

    print("\n[5] Creating PGVectorStore...")

    store = PGVectorStore()

    # ----------------------------------------------------------
    # 6. Clean previous test data
    # ----------------------------------------------------------

    print("\n[6] Cleaning previous test data...")

    deleted = store.delete_document(
        TEST_DOCUMENT_ID
    )

    print(
        f"Deleted old test rows: {deleted}"
    )

    # ----------------------------------------------------------
    # 7. Insert document
    # ----------------------------------------------------------

    print("\n[7] Inserting chunks into PostgreSQL...")

    inserted = store.add(
        document_id=TEST_DOCUMENT_ID,
        filename=TEST_FILENAME,
        chunks=chunks,
        embeddings=embeddings
    )

    print(
        f"Inserted chunks: {inserted}"
    )

    # ----------------------------------------------------------
    # 8. Verify document exists
    # ----------------------------------------------------------

    print("\n[8] Checking document existence...")

    exists = store.document_exists(
        TEST_DOCUMENT_ID
    )

    print(
        f"Document exists: {exists}"
    )

    # ----------------------------------------------------------
    # 9. Check chunk count
    # ----------------------------------------------------------

    chunk_count = store.document_chunk_count(
        TEST_DOCUMENT_ID
    )

    print(
        f"Document chunk count: {chunk_count}"
    )

    # ----------------------------------------------------------
    # 10. Test vector search
    # ----------------------------------------------------------

    print("\n[9] Testing pgvector similarity search...")

    query = "What is Python?"

    query_embedding = embedder.embed_query(
        query
    )

    scores, ids = store.search(
        query_vector=query_embedding,
        top_k=3
    )

    print(
        f"\nQuery: {query}"
    )

    print(
        f"Results returned: {len(ids)}"
    )

    for rank, (score, row_id) in enumerate(
        zip(scores, ids),
        start=1
    ):

        print(
            f"\nResult {rank}"
        )

        print(
            f"PostgreSQL ID : {row_id}"
        )

        print(
            f"Similarity    : {score:.4f}"
        )

    # ----------------------------------------------------------
    # 11. Retrieve actual chunks
    # ----------------------------------------------------------

    print("\n[10] Retrieving matched chunks...")

    results = store.get_chunks(
        ids
    )

    for rank, row in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- Result {rank} ---"
        )

        print(
            f"ID          : {row.id}"
        )

        print(
            f"Document ID : {row.document_id}"
        )

        print(
            f"Filename    : {row.filename}"
        )

        print(
            f"Chunk Index : {row.chunk_index}"
        )

        print(
            f"Content     : {row.content[:300]}"
        )

    # ----------------------------------------------------------
    # 12. Check total database count
    # ----------------------------------------------------------

    print("\n[11] Database count...")

    total = store.count()

    print(
        f"Total chunks in PostgreSQL: {total}"
    )

    # ----------------------------------------------------------
    # 13. Cleanup
    # ----------------------------------------------------------

    print("\n[12] Cleaning test document...")

    # deleted = store.delete_document(
    #     TEST_DOCUMENT_ID
    # )

    # print(
    #     f"Deleted test chunks: {deleted}"
    # )

    # exists_after_delete = store.document_exists(
    #     TEST_DOCUMENT_ID
    # )

    # print(
    #     f"Document exists after cleanup: "
    #     f"{exists_after_delete}"
    # )

    # ----------------------------------------------------------
    # Final verification
    # ----------------------------------------------------------

    print("\n" + "=" * 60)

    if (
        inserted == len(chunks)
        and exists
        and chunk_count == len(chunks)
        and len(ids) > 0
        and not exists_after_delete
    ):

        print("ALL PGVECTOR TESTS PASSED")

    else:

        print("PGVECTOR TEST FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()