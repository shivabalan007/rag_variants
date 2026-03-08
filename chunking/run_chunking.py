from ingestion.run_ingestion import ingest
from chunking.base import ChunkConfig
from chunking.fixed_chunker import fixed_chunk_document


if __name__ == "__main__":
    docs = ingest("data/test1.txt")

    config = ChunkConfig(
        chunk_size=300,
        overlap=50
    )

    all_chunks = []

    for doc in docs:
        chunks = fixed_chunk_document(doc,config)
        all_chunks.extend(chunks)

    print(f"Total documents ingested: {len(docs)}")
    print(f"Total chunks created: {len(all_chunks)}")

    for c in all_chunks[:3]:
        print("---")
        print(c.metadata)
        print(c.text[:200])