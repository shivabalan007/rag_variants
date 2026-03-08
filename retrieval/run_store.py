import numpy as np

from ingestion.run_ingestion import ingest
from chunking.base import ChunkConfig
from chunking.fixed_chunker import fixed_chunk_document
from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore

if __name__ == "__main__":
    # Ingest documents
    docs = ingest("data/test1.txt")

    # Chunk documents
    chunk_config = ChunkConfig(chunk_size=300, overlap=50)

    chunks = []
    for doc in docs:
        chunks.extend(fixed_chunk_document(doc,chunk_config))

    print(f"Total chunks created: {len(chunks)}")

    # Embed chunks
    embedder = Embedder(
        EmbeddingConfig(model_name="all-MiniLM-L6-v2")
    )
    embeddings = embedder.embed_documents(chunks)

    # 4. Store vectors
    dim = embeddings.shape[1]
    store = VectorStore(dim)
    store.add(embeddings)

    print(f"Stored {store.index.ntotal} vectors")

    # 5. Dummy search (self-check)
    query_vector = embeddings[0:1]
    scores, indices = store.search(query_vector, top_k=3)

    print("Scores:", scores)
    print("Indices:", indices)