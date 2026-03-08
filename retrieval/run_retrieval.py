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
    doc_embeddings = embedder.embed_documents(chunks)

    # 4. Store vectors
    store = VectorStore(doc_embeddings.shape[1])
    store.add(doc_embeddings)

    # 5. User query
    query = "What is the main topic of the document?"

    # 6 . Embed query
    query_embedding = embedder.embed_documents(
        [type(chunks[0])(text=query, metadata={})]
    )

    # 7. Search
    scores, indices = store.search(query_embedding, top_k=3)


    print("Scores:", scores)
    print("Indices:", indices)

    # 8. Map results back to chunks
    print("\nTop chunks:\n")
    for idx, score in zip(indices[0], scores[0]):
        chunk = chunks[idx]
        print(f"Score: {score:.4f}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Chunk text: {chunk.text[:200]}...\n")
        print("--------------")