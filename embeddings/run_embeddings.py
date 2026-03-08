from ingestion.run_ingestion import ingest
from chunking.base import ChunkConfig
from chunking.fixed_chunker import fixed_chunk_document

from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder 

if __name__ == "__main__":
    docs = ingest("data/test1.txt")

    #Chunk documents
    chunk_config = ChunkConfig(chunk_size=300, overlap=50)

    chunks = []
    for doc in docs:
        chunks.extend(fixed_chunk_document(doc,chunk_config))

    print(f"Total chunks created: {len(chunks)}")

    #Embed chunks
    embed_config = EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        normalize=True
    )

    embedder = Embedder(embed_config)
    embeddings = embedder.embed_documents(chunks)

    print(f"Embeddings shape: {embeddings.shape}")


    