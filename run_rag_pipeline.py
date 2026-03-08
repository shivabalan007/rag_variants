from orchestration.rag_pipeline import RAGPipeline

from ingestion.run_ingestion import ingest
from chunking.base import ChunkConfig
from chunking.fixed_chunker import fixed_chunk_document

from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder

from retrieval.vector_store import VectorStore


def main():

    # 1️⃣ Ingest
    docs = ingest("data/test1.txt")

    # 2️⃣ Chunk
    chunk_config = ChunkConfig(chunk_size=300, overlap=50)

    chunks = []

    for doc in docs:
        chunks.extend(fixed_chunk_document(doc, chunk_config))

    print(f"Total chunks created: {len(chunks)}")

    # 3️⃣ Embed
    embedder = Embedder(
        EmbeddingConfig(model_name="all-MiniLM-L6-v2")
    )

    embeddings = embedder.embed_documents(chunks)

    # 4️⃣ Store
    dim = embeddings.shape[1]

    store = VectorStore(dim)

    store.add(embeddings)

    print(f"Stored {store.index.ntotal} vectors")

    # 5️⃣ Create Pipeline
    pipeline = RAGPipeline(embedder, store, chunks)

    # 6️⃣ Ask question
    query = input("\nEnter your question: ")

    result_state = pipeline.run(query)

    print("\nFinal Answer:\n", result_state.answer)
    print("Final Decision:", result_state.decision)


if __name__ == "__main__":
    main()