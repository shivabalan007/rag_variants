from orchestration.rag_pipeline import RAGPipeline

from ingestion.run_ingestion import ingest
from ingestion.base import Document
from chunking.base import ChunkConfig
from chunking.semantic_window_chunker import semantic_chunk
from chunking.sliding_window_chunker import sliding_window_chunk


from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder

from retrieval.vector_store import VectorStore


def main():

    # 1️⃣ Ingest
    docs = ingest("data/test1.txt")

    # 2️⃣ Chunk
    chunks = []

    for doc in docs:
        semantic_chunks = semantic_chunk(doc.text)

        for sc in semantic_chunks:
            window_chunks = sliding_window_chunk(sc, chunk_size=300, overlap=50)

        for chunk in window_chunks:
            chunks.append(Document(text=chunk, metadata=doc.metadata))

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