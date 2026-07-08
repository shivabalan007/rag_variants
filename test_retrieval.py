import faiss
import pickle

from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder

from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import CrossEncoderReranker


def main():

    # ----------------------------
    # Load Embedder
    # ----------------------------

    embedder = Embedder(
        EmbeddingConfig(
            model_name="all-MiniLM-L6-v2"
        )
    )

    # ----------------------------
    # Load FAISS Index
    # ----------------------------

    index = faiss.read_index("artifacts/faiss.index")

    store = VectorStore(index.d)
    store.index = index

    # ----------------------------
    # Load Chunks
    # ----------------------------

    with open("artifacts/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # ----------------------------
    # Initialize Components
    # ----------------------------

    retriever = HybridRetriever(chunks)

    reranker = CrossEncoderReranker()

    # ----------------------------
    # Test Loop
    # ----------------------------

    while True:

        query = input("\nEnter Query (exit to quit): ")

        if query.lower() == "exit":
            break

        retrieval_result = retriever.hybrid_search(
            query=query,
            vector_store=store,
            embedder=embedder,
            top_k=10
        )

        retrieval_result = reranker.rerank(
            query=query,
            retrieval_result=retrieval_result,
            top_k=3
        )

        print("\n========== Retrieval Result ==========")

        print("Retrieved Count :", retrieval_result.retrieved_count)

        print("Retrieval Latency :", f"{retrieval_result.retrieval_latency:.4f} sec")

        print("Rerank Latency :", f"{retrieval_result.rerank_latency:.4f} sec")

        print("\nTop Chunks\n")

        for i, item in enumerate(retrieval_result.retrieved_chunks, start=1):

            print(f"Chunk {i}")

            print(
                f"Vector Score : {item.vector_score:.4f}"
            )

            print(
                f"BM25 Score   : {item.bm25_score:.4f}"
            )

            print(
                f"Rerank Score : {item.rerank_score:.4f}"
            )

            print("\nPreview:")

            print(item.chunk.text[:300])

            print("-" * 80)


if __name__ == "__main__":
    main()