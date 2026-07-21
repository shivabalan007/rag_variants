import time

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list,
        top_k: int = 3
    ) -> list:
        """
        Rerank a list of retrieved chunks using CrossEncoder.
        Compatible with RAG V1 and RAG V2.
        """

        start_time = time.perf_counter()

        if not chunks:
            return []

        # Create (query, chunk) pairs
        pairs = [
            (
                query,
                chunk.text if hasattr(chunk, "text") else str(chunk)
            )
            for chunk in chunks
        ]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Sort chunks by score
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Return only the chunks
        reranked_chunks = [
            chunk
            for chunk, _ in ranked[:top_k]
        ]

        print(
            f"Rerank Latency: {time.perf_counter() - start_time:.3f}s"
        )

        return reranked_chunks