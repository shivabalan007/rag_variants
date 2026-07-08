import time

from sentence_transformers import CrossEncoder

from core.retrieval_result import RetrievalResult


class CrossEncoderReranker:
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        top_k: int = 3
    ) -> RetrievalResult:

        start_time = time.perf_counter()

        if not retrieval_result.retrieved_chunks:
            return retrieval_result

        # Create (query, chunk) pairs
        pairs = [
            (query, item.chunk.text)
            for item in retrieval_result.retrieved_chunks
        ]

        # Predict CrossEncoder scores
        scores = self.model.predict(pairs)

        # Attach rerank score to each chunk
        for item, score in zip(retrieval_result.retrieved_chunks, scores):
            item.rerank_score = float(score)

        # Sort by rerank score
        retrieval_result.retrieved_chunks.sort(
            key=lambda item: item.rerank_score,
            reverse=True
        )

        # Keep only Top-K
        retrieval_result.retrieved_chunks = (
            retrieval_result.retrieved_chunks[:top_k]
        )

        retrieval_result.retrieved_count = len(
            retrieval_result.retrieved_chunks
        )

        retrieval_result.rerank_latency = (
            time.perf_counter() - start_time
        )

        return retrieval_result


"""
Uses CrossEncoder (ms-marco-MiniLM-L-6-v2) to rerank retrieved chunks and
updates RetrievalResult with rerank scores while preserving chunk-to-score
relationships for routing, monitoring, and memory.
"""