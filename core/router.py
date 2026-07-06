

class QueryRouter:

    def __init__(self):
        # Default thresholds can later be moved to config.py
        self.similarity_threshold = 0.50
        self.rerank_threshold = 0.50

    def route(self, similarity_score, rerank_score, retrieved_chunks):
        """
        Decide whether to use Document RAG or Web Fallback.
        similarity_score : float
            Highest similarity score returned by retrieval.

        rerank_score : float
            Highest CrossEncoder reranker score.

        retrieved_chunks : int
            Number of retrieved chunks.
        """

        # No chunks found
        if retrieved_chunks == 0:
            return {
                "route": "web",
                "reason": "No relevant chunks found.",
                "confidence": 0.0
            }

        # Average confidence
        confidence = (similarity_score + rerank_score) / 2

        # Both scores are low
        if (
            similarity_score < self.similarity_threshold
            and rerank_score < self.rerank_threshold
        ):

            return {
                "route": "web",
                "reason": "Low retrieval confidence.",
                "confidence": confidence
            }

        # Good confidence
        return {
            "route": "document",
            "reason": "Relevant information found in uploaded documents.",
            "confidence": confidence
        }


if __name__ == "__main__":

    router = QueryRouter()

    decision = router.route(
        similarity_score=0.82,
        rerank_score=0.91,
        retrieved_chunks=5
    )

    print(decision)

    decision = router.route(
        similarity_score=0.20,
        rerank_score=0.30,
        retrieved_chunks=2
    )

    print(decision)

    decision = router.route(
        similarity_score=0.0,
        rerank_score=0.0,
        retrieved_chunks=0
    )

    print(decision)

"""
This module decides whether the RAG system should:
1. Answer using the uploaded documents
2. Trigger Web Fallback

Currently the router only makes a decision.
Actual web search will be implemented later.
"""

