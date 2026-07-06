from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class RetrievalResult:
    """
    Stores everything produced during the retrieval stage.

    Every future module (Router, Monitoring, Memory,
    Web Fallback, Evaluation) should read from this object
    instead of recomputing retrieval information.
    """

    # Retrieved Documents
    chunks: List[Any] = field(default_factory=list)


    # Vector Search
    vector_scores: List[float] = field(default_factory=list)

    # BM25 Search
    bm25_scores: List[float] = field(default_factory=list)

    # CrossEncoder
    rerank_scores: List[float] = field(default_factory=list)

    # Metadata
    retrieved_count: int = 0

    retrieval_latency: float = 0.0
    rerank_latency: float = 0.0

    source: str = "document"

    # Future Fields
    web_results: Optional[List[Any]] = None

    confidence: float = 0.0

    route: Optional[str] = None

    reason: Optional[str] = None

    def top_vector_score(self):
        if not self.vector_scores:
            return 0.0
        return max(self.vector_scores)

    def top_bm25_score(self):
        if not self.bm25_scores:
            return 0.0
        return max(self.bm25_scores)

    def top_rerank_score(self):
        if not self.rerank_scores:
            return 0.0
        return max(self.rerank_scores)

    def has_documents(self):
        return len(self.chunks) > 0
    
"""
Stores all retrieval outputs (chunks, vector scores, BM25 scores, rerank scores, latency, metadata) in a single object. Prevents information loss between retrieval stages and supports routing, monitoring, memory, and future production features.
"""