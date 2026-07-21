from dataclasses import dataclass
from datetime import datetime


@dataclass
class PipelineMetrics:
    # Query Information
    query: str = ""
    rewritten_query: str = ""

    # Routing
    route: str = ""
    answer_source: str = ""
    confidence: float = 0.0
    confidence_level: str = ""

    # Retrieval
    retrieved_chunks: int = 0
    retrieval_latency: float = 0.0
    rerank_latency: float = 0.0

    # Web Search
    web_search_used: bool = False
    web_provider: str = ""
    web_latency: float = 0.0
    web_results: int = 0

    # Generation
    generation_latency: float = 0.0

    # Evaluation
    overlap: float = 0.0
    faithfulness: str = ""
    relevance: str = ""

    # Cost & Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    estimated_cost: float = 0.0

    # Total Runtime
    total_latency: float = 0.0

    # Timestamp
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    """
    Stores runtime metrics for a single RAG pipeline execution.

    These metrics are collected throughout the pipeline and can later
    be logged, displayed in Streamlit, or stored in PostgreSQL.
    """




