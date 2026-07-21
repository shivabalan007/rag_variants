from pipelines.rag_v3_pipeline import RAGV3Pipeline

_pipeline = None


def run_rag_v3(query, embedder, store, chunks, reranker=None):
    """
    Execute the Advanced Agentic RAG V3 pipeline.

    Parameters
    ----------
    query : str
        User question.

    embedder : Embedder
        SentenceTransformer embedder.

    store : VectorStore
        FAISS vector store wrapper.

    chunks : list
        Indexed document chunks.

    reranker : CrossEncoderReranker, optional
        Not required here because RAGV3Pipeline creates its own reranker.
        Kept only for compatibility with the existing Streamlit app.

    Returns
    -------
    AgentState
        Final pipeline state containing:
        - answer
        - rewritten_query
        - retrieval_result
        - routing decision
        - confidence
        - answer source
        - web search result
        - evaluation
        - monitoring metrics
    """

    global _pipeline

    # Create pipeline only once
    if _pipeline is None:
        _pipeline = RAGV3Pipeline(
            embedder=embedder,
            store=store.index,
            chunks=chunks
        )

    # Execute pipeline
    state = _pipeline.run(query)

    return state

"""
Wrapper for the Advanced Agentic RAG V3 Pipeline.
Used by Streamlit to execute the pipeline and return the final AgentState.
"""