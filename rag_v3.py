from pipelines.rag_v3_pipeline import RAGV3Pipeline

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.manager import MemoryManager

from retrieval.pg_vector_store import PGVectorStore
from retrieval.pg_hybrid_search import PGHybridRetriever



_pipeline = {}


def run_rag_v3(query, embedder, session_id, reranker, user_id="default_user"):
    global _pipeline

    print("="*60)
    print("Session ID :", session_id)
    print("Pipelines :", list(_pipeline.keys()))
    print("="*60)

    if session_id not in _pipeline:
        print(">>> Creating NEW Pipeline")


        short_memory = ShortTermMemory(session_id=session_id, max_messages=20)
        long_memory = LongTermMemory()

        memory = MemoryManager(
            short_memory=short_memory,
            long_memory=long_memory,
            session_id=session_id,
            user_id=user_id
        )

        print(">>> Calling load_session()")
        memory.load_session()

        pg_store = PGVectorStore()

        pg_hybrid_retriever = PGHybridRetriever(vector_store=pg_store,auto_refresh=False)

        _pipeline[session_id] = RAGV3Pipeline(
            embedder=embedder,
            store=pg_store,
            chunks=None,
            memory=memory,
            hybrid_retriever=pg_hybrid_retriever,
            reranker=reranker,
        )
    else:
        print(">>> Reusing Existing Pipeline")


    # Execute pipeline
    state = _pipeline[session_id].run(query)

    return state

"""
Wrapper for the Advanced Agentic RAG V3 Pipeline.
Used by Streamlit to execute the pipeline and return the final AgentState.
"""