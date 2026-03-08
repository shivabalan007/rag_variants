from langgraph.graph import StateGraph, END

from orchestration.decision_engine import DecisionEngine
from retrieval.query_rewriter import rewrite_query
from retrieval.reranker import CrossEncoderReranker
from llm.generator import generate_answer

from evaluation.overlap import context_overlap_score
from evaluation.faithfulness import check_faithfulness
from evaluation.relevance import check_relevance


def build_langgraph_agent(embedder, store, chunks, similarity_threshold=0.5):

    reranker = CrossEncoderReranker()
    decision_engine = DecisionEngine()

    # -------------------------
    # Rewrite Node
    # -------------------------
    def rewrite(state):

        query = state.get("original_query")

        if not isinstance(query, str):
            query = ""

        rewritten = rewrite_query(query)

        return {
            "original_query": query,
            "rewritten_query": rewritten
        }

    # -------------------------
    # Retrieve Node
    # -------------------------
    def retrieve(state):

        query_vector = embedder.embed_query(state.get("rewritten_query"))

        top_k = 3 + state.get("attempt", 0) * 2

        scores, indices = store.search(
            query_vector,
            top_k
        )

        filtered = []

        for score, idx in zip(scores[0], indices[0]):
            if score >= similarity_threshold:
                filtered.append(chunks[idx])

        return {
            "similarity_scores": scores[0],
            "candidate_chunks": filtered
        }

    # -------------------------
    # Rerank Node
    # -------------------------
    def rerank(state):

        candidate_chunks = state.get("candidate_chunks", [])

        if not candidate_chunks:
            return {"reranked_chunks": []}

        query = state.get("rewritten_query") or state.get("original_query")
        
        if not isinstance(query, str):
            return {"reranked_chunks": candidate_chunks[:3]}

        valid_chunks = [c for c in candidate_chunks if hasattr(c, "text")]  
        
        if not valid_chunks:
            return {"reranked_chunks": []}

        reranked = reranker.rerank(
            query,
            candidate_chunks,
            top_k=3
        )

        return {
            "reranked_chunks": reranked
        }

    # -------------------------
    # Generate Node
    # -------------------------
    def generate(state):

        reranked_chunks = state.get("reranked_chunks", [])

        if not reranked_chunks:
            return {"answer": "I don't know based on similarity threshold"}

        texts = [chunk.text for chunk in reranked_chunks]

        answer = generate_answer(
            state.get("rewritten_query"),
            texts
        )

        return {
            "answer": answer
        }

    # -------------------------
    # Evaluate Node
    # -------------------------
    def evaluate(state):

        answer = state.get("answer", "")

        if answer.startswith("I don't know"):
            return {
                "overlap": 0,
                "faithfulness": "No",
                "relevance": "No"
            }

        texts = [chunk.text for chunk in state.get("reranked_chunks", [])]

        overlap = context_overlap_score(
            answer,
            texts
        )

        faithfulness = check_faithfulness(
            state.get("original_query"),
            answer,
            texts
        )

        relevance = check_relevance(
            state.get("original_query"),
            answer
        )

        return {
            "overlap": overlap,
            "faithfulness": faithfulness,
            "relevance": relevance
        }

    # -------------------------
    # Decision Node
    # -------------------------
    def decide(state):

        decision = decision_engine.decide(
            state.get("overlap"),
            state.get("faithfulness")
        )

        if decision == "ACCEPT":
            return {"decision": "ACCEPT"}

        if state.get("attempt", 0) >= 1:
            return {
                "decision": "STOP",
                "answer": "I don't know based on the provided context."
            }

        return {
            "decision": "RETRY",
            "attempt": state.get("attempt", 0) + 1
        }

    # -------------------------
    # Build Graph
    # -------------------------

    graph = StateGraph(dict)

    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)
    graph.add_node("decide", decide)

    graph.set_entry_point("rewrite")

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_edge("evaluate", "decide")

    graph.add_conditional_edges(
        "decide",
        lambda state: state["decision"],
        {
            "ACCEPT": END,
            "RETRY": "retrieve",
            "STOP": END
        }
    )

    return graph.compile()