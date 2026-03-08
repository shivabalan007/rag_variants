from orchestration.agent_state import AgentState
from retrieval.query_rewriter import rewrite_query
from retrieval.reranker import CrossEncoderReranker
from llm.generator import generate_answer

from evaluation.overlap import context_overlap_score
from evaluation.faithfulness import check_faithfulness
from evaluation.relevance import check_relevance

from orchestration.decision_engine import DecisionEngine


class RAGPipeline:

    def __init__(self, embedder, store, chunks,
                 similarity_threshold=0.5,
                 max_attempts=2):

        self.embedder = embedder
        self.store = store
        self.chunks = chunks

        self.threshold = similarity_threshold
        self.max_attempts = max_attempts

        self.reranker = CrossEncoderReranker()
        self.decision_engine = DecisionEngine()

    # Stage 1: Rewrite
    def rewrite(self, state: AgentState):
        state.rewritten_query = rewrite_query(state.original_query)
        return state

    # Stage 2: Retrieve + threshold filtering
    def retrieve(self, state: AgentState, top_k):

        query_vector = self.embedder.embed_query(state.rewritten_query)

        scores, indices = self.store.search(query_vector, top_k)

        state.similarity_scores = scores[0]

        filtered = []

        for score, idx in zip(scores[0], indices[0]):
            if score >= self.threshold:
                filtered.append(self.chunks[idx])

        state.candidate_chunks = filtered

        return state

    # Stage 3: Rerank
    def rerank(self, state: AgentState):

        if not state.candidate_chunks:
            state.reranked_chunks = []
            return state

        reranked = self.reranker.rerank(
            state.rewritten_query,
            state.candidate_chunks,
            top_k=3
        )

        state.reranked_chunks = reranked

        return state

    # Stage 4: Generate
    def generate(self, state: AgentState):

        if not state.reranked_chunks:
            state.answer = "I don't know based on similarity threshold"
            return state

        texts = [chunk.text for chunk in state.reranked_chunks]

        state.answer = generate_answer(state.rewritten_query, texts)

        return state

    # Stage 5: Evaluate
    def evaluate(self, state: AgentState):

        if state.answer.startswith("I don't know"):
            state.overlap = 0
            state.faithfulness = "NO"
            state.relevance = "NO"
            return state

        texts = [chunk.text for chunk in state.reranked_chunks]

        state.overlap = context_overlap_score(state.answer, texts)

        state.faithfulness = check_faithfulness(
            state.original_query,
            state.answer,
            texts
        )

        state.relevance = check_relevance(
            state.original_query,
            state.answer
        )

        return state

    # Stage 6: Decide
    def run(self, query: str):

        state = AgentState(query)

        for attempt in range(self.max_attempts):

            state.attempt = attempt + 1

            print(f"\n=== ATTEMPT {state.attempt} ===")

            state = self.rewrite(state)

            state = self.retrieve(state, top_k=3 + attempt * 2)

            state = self.rerank(state)

            state = self.generate(state)

            state = self.evaluate(state)

            decision = self.decision_engine.decide(
                state.overlap,
                state.faithfulness
            )

            state.decision = decision

            print("\nANSWER:\n", state.answer)
            print("OVERLAP:", state.overlap)
            print("FAITHFULNESS:", state.faithfulness)
            print("RELEVANCE:", state.relevance)
            print("DECISION:", state.decision)

            if decision == "ACCEPT":
                return state

        state.answer = "I don't know based on the provided context."
        state.decision = "REFUSE"

        return state