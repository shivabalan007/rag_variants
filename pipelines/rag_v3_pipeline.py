from orchestration.agent_state import AgentState

from retrieval.query_rewriter import rewrite_query
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

from llm.generator import generate_answer

from evaluation.overlap import context_overlap_score
from evaluation.faithfulness import check_faithfulness
from evaluation.relevance import check_relevance

from core.router import QueryRouter

from web.search import WebSearcher

from monitoring.metrics import PipelineMetrics
from monitoring.logger import PipelineLogger
from monitoring.latency import LatencyTracker
from  monitoring.cost_tracker import CostTracker


class RAGV3Pipeline:

    def __init__(self, embedder, store, chunks):

        self.embedder = embedder
        self.store = store
        self.chunks = chunks

        self.hybrid_retriever = HybridRetriever(chunks)
        self.reranker = CrossEncoderReranker()
        self.router = QueryRouter()
        self.web_searcher = WebSearcher()

        self.metrics_logger = PipelineLogger()
        self.latency_tracker = LatencyTracker()
        self.cost_tracker = CostTracker()

    # Stage 1 : Rewrite Query

    def rewrite(self, state: AgentState):
        self.latency_tracker.start("rewrite")

        state.rewritten_query = rewrite_query(
            state.original_query
        )
        
        self.latency_tracker.stop("rewrite")

        return state

    # Stage 2 : Retrieve

    def retrieve(self, state: AgentState, top_k=10):
        self.latency_tracker.start("retrieve")

        state.retrieval_result = self.hybrid_retriever.hybrid_search(
            state.rewritten_query,
            self.store,
            self.embedder,
            top_k=top_k
        )

        print("\n===== RETRIEVAL =====")
        print("Retrieved Chunks :", state.retrieval_result.retrieved_count)
        for i, item in enumerate(state.retrieval_result.retrieved_chunks, 1):
            print(f"\nChunk {i}")
            print("Vector :", item.vector_score)
            print("BM25   :", item.bm25_score)
        
        self.latency_tracker.stop("retrieve")

        return state

    # Stage 3 : Rerank

    def rerank(self, state: AgentState):
        self.latency_tracker.start("rerank")

        if not state.retrieval_result.has_documents():
            return state

        state.retrieval_result = self.reranker.rerank(
            state.rewritten_query,
            state.retrieval_result,
            top_k=3
        )
        
        print("\n===== RERANK =====")
        print("Top Chunks :", state.retrieval_result.retrieved_count)
        for item in state.retrieval_result.retrieved_chunks:
            print(item.rerank_score)
        
        self.latency_tracker.stop("rerank")

        return state

    # Stage 4 : Route

    def route(self, state: AgentState):
        self.latency_tracker.start("route")

        decision = self.router.route(state.retrieval_result)

        state.route = decision.route
        state.route_reason = decision.reason
        state.route_confidence = decision.confidence
        state.confidence_level = decision.confidence_level

        self.latency_tracker.stop("route")

        return state

    # Stage 5 : Generate

    def generate(self, state: AgentState):
        self.latency_tracker.start("generate")

    # LOW confidence -> Direct Web Search
        if state.confidence_level == "LOW":
        
            print("\n========== WEB FALLBACK ==========")
            print("Reason :", "Low retrieval confidence")
            print("Confidence :", state.route_confidence)

            self.latency_tracker.start("web_search")

            state.web_result = self.web_searcher.search(state.rewritten_query)

            self.latency_tracker.stop("web_search")

            if not state.web_result.has_results():

                state.answer = "I couldn't find relevant information."
                state.answer_source = "Web"

                self.latency_tracker.stop("generate")

                return state

            texts = [
                source.content
                for source in state.web_result.sources
            ]

            state.answer = generate_answer(state.rewritten_query,texts)

            state.answer_source = "Web"

            state.warning = None

            self.latency_tracker.stop("generate")

            return state

    # HIGH / MEDIUM -> Document Answer

        texts = [
            item.chunk.text
            for item in state.retrieval_result.retrieved_chunks
        ]

        state.answer = generate_answer(state.rewritten_query,texts)

        if state.confidence_level == "MEDIUM":

            state.warning = (
                "Answer generated from uploaded documents with medium confidence."
            )

    # Automatic Web fallback

        if state.answer.startswith("I don't know"):

            print("\n========== WEB FALLBACK ==========")
            print("Reason :", "Document answer unavailable")
            print("Confidence :", state.route_confidence)

            self.latency_tracker.start("web_search")

            state.web_result = self.web_searcher.search(state.rewritten_query)

            self.latency_tracker.stop("web_search")

            if state.web_result.has_results():

                texts = [
                    source.content
                    for source in state.web_result.sources
                ]

                state.answer = generate_answer(state.rewritten_query,texts)

                state.answer_source = "Web"

                self.latency_tracker.stop("generate")

                return state

        state.answer_source = "Document"

        self.latency_tracker.stop("generate")

        return state

    # Stage 6 : Evaluate

    def evaluate(self, state: AgentState):
        self.latency_tracker.start("evaluate")

        if state.answer.startswith("I don't know"):

            state.overlap = 0
            state.faithfulness = "NO"
            state.relevance = "NO"

            self.latency_tracker.stop("evaluate")

            return state
        
        # Document Answer

        if state.answer_source == "Document":

            texts = [
                item.chunk.text
                for item in state.retrieval_result.retrieved_chunks
            ]

        # Web Answer

        else:
            texts = [
                source.content
                for source in state.web_result.sources
            ]


        state.overlap = context_overlap_score(
            state.answer,
            texts
        )

        state.faithfulness = check_faithfulness(
            state.original_query,
            state.answer,
            texts
        )

        state.relevance = check_relevance(
            state.original_query,
            state.answer
        )

        self.latency_tracker.stop("evaluate")

        return state

    # Stage 7 : Monitoring
    def monitor(self, state: AgentState):

        """
        Phase 3

        Monitor

        - latency
        - cost
        - tokens
        - route
        - confidence
        """

        return state

    # Stage 8 : Memory

    def memory(self, state: AgentState):

        """
        Phase 2

        Redis

        PostgreSQL

        Mem0
        """

        return state

    # Pipeline Runner

    def run(self, query):

        self.latency_tracker.reset()

        state = AgentState(query)

        state = self.rewrite(state)

        state = self.retrieve(state)

        state = self.rerank(state)

        state = self.route(state)

        state = self.generate(state)

        state = self.evaluate(state)

        state = self.monitor(state)

        state = self.memory(state)

        metrics = PipelineMetrics()

        metrics.query = state.original_query
        metrics.rewritten_query = state.rewritten_query

        metrics.route = state.route
        metrics.answer_source = state.answer_source
        metrics.confidence = state.route_confidence
        metrics.confidence_level = state.confidence_level

        metrics.retrieved_chunks = (
            state.retrieval_result.retrieved_count
        )

        metrics.retrieval_latency = (
            state.retrieval_result.retrieval_latency
        )

        metrics.rerank_latency = (
            state.retrieval_result.rerank_latency
        )

        metrics.generation_latency = (
            self.latency_tracker.get("generate")
        )

        print("\nTracked Latencies:")
        print(self.latency_tracker.as_dict())

        metrics.total_latency = (
            self.latency_tracker.total()
        )

        metrics.overlap = state.overlap
        metrics.faithfulness = state.faithfulness
        metrics.relevance = state.relevance

        if state.web_result:

            metrics.web_search_used = True
            metrics.web_provider = state.web_result.provider
            metrics.web_latency = state.web_result.search_latency
            metrics.web_results = state.web_result.result_count

        cost = self.cost_tracker.estimate(
            prompt=state.rewritten_query,
            response=state.answer
        )

        metrics.prompt_tokens = cost.prompt_tokens
        metrics.completion_tokens = cost.completion_tokens
        metrics.total_tokens = cost.total_tokens
        metrics.estimated_cost = cost.estimated_cost

        state.metrics = metrics
        
        self.metrics_logger.log(metrics)

        return state
    
if __name__ == "__main__":

    import faiss
    import pickle

    from retrieval.reranker import CrossEncoderReranker

    # Load Embedder
    from embeddings.base import EmbeddingConfig
    from embeddings.embedder import Embedder

    embedder = Embedder(
        EmbeddingConfig(
            model_name="all-MiniLM-L6-v2"
        )
    )

    # Load FAISS Index
    index = faiss.read_index("artifacts/faiss.index")

    # Load Chunks
    with open("artifacts/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Create Pipeline
    pipeline = RAGV3Pipeline(
        embedder=embedder,
        store=index,
        chunks=chunks
    )

    while True:

        query = input("\nEnter Query (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        state = pipeline.run(query)

        print("\n==============================")
        print("Original Query :", state.original_query)
        print("Rewritten Query:", state.rewritten_query)

        print("\nRetrieved Chunks :", state.retrieval_result.retrieved_count)
        for i, item in enumerate(state.retrieval_result.retrieved_chunks, 1):

            print(f"\nChunk {i}")
            print("Vector :", item.vector_score)
            print("BM25   :", item.bm25_score)
            print("Rerank :", item.rerank_score)

        print("\nRoute            :", state.route)
        print("Answer Source    :", state.answer_source)
        print("Reason           :", state.route_reason)
        print("Confidence       :", state.route_confidence)
        print("Confidence Level :", state.confidence_level)
        if state.warning:
            print("Warning         :", state.warning)

        if state.web_result:

            print("\n========== WEB SEARCH ==========")
            print("Provider :", state.web_result.provider)
            print("Results  :", state.web_result.result_count)
            print("Latency  :", f"{state.web_result.search_latency:.3f}s")

        print("\nAnswer")
        print(state.answer)

        if state.answer_source == "Web":

            print("\nSources")

            for i, source in enumerate(state.web_result.sources, 1):

                print(f"{i}. {source.title}")
                print(f"   {source.url}")

        print("\nEvaluation")
        print("Overlap      :", state.overlap)
        print("Faithfulness :", state.faithfulness)
        print("Relevance    :", state.relevance)
        if state.metrics:
            print("\nTotal Latency :", f"{state.metrics.total_latency:.3f}s")
            print("Estimated Cost:", f"${state.metrics.estimated_cost:.6f}")
        print("==============================")