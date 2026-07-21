from monitoring.metrics import PipelineMetrics


class PipelineLogger:

    def log(self, metrics: PipelineMetrics):

        print("\n========== PIPELINE METRICS ==========")

        # Query
        print("Query               :", metrics.query)
        print("Rewritten Query     :", metrics.rewritten_query)

        # Routing
        print("\n----- Routing -----")

        print("Route               :", metrics.route)
        print("Answer Source       :", metrics.answer_source)
        print("Confidence          :", round(metrics.confidence, 3))
        print("Confidence Level    :", metrics.confidence_level)

        # Retrieval
        print("\n----- Retrieval -----")

        print("Retrieved Chunks    :", metrics.retrieved_chunks)
        print(
            "Retrieval Latency   :",
            f"{metrics.retrieval_latency:.3f} sec"
        )
        print(
            "Rerank Latency      :",
            f"{metrics.rerank_latency:.3f} sec"
        )

        # Web Search
        print("\n----- Web Search -----")

        print("Web Search Used     :", metrics.web_search_used)

        if metrics.web_search_used:

            print("Provider            :", metrics.web_provider)
            print("Results             :", metrics.web_results)
            print(
                "Search Latency      :",
                f"{metrics.web_latency:.3f} sec"
            )

        # Generation
        print("\n----- Generation -----")

        print(
            "Generation Latency  :",
            f"{metrics.generation_latency:.3f} sec"
        )

        # Evaluation
        print("\n----- Evaluation -----")

        print("Faithfulness        :", metrics.faithfulness)
        print("Relevance           :", metrics.relevance)
        print("Overlap             :", round(metrics.overlap, 3))

        # Token Usage
        print("\n----- Token Usage -----")

        print("Prompt Tokens       :", metrics.prompt_tokens)
        print("Completion Tokens   :", metrics.completion_tokens)
        print("Total Tokens        :", metrics.total_tokens)

        # Cost
        print("\n----- Cost -----")

        print(
            "Estimated Cost      :",
            f"${metrics.estimated_cost:.6f}"
        )

        # Total Runtime
        print("\n----- Runtime -----")

        print(
            "Total Latency       :",
            f"{metrics.total_latency:.3f} sec"
        )

        # Timestamp
        print("\nTimestamp           :", metrics.timestamp)

        print("======================================")

    """
    Logs pipeline execution metrics.

    This logger is responsible only for displaying
    or recording pipeline metrics. It does not
    calculate any metrics.
    """