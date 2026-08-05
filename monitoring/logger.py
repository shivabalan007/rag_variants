from monitoring.metrics import PipelineMetrics


class PipelineLogger:

    def log(self, metrics: PipelineMetrics):

        print("\n========== PIPELINE METRICS ==========")

        print("Query               :", metrics.query)
        print("Rewritten Query     :", metrics.rewritten_query)

        print("\n----- Routing -----")

        print("Route               :", metrics.route)
        print("Answer Source       :", metrics.answer_source)

        if metrics.confidence is not None:
            print("Confidence          :", round(metrics.confidence, 3))
        else:
            print("Confidence          : N/A")

        print(
            "Confidence Level    :",
            metrics.confidence_level
            if metrics.confidence_level is not None
            else "N/A"
        )

        print("\n----- Retrieval -----")

        if metrics.retrieved_chunks is not None:

            print("Retrieved Chunks    :", metrics.retrieved_chunks)

            print(
                "Retrieval Latency   :",
                f"{metrics.retrieval_latency:.3f} sec"
                if metrics.retrieval_latency is not None
                else "N/A"
            )

            print(
                "Rerank Latency      :",
                f"{metrics.rerank_latency:.3f} sec"
                if metrics.rerank_latency is not None
                else "N/A"
            )

        else:

            print("Retrieved Chunks    : N/A")
            print("Retrieval Latency   : N/A")
            print("Rerank Latency      : N/A")

        print("\n----- Web Search -----")

        print("Web Search Used     :", metrics.web_search_used)

        if metrics.web_search_used:

            print("Provider            :", metrics.web_provider)
            print("Results             :", metrics.web_results)

            print(
                "Search Latency      :",
                f"{metrics.web_latency:.3f} sec"
                if metrics.web_latency is not None
                else "N/A"
            )

        print("\n----- Generation -----")

        print(
            "Generation Latency  :",
            f"{metrics.generation_latency:.3f} sec"
            if metrics.generation_latency is not None
            else "N/A"
        )

        print("\n----- Evaluation -----")

        print(
            "Faithfulness        :",
            metrics.faithfulness
            if metrics.faithfulness is not None
            else "N/A"
        )

        print(
            "Relevance           :",
            metrics.relevance
            if metrics.relevance is not None
            else "N/A"
        )

        print(
            "Overlap             :",
            metrics.overlap
            if metrics.overlap is not None
            else "N/A"
        )

        print("\n----- Token Usage -----")

        print("Prompt Tokens       :", metrics.prompt_tokens)
        print("Completion Tokens   :", metrics.completion_tokens)
        print("Total Tokens        :", metrics.total_tokens)

        print("\n----- Cost -----")

        print(
            "Estimated Cost      :",
            f"${metrics.estimated_cost:.6f}"
        )

        print("\n----- Runtime -----")

        print(
            "Total Latency       :",
            f"{metrics.total_latency:.3f} sec"
            if metrics.total_latency is not None
            else "N/A"
        )

        print("\nTimestamp           :", metrics.timestamp)

        print("======================================")

"""
    Logs pipeline execution metrics.

    This logger is responsible only for displaying
    or recording pipeline metrics. It does not
    calculate any metrics.
"""