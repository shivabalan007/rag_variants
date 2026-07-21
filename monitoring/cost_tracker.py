from dataclasses import dataclass


@dataclass
class CostEstimate:
    """
    Stores token usage and estimated API cost
    for a single LLM request.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


class CostTracker:
    """
    Estimates token usage and API cost.

    Current Version:
        - Estimates tokens from text length.

    Future Version:
        - Use actual usage returned by
          OpenRouter/OpenAI APIs.
    """

    # Estimated pricing per 1M tokens (USD)
    # Update these values based on your model.
    INPUT_COST_PER_MILLION = 0.15
    OUTPUT_COST_PER_MILLION = 0.60

    def estimate(
        self,
        prompt: str,
        response: str
    ) -> CostEstimate:
        """
        Estimate token usage and API cost.
        """

        prompt_tokens = self._estimate_tokens(prompt)
        completion_tokens = self._estimate_tokens(response)

        total_tokens = (
            prompt_tokens +
            completion_tokens
        )

        input_cost = (
            prompt_tokens / 1_000_000
        ) * self.INPUT_COST_PER_MILLION

        output_cost = (
            completion_tokens / 1_000_000
        ) * self.OUTPUT_COST_PER_MILLION

        total_cost = input_cost + output_cost

        return CostEstimate(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=total_cost
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Rough token estimation.

        Rule of thumb:
        ~1 token ≈ 4 characters.
        """

        if not text:
            return 0

        return max(
            1,
            len(text) // 4
        )


if __name__ == "__main__":

    tracker = CostTracker()

    prompt = (
        "What is Machine Learning?"
    )

    response = (
        "Machine learning is a subset of artificial intelligence "
        "that enables systems to learn from data."
    )

    metrics = tracker.estimate(
        prompt,
        response
    )

    print("\n========== COST TRACKER ==========")

    print("Prompt Tokens     :", metrics.prompt_tokens)
    print("Completion Tokens :", metrics.completion_tokens)
    print("Total Tokens      :", metrics.total_tokens)
    print("Estimated Cost    : $", round(metrics.estimated_cost, 8))