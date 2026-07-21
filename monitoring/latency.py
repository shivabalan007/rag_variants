import time


class LatencyTracker:
    def __init__(self):
        self._start_times = {}
        self._latencies = {}

    def start(self, stage: str):
        """
        Start timing a pipeline stage.
        """
        self._start_times[stage] = time.perf_counter()

    def stop(self, stage: str):
        """
        Stop timing a pipeline stage.
        """
        if stage not in self._start_times:
            raise ValueError(
                f"Stage '{stage}' was never started."
            )

        self._latencies[stage] = (
            time.perf_counter()
            - self._start_times[stage]
        )

    def get(self, stage: str) -> float:
        """
        Get latency of a stage.
        """
        return self._latencies.get(stage, 0.0)

    def total(self) -> float:
        """
        Returns total latency of all tracked stages.
        """
        return sum(self._latencies.values())

    def reset(self):
        """
        Clears all recorded timings.
        """
        self._start_times.clear()
        self._latencies.clear()

    def as_dict(self):
        """
        Returns all recorded latencies.
        """
        return dict(self._latencies)


if __name__ == "__main__":

    import time

    tracker = LatencyTracker()

    tracker.start("retrieval")
    time.sleep(0.25)
    tracker.stop("retrieval")

    tracker.start("rerank")
    time.sleep(0.40)
    tracker.stop("rerank")

    tracker.start("generation")
    time.sleep(0.60)
    tracker.stop("generation")

    print("\n========== LATENCY ==========")

    for stage, latency in tracker.as_dict().items():
        print(f"{stage:<15}: {latency:.3f} sec")

    print("-----------------------------")
    print(f"Total Latency : {tracker.total():.3f} sec")

    """
    Utility class for measuring execution latency
    of different RAG pipeline stages.
    """