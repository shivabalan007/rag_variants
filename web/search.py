import os
import time

from dotenv import load_dotenv
from tavily import TavilyClient

from web.web_result import WebResult

load_dotenv()


class WebSearcher:
    """
    Searches the web using Tavily and returns
    a structured WebResult object.
    """

    def __init__(self):

        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY not found in .env"
            )

        self.client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> WebResult:

        start_time = time.perf_counter()

        result = WebResult(
            query=query,
            provider="Tavily",
        )

        try:

            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
            )

            for item in response.get("results", []):

                result.add_source(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                )

            result.success = True

        except Exception as e:

            result.success = False
            result.error = str(e)

        result.search_latency = (
            time.perf_counter() - start_time
        )

        return result


if __name__ == "__main__":

    searcher = WebSearcher()

    while True:

        query = input("\nEnter Query (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        result = searcher.search(query)

        print("\n========== WEB SEARCH ==========")

        print("Provider :", result.provider)
        print("Success  :", result.success)
        print("Latency  :", f"{result.search_latency:.3f}s")
        print("Results  :", result.result_count)

        if result.error:
            print("Error    :", result.error)

        for i, source in enumerate(result.sources, start=1):

            print(f"\nResult {i}")

            print("Title :", source.title)
            print("URL   :", source.url)
            print("Content:")
            print(source.content[:300])
            print("-" * 80)