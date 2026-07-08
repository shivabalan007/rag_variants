from dataclasses import dataclass, field
from typing import List


@dataclass
class WebSource:
    title: str
    url: str
    content: str


@dataclass
class WebResult:
    query: str = ""

    provider: str = ""

    sources: List[WebSource] = field(default_factory=list)

    search_latency: float = 0.0

    success: bool = False

    error: str = ""

    def add_source(
        self,
        title: str,
        url: str,
        content: str,
    ):

        self.sources.append(
            WebSource(
                title=title,
                url=url,
                content=content,
            )
        )

    def has_results(self):

        return len(self.sources) > 0

    @property
    def result_count(self):

        return len(self.sources)

    def get_context(self):

        return "\n\n".join(
            source.content
            for source in self.sources
        )
    
"""
Stores structured web search results for the RAG pipeline. Preserves search
metadata, sources, latency, and extracted content so generation, monitoring,
and persistence can reuse the same web search information.
"""