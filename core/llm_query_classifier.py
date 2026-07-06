import os
from enum import Enum
from dataclasses import dataclass

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class QueryType(Enum):
    DOCUMENT = "document"
    GENERAL = "general"
    HYBRID = "hybrid"


@dataclass
class QueryClassification:
    query_type: QueryType
    reason: str


CLASSIFIER_PROMPT = ChatPromptTemplate.from_template("""
You are an intelligent query classifier for a Retrieval-Augmented Generation (RAG) system.

Classify the user's query into EXACTLY ONE category.

DOCUMENT
- The answer should come from the uploaded documents.
- Examples:
  - Summarize this PDF.
  - Explain Chapter 3.
  - According to the uploaded report...
  - What does this document say about AI?

GENERAL
- The answer is general world knowledge.
- Examples:
  - Who is Elon Musk?
  - What is Python?
  - What is the capital of Japan?
  - Explain quantum computing.

HYBRID
- The query could be answered from either uploaded documents or external knowledge.
- Examples:
  - Explain Retrieval-Augmented Generation.
  - What is Artificial Intelligence?
  - Explain Machine Learning.
  - What is Docker?

Return ONLY in this format.

CATEGORY: DOCUMENT
REASON: ...

OR

CATEGORY: GENERAL
REASON: ...

OR

CATEGORY: HYBRID
REASON: ...

User Query:
{query}
""")


class QueryClassifier:

    def __init__(self):

        self.llm = ChatOpenAI(
            model="deepseek/deepseek-chat",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=100,
        )

        self.chain = (
            CLASSIFIER_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def classify(self, query: str) -> QueryClassification:

        response = self.chain.invoke({"query": query}).strip()

        category = "GENERAL"
        reason = response

        for line in response.splitlines():

            if line.upper().startswith("CATEGORY:"):
                category = line.split(":", 1)[1].strip().upper()

            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        if category == "DOCUMENT":
            query_type = QueryType.DOCUMENT

        elif category == "HYBRID":
            query_type = QueryType.HYBRID

        else:
            query_type = QueryType.GENERAL

        return QueryClassification(
            query_type=query_type,
            reason=reason
        )


if __name__ == "__main__":

    classifier = QueryClassifier()

    while True:

        query = input("\nEnter Query (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        result = classifier.classify(query)

        print("\nQuery Type :", result.query_type.value)
        print("Reason     :", result.reason)

"""
Uses an LLM to classify user queries as DOCUMENT, GENERAL, or HYBRID before retrieval. Enables intelligent routing so the RAG pipeline can decide whether to answer from uploaded documents, external web search, or both.
"""