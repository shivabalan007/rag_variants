from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from retrieval.query_rewriter import rewrite_query
from retrieval.reranker import CrossEncoderReranker
from evaluation.faithfulness import check_faithfulness
from evaluation.relevance import check_relevance

import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatOpenAI(
        model="deepseek/deepseek-chat",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=800,
    )


RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a factual assistant. Answer the question using ONLY the provided context.

Rules:
- Answer directly and concisely.
- Use ONLY information from the context. Do not use prior knowledge.
- Cite the source number(s) like [1] or [2] inline.
- If the context does not contain the answer, respond with exactly:
  I don't know based on the provided context.
- Do not hedge, speculate, or add caveats.
- Do not repeat the question.

Context:
{context}

Question: {question}

Answer:""")


def format_chunks(chunks):
    return "\n\n".join(
        f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)
    )


def build_retriever(embedder, store, chunks, reranker, top_k=10):
    def retrieve_and_rerank(query):
        # Embed query
        query_vector = embedder.embed_query(query)

        # Vector search
        scores, indices = store.search(query_vector, top_k=top_k)

        # Get candidate chunks
        candidates = [chunks[i] for i in indices[0]]

        # Rerank
        reranked = reranker.rerank(query, candidates, top_k=3)

        # Return text list
        return [c.text if hasattr(c, "text") else c for c in reranked]

    return RunnableLambda(retrieve_and_rerank)


def build_rag_v3_chain(embedder, store, chunks, reranker):
    llm = get_llm()
    retriever = build_retriever(embedder, store, chunks, reranker)

    chain = (
        {
            "context": RunnableLambda(rewrite_query) | retriever | RunnableLambda(format_chunks),
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def run_rag_v3(query, embedder, store, chunks, reranker):

    # Build chain
    chain = build_rag_v3_chain(embedder, store, chunks, reranker)

    # Run chain
    answer = chain.invoke(query)

    # Evaluate
    query_vector = embedder.embed_query(query)
    scores, indices = store.search(query_vector, top_k=5)
    candidates = [chunks[i] for i in indices[0]]
    reranked = reranker.rerank(query, candidates, top_k=3)
    retrieved_texts = [c.text if hasattr(c, "text") else c for c in reranked]

    faithfulness = check_faithfulness(query, answer, retrieved_texts)
    relevance = check_relevance(query, answer)

    return answer, faithfulness, relevance