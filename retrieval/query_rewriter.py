from llm.openrouter_client import llm

def rewrite_query(query):
    prompt = f"""
Rewrite the following user query into a clear,
well-formed question optimized for semantic retrieval.

Do NOT answer the question.
Only return the rewritten query.

User Query:
{query}
"""

    rewritten = llm(prompt, temperature=0.0)
    return rewritten.strip()