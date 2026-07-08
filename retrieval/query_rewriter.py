from llm.openrouter_client import llm

def rewrite_query(query):
    """
    Takes raw user query and rewrites it to be more specific and retrieval-friendly. Expands abbreviations, adds context — improves chunk matching quality.
    """
    prompt = f"""
You are a query rewriting assistant.

Rewrite the user's query into a grammatically correct, natural English question optimized for semantic retrieval.

Rules:
- Preserve the original meaning.
- Expand abbreviations when appropriate (e.g., AI → Artificial Intelligence).
- Do NOT answer the question.
- Do NOT add extra information.
- Return only the rewritten query as one well-formed sentence with proper spacing and punctuation.


User Query:
{query}
"""
    
    rewritten = llm(prompt, temperature=0.0)
    rewritten = " ".join(rewritten.split())
    return rewritten.strip()

if __name__ == "__main__":
    print(rewrite_query("what is ai"))