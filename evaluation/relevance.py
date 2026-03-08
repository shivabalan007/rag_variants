from llm.openrouter_client import llm

def check_relevance(query, answer):
    prompt = f"""
You are a strict evaluator.

Determine if the answer directly and fully addresses the user's question.

Respond ONLY with YES or NO.

Question:
{query}

Answer:
{answer}
"""

    result = llm(prompt, temperature=0.0)
    return result.strip()