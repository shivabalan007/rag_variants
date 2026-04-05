from llm.openrouter_client import llm

def check_relevance(query, answer):
    prompt = f"""You are an evaluator checking if an answer addresses the question.

Rules:
- If the answer directly addresses what the question is asking, respond YES.
- If the answer is off-topic or says "I don't know", respond NO.
- Respond ONLY with YES or NO. Nothing else.

Question:
{query}

Answer:
{answer}"""

    result = llm(prompt, temperature=0.0)
    return result.strip().upper()