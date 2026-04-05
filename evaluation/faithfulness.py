from llm.openrouter_client import llm

def check_faithfulness(query, answer, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""You are an evaluator checking if an answer is supported by the given context.

Rules:
- If the core claims in the answer are supported by the context, respond YES.
- If the answer contains claims clearly not present in the context, respond NO.
- Ignore citation markers like [1] or [2].
- Respond ONLY with YES or NO. Nothing else.

Context:
{context}

Answer:
{answer}"""

    result = llm(prompt, temperature=0.0)
    return result.strip().upper()