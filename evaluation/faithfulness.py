from llm.openrouter_client import llm

def check_faithfulness(query, answer, context_chunks):
    context_chunks = "\n\n".join(context_chunks)

    prompt = f"""
You are a strict evaluator.
Determine whether the answer is fully supported by the provided context.        
Rules:
- If every claim in the answer is supported by the context, respond with: YES
- If any part of the answer is not supported by the context, respond with: NO
- Do not explain. Respond only with YES or NO.

Context:
{context_chunks}

Answer:
{answer}
"""
    result = llm(prompt, temperature=0.0)

    return result.strip()
