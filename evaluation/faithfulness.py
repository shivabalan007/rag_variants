from llm.openrouter_client import llm


def check_faithfulness(query, answer, context_chunks):

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an expert RAG evaluation assistant.

Your task is to determine whether the answer is completely supported by the retrieved document context.

==================================================
Retrieved Context
==================================================

{context}

==================================================
User Question
==================================================

{query}

==================================================
Generated Answer
==================================================

{answer}

==================================================
Evaluation Rules
==================================================

Evaluate ONLY factual support.

Return:

YES

if EVERY factual claim in the answer is supported by the retrieved context.

Return:

NO

if ANY factual claim is:

- missing from the context
- contradicted by the context
- invented
- inferred without evidence
- based on outside knowledge

Ignore:

- writing style
- grammar
- wording differences
- sentence structure
- citation markers such as [1] or [2]

Special Cases

1. If the answer correctly says:

"I don't know based on the provided context."

return:

YES

2. If the answer contains both supported and unsupported facts,

return:

NO

3. If the answer introduces even one new factual claim
that cannot be verified from the context,

return:

NO

==================================================
Output Rules
==================================================

Return ONLY ONE WORD.

Allowed outputs:

YES

NO

Do not explain.

Do not justify.

Do not output punctuation.

"""

    result = llm(
        prompt,
        temperature=0.0
    )

    return result.strip().upper()

"""
Asks LLM whether every core claim in the answer is supported by retrieved context. Returns YES or NO — catches hallucinations and unsupported statements.
"""