from llm.groq_client import llm


def check_relevance(query, answer):

    prompt = f"""
You are an expert RAG evaluation assistant.

Your task is to determine whether the generated answer actually answers the user's question.

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

Evaluate ONLY whether the answer addresses the user's question.

Return:

YES

if the answer:

- directly answers the question
- stays on the requested topic
- provides useful information related to the question
- partially answers the question while remaining relevant

Return:

NO

if the answer:

- is unrelated to the question
- changes the subject
- avoids answering
- is generic without addressing the query
- only repeats the question
- says:
  "I don't know based on the provided context."
- refuses without answering
- discusses a different topic

Ignore:

- grammar
- writing style
- formatting
- citations such as [1] or [2]
- answer length

Examples

Question:
What is Python?

Answer:
Python is an interpreted programming language.

YES

Question:
What is Python?

Answer:
Python was created by Guido van Rossum and is widely used.

YES

Question:
What is Python?

Answer:
Java is an object-oriented language.

NO

Question:
What is Python?

Answer:
I don't know based on the provided context.

NO

Question:
Explain OOP.

Answer:
Object-Oriented Programming organizes software around objects.

YES

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

    # LLM/provider failure.
    # Evaluation could not be performed.
    if result is None:
        return "N/A"

    result = result.strip().upper()

    # Protect against unexpected model output.
    if result not in {"YES", "NO"}:
        return "N/A"

    return result


"""
Asks LLM whether the answer directly addresses the user's question.
Returns YES, NO, or N/A when the evaluation LLM is unavailable.
"""