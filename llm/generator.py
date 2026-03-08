from llm.openrouter_client import llm
from llm.prompt import build_prompt


def generate_answer(query, retrieved_chunks):

    if not retrieved_chunks:
        return "I don't know based on the provided context."

    # Build the RAG prompt
    prompt = build_prompt(query, retrieved_chunks)

    # Call LLM
    response = llm(prompt, temperature=0.1)

    return response.strip()