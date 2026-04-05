def build_prompt(query, chunks):
    context_text = ""
    for i, chunk in enumerate(chunks):
        context_text += f"[{i+1}] {chunk}\n\n"

    prompt = f"""You are a factual assistant. Answer the question using ONLY the provided context.

Rules:
- Answer directly and concisely.
- Use ONLY information from the context. Do not use prior knowledge.
- Cite the source number(s) like [1] or [2] inline.
- If the context does not contain the answer, respond with exactly:
  I don't know based on the provided context.
- Do not hedge, speculate, or add caveats.
- Do not repeat the question.

Context:
{context_text}
Question: {query}
Answer:"""

    return prompt