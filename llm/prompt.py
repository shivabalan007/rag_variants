def build_prompt(query, chunks):
    context_text = ""
    for i, chunk in enumerate(chunks):
        context_text += f"[{i+1}] {chunk}\n\n"

    prompt = f"""
You are a helpful assistant that answers questions based on the provided context.
You are a factual question answering assistant. You should only answer based on the provided context. 

Rules:
- Use ONLY the information provided in the context.
- Do NOT use any external information or prior knowledge.
- If the answer is not contained in the context, say:
    "I don't know based on the provided context."
- Cite sources using [number].
-Cite at least one source in each answer.
- If multiple sources support the same answer, cite all relevant sources.


Context:
{context_text}

Question: {query}

Answer:
"""
    
    return prompt.strip()

