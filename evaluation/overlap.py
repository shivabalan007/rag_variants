import re 

def tokenize(text):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

def context_overlap_score(answer, context_chunks):
    answer_tokens = set(tokenize(answer))

    context_text = " ".join(context_chunks)
    context_tokens = set(tokenize(context_text))

    if not answer_tokens:
        return 0.0
    
    overlap = answer_tokens.intersection(context_tokens)
    score = len(overlap) / len(answer_tokens)

    return score 

"""
Computes keyword overlap score between generated answer and retrieved chunks. Non-LLM signal — fast, deterministic, unbiased by the same model that generated the answer.
"""