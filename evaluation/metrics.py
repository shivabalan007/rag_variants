from rouge_score import rouge_scorer


def precision_at_k(retrieved_chunks, expected_answer, k):
    """
    Precision@K — of the top K chunks retrieved, how many are relevant.
    Relevance = chunk text overlaps with expected answer keywords.
    """
    top_k = retrieved_chunks[:k]

    expected_keywords = set(expected_answer.lower().split())

    relevant = 0
    for chunk in top_k:
        chunk_words = set(chunk.lower().split())
        overlap = chunk_words & expected_keywords
        if len(overlap) >= 3:                               # at least 3 keyword matches = relevant
            relevant += 1

    return round(relevant / k, 3) if k > 0 else 0.0


def recall_at_k(retrieved_chunks, expected_answer, k):
    """
    Recall@K — how much of the expected answer is covered by top K chunks.
    """
    top_k = retrieved_chunks[:k]

    expected_keywords = set(expected_answer.lower().split())

    found_keywords = set()
    for chunk in top_k:
        chunk_words = set(chunk.lower().split())
        found_keywords |= chunk_words & expected_keywords

    return round(len(found_keywords) / len(expected_keywords), 3) if expected_keywords else 0.0


def mrr_score(retrieved_chunks, expected_answer):
    """
    MRR — Mean Reciprocal Rank.
    Finds rank of first relevant chunk and returns 1/rank.
    """
    expected_keywords = set(expected_answer.lower().split())

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        chunk_words = set(chunk.lower().split())
        overlap = chunk_words & expected_keywords
        if len(overlap) >= 3:
            return round(1 / rank, 3)

    return 0.0


def rouge_score(generated_answer, expected_answer):
    """
    ROUGE-L — measures longest common subsequence between generated and expected answer.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(expected_answer, generated_answer)
    return round(scores["rougeL"].fmeasure, 3)