from evaluation.overlap import context_overlap_score


def run_overlap_evaluation(query, retrieved_chunks, answer):
    score = context_overlap_score(answer, retrieved_chunks)

    print("QUERY:")
    print(query)
    print("\nRETRIEVED CONTEXT:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"[{i}] {chunk}")

    print("\nANSWER:")
    print(answer)

    print("\nOVERLAP SCORE:", round(score, 3))

    if score < 0.3:
        print("⚠️ Likely hallucination or unsupported answer")
    else:
        print("✅ Answer appears grounded in context")


if __name__ == "__main__":
    # Example retrieved chunks from your OOP notes
    retrieved_chunks = [
        "Encapsulation is the process of bundling data and methods together within a class.",
        "It hides internal details and protects data from external access."
    ]

    # GOOD answer (grounded)
    good_answer = (
        "Encapsulation bundles data and methods within a class "
        "and hides internal details to protect data from external access."
    )

    # BAD answer (hallucinated)
    bad_answer = (
        "Encapsulation improves runtime performance and memory efficiency "
        "by optimizing object creation."
    )

    #query = "What is encapsulation in OOP?"
    query = "What are the four pillars of OOP?"

    print("\n=== GOOD ANSWER EVALUATION ===\n")
    run_overlap_evaluation(query, retrieved_chunks, good_answer)

    print("\n=== BAD ANSWER EVALUATION ===\n")
    run_overlap_evaluation(query, retrieved_chunks, bad_answer)
