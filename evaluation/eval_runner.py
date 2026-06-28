import json
import pickle
import faiss
import sys
import time

from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.reranker import CrossEncoderReranker

from rag_v1 import run_rag_v1
from rag_v2 import run_rag_v2
from rag_v3 import run_rag_v3

from evaluation.metrics import precision_at_k, recall_at_k, mrr_score, rouge_score


def load_system():
    index = faiss.read_index("artifacts/faiss.index")

    with open("artifacts/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedder = Embedder(EmbeddingConfig(model_name="all-MiniLM-L6-v2"))

    store = VectorStore(index.d)
    store.index = index

    reranker = CrossEncoderReranker()

    return embedder, store, chunks, reranker


def load_eval_dataset():
    with open("data/eval_dataset.json", "r") as f:
        return json.load(f)


def get_retrieved_chunks(query, embedder, store, chunks, reranker, top_k=5):
    from retrieval.query_rewriter import rewrite_query

    rewritten = rewrite_query(query)
    query_vector = embedder.embed_query(rewritten)
    scores, indices = store.search(query_vector, top_k=top_k)
    retrieved = [chunks[i] for i in indices[0]]
    reranked = reranker.rerank(rewritten, retrieved, top_k=top_k)
    return [chunk.text if hasattr(chunk, "text") else chunk for chunk in reranked]


def print_table(results, pipeline_name):
    print(f"\n{'='*92}")
    print(f"  {pipeline_name} EVALUATION RESULTS")
    print(f"{'='*92}")
    print(f"{'Question':<38} {'P@3':>6} {'R@3':>6} {'MRR':>6} {'ROUGE':>6} {'Faith':>6} {'Rel':>6} {'ms':>7}")
    print(f"{'-'*92}")

    total_p, total_r, total_mrr, total_rouge, total_ms = 0, 0, 0, 0, 0

    for r in results:
        q = r["question"][:36] + ".." if len(r["question"]) > 36 else r["question"]
        print(
            f"{q:<38} "
            f"{r['precision_at_3']:>6} "
            f"{r['recall_at_3']:>6} "
            f"{r['mrr']:>6} "
            f"{r['rouge']:>6} "
            f"{r['faithfulness']:>6} "
            f"{r['relevance']:>6} "
            f"{r['latency_ms']:>7}"
        )
        total_p     += r["precision_at_3"]
        total_r     += r["recall_at_3"]
        total_mrr   += r["mrr"]
        total_rouge += r["rouge"]
        total_ms    += r["latency_ms"]

    n = len(results)
    print(f"{'-'*92}")
    print(
        f"{'AVERAGE':<38} "
        f"{round(total_p/n,3):>6} "
        f"{round(total_r/n,3):>6} "
        f"{round(total_mrr/n,3):>6} "
        f"{round(total_rouge/n,3):>6} "
        f"{'':>6} "
        f"{'':>6} "
        f"{round(total_ms/n):>7}"
    )
    print(f"{'='*92}\n")


def evaluate_pipeline(pipeline_name, run_fn, embedder, store, chunks, reranker, dataset):
    results = []

    for i, item in enumerate(dataset):
        question = item["question"]
        expected = item["expected_answer"]

        print(f"  [{i+1}/{len(dataset)}] {question[:60]}")

        retrieved_texts = get_retrieved_chunks(
            question, embedder, store, chunks, reranker, top_k=5
        )

        # Measure latency
        start = time.time()

        if pipeline_name == "RAG v1":
            answer, faith, rel = run_fn(question, embedder, store, chunks, reranker)
        elif pipeline_name == "RAG v2":
            answer, faith, rel = run_fn(question, embedder, store, chunks)
        else:                                               # RAG v3
            answer, faith, rel = run_fn(question, embedder, store, chunks, reranker)

        latency_ms = round((time.time() - start) * 1000)

        p3    = precision_at_k(retrieved_texts, expected, k=3)
        r3    = recall_at_k(retrieved_texts, expected, k=3)
        mrr   = mrr_score(retrieved_texts, expected)
        rouge = rouge_score(answer, expected)

        results.append({
            "question":       question,
            "precision_at_3": p3,
            "recall_at_3":    r3,
            "mrr":            mrr,
            "rouge":          rouge,
            "faithfulness":   faith,
            "relevance":      rel,
            "latency_ms":     latency_ms,
        })

    return results


def main():
    print("\nLoading system...")
    embedder, store, chunks, reranker = load_system()

    print("Loading eval dataset...")
    dataset = load_eval_dataset()

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("v1", "both"):
        print(f"\nEvaluating RAG v1 ({len(dataset)} questions)...")
        v1_results = evaluate_pipeline(
            "RAG v1", run_rag_v1, embedder, store, chunks, reranker, dataset
        )
        print_table(v1_results, "RAG v1 — Simple Pipeline")

    if mode in ("v2", "both"):
        print(f"\nEvaluating RAG v2 ({len(dataset)} questions)...")
        v2_results = evaluate_pipeline(
            "RAG v2", run_rag_v2, embedder, store, chunks, reranker, dataset
        )
        print_table(v2_results, "RAG v2 — Agentic Pipeline")

    if mode in ("v3", "both"):
        print(f"\nEvaluating RAG v3 ({len(dataset)} questions)...")
        v3_results = evaluate_pipeline(
            "RAG v3", run_rag_v3, embedder, store, chunks, reranker, dataset
        )
        print_table(v3_results, "RAG v3 — LangChain Pipeline")


if __name__ == "__main__":
    main()