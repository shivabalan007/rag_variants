from retrieval.query_rewriter import rewrite_query
from retrieval.reranker import CrossEncoderReranker
from llm.generator import generate_answer
from evaluation.faithfulness import check_faithfulness
from evaluation.relevance import check_relevance

def run_rag_v1(query, embedder, store, chunks):
    # REWRITE QUERY
    rewritten_query = rewrite_query(query)

    # EMBED REWRITTEN QUERY
    query_vector = embedder.embed_query(rewritten_query)

    # VECTOR SEARCH
    scores, indices = store.search(query_vector, top_k=3)

    retrieved = [chunks[i] for i in indices[0]]
 
    # RERANK RESULTS
    reranker = CrossEncoderReranker()
    reranked_chunks = reranker.rerank(rewritten_query, retrieved, top_k=3)

    texts = [chunk.text if hasattr(chunk, "text" ) else chunk for chunk in reranked_chunks]

    # GENERATE ANSWER
    answer = generate_answer(query, texts)

    # EVALUATE ANSWER
    faithfulness = check_faithfulness(query, answer, texts)
    relevance = check_relevance(query, answer)

    return answer, faithfulness, relevance
    