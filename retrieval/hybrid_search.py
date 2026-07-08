import time

from rank_bm25 import BM25Okapi

from core.retrieval_result import RetrievalResult


class HybridRetriever:

    def __init__(self, chunks):
        self.chunks = chunks

        tokenized = [chunk.text.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def hybrid_search(self, query, vector_store, embedder, top_k=10):

        start_time = time.perf_counter()

        # -------------------------------
        # Dense Retrieval (FAISS)
        # -------------------------------

        query_vector = embedder.embed_query(query)

        vector_scores, vector_indices = vector_store.search(
            query_vector,
            top_k
        )

        # -------------------------------
        # Sparse Retrieval (BM25)
        # -------------------------------

        tokenized_query = query.split()

        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k]

        # -------------------------------
        # Merge Results
        # -------------------------------

        result = RetrievalResult()

        added_chunks = set()

        # Add FAISS results
        for score, index in zip(vector_scores[0], vector_indices[0]):

            if index == -1:
                continue

            chunk = self.chunks[index]

            if chunk.text in added_chunks:
                continue

            added_chunks.add(chunk.text)

            result.add_chunk(
                chunk=chunk,
                vector_score=float(score),
                bm25_score=0.0,
            )

        # Add BM25 results
        for index in bm25_indices:

            chunk = self.chunks[index]

            if chunk.text in added_chunks:

                # Update existing BM25 score
                for item in result.retrieved_chunks:
                    if item.chunk.text == chunk.text:
                        item.bm25_score = float(bm25_scores[index])
                        break

            else:

                added_chunks.add(chunk.text)

                result.add_chunk(
                    chunk=chunk,
                    vector_score=0.0,
                    bm25_score=float(bm25_scores[index]),
                )

        result.retrieval_latency = (
            time.perf_counter() - start_time
        )

        return result


"""
Performs hybrid retrieval by combining FAISS semantic search and BM25 keyword
search into a RetrievalResult. Preserves chunk-to-score relationships so
routing, monitoring, memory, and reranking can reuse the same retrieval data.
"""