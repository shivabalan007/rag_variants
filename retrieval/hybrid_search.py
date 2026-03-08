from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized = [chunk.text.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def hybrid_search(self, query, vector_store, embedder, tok_k=5):
        #Vector search
        query_vector = embedder.embed_query(query)
        v_scores, v_indices = vector_store.search(query_vector, top_k=tok_k)

        #BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        #Combine scores
        combined = []

        for idx in range(len(self.chunks)):
            vector_score = 0
            import faiss
            if i in v_indices[0]:
                idx_pos = list(v_indices[0]).index(i)
                vector_score = v_scores[0][idx_pos]

            score = vector_score + 0.3 * bm25_scores[i]
            combined.append((self.chunks[i], score))

        combined.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in combined[:top_k]]
