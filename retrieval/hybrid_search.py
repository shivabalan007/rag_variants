from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, chunks):
        self.chunks = chunks

        tokenized = [chunk.text.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def hybrid_search(self, query, vector_store, embedder, top_k=10):

        # -------- Vector Search --------
        query_vector = embedder.embed_query(query)

        v_scores, v_indices = vector_store.search(query_vector, top_k)

        vector_chunks = [self.chunks[i] for i in v_indices[0]]

        # -------- BM25 Search --------
        tokenized_query = query.split()

        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k]

        bm25_chunks = [self.chunks[i] for i in bm25_indices]

        # -------- Combine results --------
        combined = []

        for v, b in zip(vector_chunks, bm25_chunks):
            combined.append(v)
            combined.append(b)

        # remove duplicates
        unique_chunks = list({chunk.text: chunk for chunk in combined}.values())

        return unique_chunks[:15]