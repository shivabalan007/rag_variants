from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_k=3):
        pairs = [(query, chunk.text) for chunk in chunks]

        scores = self.model.predict(pairs)

        scored = list(zip(chunks, scores))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in scored[:top_k]]