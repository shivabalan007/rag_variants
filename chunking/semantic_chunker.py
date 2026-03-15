from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_chunk(text, threshold=0.7):

    sentences = text.split(".")

    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):

        sim = np.dot(
            embeddings[i],
            embeddings[i-1]
        ) / (
            np.linalg.norm(embeddings[i]) *
            np.linalg.norm(embeddings[i-1])
        )

        if sim < threshold:
            chunks.append(".".join(current_chunk))
            current_chunk = []

        current_chunk.append(sentences[i])

    chunks.append(".".join(current_chunk))

    return chunks