import faiss
import numpy as np
import os

class VectorStore:
    def __init__(self,dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray):
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray,top_k: int):
        scores, indices = self.index.search(query_vector, top_k)
        return scores, indices

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/faiss_index")    

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}/faiss_index")