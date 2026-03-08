from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.base import Document
from embeddings.base import EmbeddingConfig

class Embedder:
    def __init__(self, config:EmbeddingConfig):
        self.config = config
        self.model = SentenceTransformer(config.model_name)

    def embed_documents(self, documents: List[Document]):
        texts = [doc.text for doc in documents]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.config.normalize 
        )

        return embeddings
    
    def embed_query(self, query):
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.config.normalize
        )
        return embedding