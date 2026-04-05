import os
import pickle
import faiss

from ingestion.run_ingestion import ingest
from ingestion.base import Document
from chunking.semantic_chunker import semantic_chunk
from chunking.sliding_window_chunker import sliding_window_chunk
from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder

print("Building FAISS index...")

# 1️⃣ Ingest
docs = ingest("data/test2.txt")

# 2️⃣ Chunk — semantic + sliding window (matches RAG pipeline)
chunks = []
for doc in docs:
    semantic_chunks = semantic_chunk(doc.text)
    for sc in semantic_chunks:
        window_chunks = sliding_window_chunk(sc, chunk_size=300, overlap=50)
        for chunk in window_chunks:                         # FIX: nested correctly
            chunks.append(Document(text=chunk, metadata=doc.metadata))

print(f"Total chunks: {len(chunks)}")

# 3️⃣ Embed
embedder = Embedder(EmbeddingConfig(model_name="all-MiniLM-L6-v2"))
embeddings = embedder.embed_documents(chunks)

# 4️⃣ Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# 5️⃣ Save artifacts
os.makedirs("artifacts", exist_ok=True)

faiss.write_index(index, "artifacts/faiss.index")

with open("artifacts/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"Index built successfully. {index.ntotal} vectors stored.")
