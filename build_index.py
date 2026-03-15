import os
import pickle
import faiss

from ingestion.run_ingestion import ingest
from chunking.base import ChunkConfig
from chunking.fixed_chunker import fixed_chunk_document
from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder

print("Building FAISS index...")

# Load documents
docs = ingest("data/test1.txt")

chunk_config = ChunkConfig(chunk_size=300, overlap=50)

chunks = []
for doc in docs:
    chunks.extend(fixed_chunk_document(doc, chunk_config))

print("Total chunks:", len(chunks))

# Embed
embedder = Embedder(
    EmbeddingConfig(model_name="all-MiniLM-L6-v2")
)

embeddings = embedder.embed_documents(chunks)

# Create index
dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)

index.add(embeddings)

# Ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

# Save index
faiss.write_index(index, "artifacts/faiss.index")

# Save chunks
with open("artifacts/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Index built successfully.")