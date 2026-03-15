import streamlit as st
import faiss
import pickle

from embeddings.base import EmbeddingConfig 
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore

from rag_v1 import run_rag_v1
from rag_v2 import run_rag_v2

st.title("Advanced RAG System")

st.write("Compare two RAG pipelines")

#Load System
@st.cache_resource
def load_system():
    index = faiss.read_index("artifacts/faiss.index")

    with open("artifacts/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedder = Embedder(
        EmbeddingConfig(model_name="all-MiniLM-L6-v2")
    )

    store = VectorStore(index.d)
    store.index = index

    return embedder, store, chunks

embedder, store, chunks = load_system()

#UI
rag_version = st.selectbox(
    "Select RAG Version", ["RAG v1 (Simple Pipeline)", "RAG v2 (Agentic Pipeline)"]
)

query = st.text_input("Ask a question")

if st.button("Run RAG"):
    if not query:
        st.warning("Please enter a question")
        st.stop()
    
    with st.spinner("Thinking..."):
        if rag_version == "RAG v1 (Simple Pipeline)":
            answer, faith, relevance = run_rag_v1(
                query, embedder, store, chunks
            )
        else:
            answer, faith, relevance = run_rag_v2(
                query, embedder, store, chunks
            )

    st.subheader("Answer")

    st.write(answer)

    st.subheader("Evaluation")

    st.write("Faithfulness:", faith)

    st.write("Relevance:", relevance)