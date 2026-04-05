import streamlit as st
import faiss
import pickle
import tempfile
import os

from ingestion.base import Document
from chunking.semantic_chunker import semantic_chunk
from chunking.sliding_window_chunker import sliding_window_chunk
from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.reranker import CrossEncoderReranker

from rag_v1 import run_rag_v1
from rag_v2 import run_rag_v2


st.set_page_config(
    page_title="Advanced RAG",
    layout="wide"
)


@st.cache_resource
def load_system():
    index = faiss.read_index("artifacts/faiss.index")

    with open("artifacts/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedder = Embedder(EmbeddingConfig(model_name="all-MiniLM-L6-v2"))

    store = VectorStore(index.d)
    store.index = index

    reranker = CrossEncoderReranker()

    return embedder, store, chunks, reranker


def process_uploaded_file(uploaded_file, embedder):
    """Chunk and index an uploaded file, return new store and chunks."""

    # Read file content
    content = uploaded_file.read().decode("utf-8", errors="ignore")

    # Chunk
    new_chunks = []
    semantic_chunks = semantic_chunk(content)
    for sc in semantic_chunks:
        window_chunks = sliding_window_chunk(sc, chunk_size=300, overlap=50)
        for chunk in window_chunks:
            new_chunks.append(Document(text=chunk, metadata={"source": uploaded_file.name}))

    # Embed
    embeddings = embedder.embed_documents(new_chunks)

    # Build new store
    dim = embeddings.shape[1]
    new_store = VectorStore(dim)
    new_store.add(embeddings)

    return new_chunks, new_store


embedder, store, chunks, reranker = load_system()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = None

if "uploaded_store" not in st.session_state:
    st.session_state.uploaded_store = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "pipeline" not in st.session_state:
    st.session_state.pipeline = "RAG v1 - Simple"


# SIDEBAR
with st.sidebar:
    st.header("Advanced RAG")

    # PIPELINE SWITCHER                                     # FEATURE 2
    st.subheader("Pipeline")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("RAG v1", use_container_width=True):
            st.session_state.pipeline = "RAG v1 - Simple"
            st.rerun()
    with col2:
        if st.button("RAG v2", use_container_width=True):
            st.session_state.pipeline = "RAG v2 - Agentic"
            st.rerun()

    st.caption(f"Active: {st.session_state.pipeline}")

    st.divider()

    # DOCUMENT UPLOAD                                       # FEATURE 1
    st.subheader("Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "csv"]
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_filename:
            with st.spinner("Indexing document..."):
                new_chunks, new_store = process_uploaded_file(uploaded_file, embedder)
                st.session_state.uploaded_chunks = new_chunks
                st.session_state.uploaded_store = new_store
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.messages = []          # clear chat on new doc
            st.success(f"Indexed: {uploaded_file.name} — {len(new_chunks)} chunks")

        active_chunks = st.session_state.uploaded_chunks
        active_store = st.session_state.uploaded_store
        active_filename = st.session_state.uploaded_filename
    else:
        active_chunks = chunks
        active_store = store
        active_filename = "test1.txt"

    st.divider()
    st.caption(f"Document: {active_filename}")
    st.caption(f"Chunks: {len(active_chunks)}")

    # NEW CHAT BUTTON                                       # FEATURE 2
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# MAIN AREA
st.title("Advanced RAG System")
st.caption(f"Pipeline: {st.session_state.pipeline} | Document: {active_filename} | Chunks: {len(active_chunks)}")

st.divider()

# Display chat messages                                     # FEATURE 3
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "|||" in message["content"]:
            parts = message["content"].split("|||")
            st.write(parts[0])                              # answer
            with st.expander("Sources"):
                st.caption(parts[1])                        # sources
            st.caption(parts[2])                            # evaluation
        else:
            st.write(message["content"])

# Chat input
query = st.chat_input("Ask a question about your document...")

if query:
    # 1. Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # 2. Display user message immediately
    with st.chat_message("user"):
        st.write(query)

    # 3. Run RAG with spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.pipeline == "RAG v1 - Simple":
                answer, faithfulness, relevance = run_rag_v1(
                    query, embedder, active_store, active_chunks, reranker
                )
            else:
                answer, faithfulness, relevance = run_rag_v2(
                    query, embedder, active_store, active_chunks
                )

    # 4. Store answer + sources + evaluation together using ||| separator
    eval_line = f"Faithfulness: {faithfulness} | Relevance: {relevance}"
    sources_line = f"Retrieved from: {active_filename}"

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"{answer}|||{sources_line}|||{eval_line}"
    })

    st.rerun()