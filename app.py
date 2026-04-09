import streamlit as st
import faiss
import pickle
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
from rag_v3 import run_rag_v3


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
    content = uploaded_file.read().decode("utf-8", errors="ignore")

    new_chunks = []
    semantic_chunks = semantic_chunk(content)
    for sc in semantic_chunks:
        window_chunks = sliding_window_chunk(sc, chunk_size=300, overlap=50)
        for chunk in window_chunks:
            new_chunks.append(Document(
                text=chunk,
                metadata={"source": uploaded_file.name}
            ))

    embeddings = embedder.embed_documents(new_chunks)
    dim = embeddings.shape[1]
    new_store = VectorStore(dim)
    new_store.add(embeddings)

    return new_chunks, new_store


embedder, store, chunks, reranker = load_system()

# ── Session state init ────────────────────────────────────────────────────────
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


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Advanced RAG")

    if st.button("+ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Chat history
    if st.session_state.messages:
        st.caption("Recent")
        user_messages = [
            m["content"] for m in st.session_state.messages
            if m["role"] == "user"
        ]
        for msg in user_messages[-5:]:                      # show last 5 questions
            label = msg[:35] + "..." if len(msg) > 35 else msg
            st.markdown(
                f'<div style="font-size:12px;padding:6px 8px;color:var(--color-text-secondary);">'
                f'{label}</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # Pipeline selector
    st.caption("Pipeline")
    for label, key in [
        ("RAG v1 — Simple",   "RAG v1 - Simple"),
        ("RAG v2 — Agentic",  "RAG v2 - Agentic"),
        ("RAG v3 — LangChain","RAG v3 - LangChain"),
    ]:
        active = st.session_state.pipeline == key
        if st.button(
            label,
            use_container_width=True,
            type="primary" if active else "secondary"
        ):
            st.session_state.pipeline = key
            st.rerun()


# ── LANDING PAGE — no document uploaded ──────────────────────────────────────
if st.session_state.uploaded_filename is None:

    st.markdown("<br><br>", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])

    with col:
        st.markdown(
            "<h2 style='text-align:center;font-weight:500;'>"
            "What would you like to explore?</h2>",
            unsafe_allow_html=True
        )

        st.chat_input(
            "Upload a document to start chatting...",
            disabled = True,
            key="landing_chat"
        )
        
        st.markdown(
            "<p style='text-align:center;color:var(--color-text-secondary);"
            "font-size:14px;margin-bottom:24px;'>"
            "Upload a document to start asking questions</p>",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "label",
            type=["txt", "pdf", "csv"],
            label_visibility="collapsed",
            key="landing_upload"
        )

        st.markdown(
            "<div style='text-align:center;font-size:11px;"
            "color:var(--color-text-tertiary);margin-top:8px;'>"
            "Supported formats: TXT · PDF · CSV</div>",
            unsafe_allow_html=True
        )

        if uploaded_file is not None:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                new_chunks, new_store = process_uploaded_file(
                    uploaded_file, embedder
                )
                st.session_state.uploaded_chunks = new_chunks
                st.session_state.uploaded_store = new_store
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.messages = []
            st.rerun()

    st.stop()


# ── ACTIVE DOCUMENT ───────────────────────────────────────────────────────────
active_chunks   = st.session_state.uploaded_chunks
active_store    = st.session_state.uploaded_store
active_filename = st.session_state.uploaded_filename


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='display:flex;align-items:center;justify-content:space-between;"
    f"padding-bottom:8px;border-bottom:0.5px solid var(--color-border-tertiary);'>"
    f"<span style='font-size:18px;font-weight:500;'>Advanced RAG System</span>"
    f"<div style='display:flex;gap:8px;'>"
    f"<span style='font-size:11px;padding:3px 10px;border-radius:20px;"
    f"background:#E6F1FB;color:#0C447C;'>{st.session_state.pipeline}</span>"
    f"<span style='font-size:11px;padding:3px 10px;border-radius:20px;"
    f"background:#EAF3DE;color:#27500A;'>{active_filename}</span>"
    f"<span style='font-size:11px;padding:3px 10px;border-radius:20px;"
    f"background:#EEEDFE;color:#3C3489;'>{len(active_chunks)} chunks</span>"
    f"</div></div>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "|||" in message["content"]:
            parts = message["content"].split("|||")
            st.write(parts[0])
            with st.expander("Sources"):
                st.caption(parts[1])
            st.caption(parts[2])
        else:
            st.write(message["content"])

# ── CHAT INPUT ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about your document...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.pipeline == "RAG v1 - Simple":
                answer, faithfulness, relevance = run_rag_v1(
                    query, embedder, active_store, active_chunks, reranker
                )
            elif st.session_state.pipeline == "RAG v2 - Agentic":
                answer, faithfulness, relevance = run_rag_v2(
                    query, embedder, active_store, active_chunks
                )
            else:
                answer, faithfulness, relevance = run_rag_v3(
                    query, embedder, active_store, active_chunks, reranker
                )

    eval_line    = f"Faithfulness: {faithfulness} | Relevance: {relevance}"
    sources_line = f"Retrieved from: {active_filename}"

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"{answer}|||{sources_line}|||{eval_line}"
    })

    st.rerun()