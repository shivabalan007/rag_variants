import streamlit as st
import os
import tempfile
import faiss
import pickle
import uuid

from ingestion.base import Document
from ingestion.run_ingestion import ingest
from chunking.semantic_chunker import semantic_chunk
from chunking.sliding_window_chunker import sliding_window_chunk
from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.pg_vector_store import PGVectorStore
from retrieval.reranker import CrossEncoderReranker
from retrieval.reranker_legacy import CrossEncoderReranker as LegacyReranker

from rag_v1 import run_rag_v1
from rag_v2 import run_rag_v2
from rag_v3 import run_rag_v3

from database.init_db import init_database

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

@st.cache_resource
def load_legacy_reranker():
    return LegacyReranker()

init_database()

legacy_reranker = load_legacy_reranker()

def process_uploaded_file(uploaded_file, embedder):

    import hashlib
    file_bytes = uploaded_file.getvalue()

    content_hash = hashlib.sha256(file_bytes).hexdigest()

    pg_store = PGVectorStore()

    existing_document_id = pg_store.get_document_by_hash(content_hash)

    if existing_document_id:
        return ([],None,pg_store,existing_document_id)
    
    with tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:

        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        documents = ingest(temp_path)

    finally:
        os.unlink(temp_path)

    content = "\n".join(document.text for document in documents)
    
    new_chunks = []
    semantic_chunks = semantic_chunk(content)

    for sc in semantic_chunks:
        window_chunks = sliding_window_chunk(sc,chunk_size=300,overlap=50)
        for chunk in window_chunks:
            new_chunks.append(
                Document(
                    text=chunk,
                    metadata={
                        "source": uploaded_file.name
                    }
                )
            )

    if not new_chunks:
        raise ValueError("No chunks were created from the uploaded file.")

    embeddings = embedder.embed_documents(new_chunks)
    # V1 / V2 → FAISS
    dim = embeddings.shape[1]
    faiss_store = VectorStore(dim)
    faiss_store.add(embeddings)

    # V3 → PostgreSQL + pgvector
    document_id = str(uuid.uuid4())

    pg_store.add(
        document_id=document_id,
        filename=uploaded_file.name,
        content_hash=content_hash,
        chunks=new_chunks,
        embeddings=embeddings
    )

    return (
        new_chunks,
        faiss_store,
        pg_store,
        document_id
    )


embedder, store, chunks, reranker = load_system()

# Session state 
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = None

if "uploaded_store" not in st.session_state:
    st.session_state.uploaded_store = None

if "uploaded_pg_store" not in st.session_state:
    st.session_state.uploaded_pg_store = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "uploaded_document_id" not in st.session_state:
    st.session_state.uploaded_document_id = None

if "pipeline" not in st.session_state:
    st.session_state.pipeline = "RAG v3 - LangChain"

if "first_question" not in st.session_state:       
    st.session_state.first_question = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# print("SESSION:", st.session_state.session_id)

# ── SIDEBAR 
with st.sidebar:
    st.header("Advanced RAG")

    if st.button("+ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.first_question = None   
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    # FIX 1: show only first question in recent
    if st.session_state.first_question:
        st.caption("Recent")
        label = (
            st.session_state.first_question[:35] + "..."
            if len(st.session_state.first_question) > 35
            else st.session_state.first_question
        )
        st.markdown(
            f'<div style="font-size:12px;padding:6px 8px;'
            f'background:var(--color-background-primary);'
            f'border-radius:6px;color:var(--color-text-primary);">'
            f'{label}</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # Pipeline selector
    st.caption("Pipeline")
    for label, key in [
        ("RAG v1 — Simple",    "RAG v1 - Simple"),
        ("RAG v2 — Agentic",   "RAG v2 - Agentic"),
        ("RAG v3 — LangChain", "RAG v3 - LangChain"),
    ]:
        active = st.session_state.pipeline == key
        if st.button(
            label,
            use_container_width=True,
            type="primary" if active else "secondary"
        ):
            st.session_state.pipeline = key
            st.rerun()


# ── LANDING PAGE 
if st.session_state.uploaded_filename is None:

    # FIX 2: bold title on landing page
    st.markdown(
        "<h1 style='text-align:center;font-weight:700;margin-top:80px;'>"
        "Advanced RAG System</h1>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # FIX 3: full width chat bar on landing — no columns
    st.chat_input(
        "Upload a document below to start chatting...",
        disabled=True,
        key="landing_chat"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(
            "<p style='text-align:center;color:var(--color-text-secondary);"
            "font-size:14px;margin-bottom:16px;'>"
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
                new_chunks, new_store,  new_pg_store, document_id = process_uploaded_file(
                    uploaded_file, embedder
                )
                st.session_state.uploaded_chunks = new_chunks
                st.session_state.uploaded_store = new_store
                st.session_state.uploaded_pg_store = new_pg_store
                st.session_state.uploaded_document_id = document_id
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.messages = []
                st.session_state.first_question = None
            st.rerun()

    st.stop()


# ── ACTIVE DOCUMENT 
active_chunks = st.session_state.uploaded_chunks
active_store = st.session_state.uploaded_store
active_pg_store = st.session_state.uploaded_pg_store
active_filename = st.session_state.uploaded_filename
active_document_id = st.session_state.uploaded_document_id

if active_document_id and active_pg_store:
    active_chunk_count = active_pg_store.document_chunk_count(active_document_id)
else:
    active_chunk_count = len(active_chunks or [])

# ── MAIN AREA 
# FIX 2: bold title on chat page
st.markdown(
    f"<div style='display:flex;align-items:center;justify-content:space-between;"
    f"padding-bottom:8px;border-bottom:0.5px solid var(--color-border-tertiary);'>"
    f"<span style='font-size:20px;font-weight:700;'>Advanced RAG System</span>"
    f"<div style='display:flex;gap:8px;'>"
    f"<span style='font-size:11px;padding:3px 10px;border-radius:20px;"
    f"background:#E6F1FB;color:#0C447C;'>{st.session_state.pipeline}</span>"
    f"<span style='font-size:11px;padding:3px 10px;border-radius:20px;"
    f"background:#EAF3DE;color:#27500A;'>{active_filename}</span>"
    f"<span style='font-size:11px;padding:3px 10px;border-radius:20px;"
    f"background:#EEEDFE;color:#3C3489;'>{active_chunk_count} chunks</span>"
    f"</div></div>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # RAG V3
            if "answer" in message:
                st.write(message["answer"])

                pipeline = message["pipeline"]
                metrics = pipeline["metrics"]
                retrieval = pipeline["retrieval"]

                is_knowledge = (pipeline["answer_source"] == "Knowledge")

                source_name = "Unknown"

                if (retrieval and retrieval.retrieved_chunks and len(retrieval.retrieved_chunks) > 0):
                    source_name = retrieval.retrieved_chunks[0].chunk.metadata.get(
                        "source",
                        "Unknown"
                    )
                
                level = pipeline["confidence_level"]

                if not is_knowledge:
                    st.info(f"🔵 {pipeline['answer_source']} Response")
                elif level == "HIGH":
                    st.success("🟢 HIGH Confidence")

                elif level == "MEDIUM":
                    st.warning("🟡 MEDIUM Confidence")
                else:
                    st.error("🔴 LOW Confidence")
                
                # Pipeline Details
                with st.expander("Pipeline Details", expanded=True):

                    confidence = (
                        f"{pipeline['confidence']:.3f}"
                        if pipeline["confidence"] is not None
                        else "N/A"
                    )

                    route = (
                        pipeline["route"]
                        if pipeline["route"] is not None
                        else "N/A"
                    )

                    confidence_level = (
                        pipeline["confidence_level"]
                        if pipeline["confidence_level"] is not None
                        else "N/A"
                    )

                    st.markdown(
                        f"""
                📄 **Route:** `{route}`

                🎯 **Confidence:** `{confidence}` (**{confidence_level}**)

                📚 **Answer Source:** `{pipeline["answer_source"]}`

                📄 **File:** `{source_name}`
                    """
                    )

                    st.write("**Reason:**", pipeline["reason"] if pipeline["reason"] else "N/A")

                    if pipeline["warning"]:
                        st.warning(pipeline["warning"])

                    if pipeline.get("rewritten_query"):
                        st.divider()
                        st.subheader("Rewritten Query")
                        st.code(pipeline["rewritten_query"])
                    
                    if is_knowledge:
                        st.divider()

                        st.write("### Evaluation")

                        if metrics.faithfulness == "YES":
                            st.success("Faithfulness : YES")
                        else:
                            st.error("Faithfulness : NO")

                        if metrics.relevance == "YES":
                            st.success("Relevance : YES")
                        else:
                            st.error("Relevance : NO")

                        st.info(f"Overlap : {pipeline['overlap']:.3f}")

                    else:
                        st.divider()

                        st.info("Evaluation not applicable.")

                    st.divider()
                    st.write("### Monitoring")

                    metrics = pipeline["metrics"]

                    if is_knowledge:
                        st.write("**Retrieval Latency:**", f"{metrics.retrieval_latency:.3f} sec")
                        st.write("**Rerank Latency:**", f"{metrics.rerank_latency:.3f} sec")
                    st.write("**Generation Latency:**", f"{metrics.generation_latency:.3f} sec")
                    st.write("**Total Latency:**", f"{metrics.total_latency:.3f} sec")
                    
                    st.divider()
                    st.write("### Token Usage")

                    st.write("Prompt Tokens :", metrics.prompt_tokens)
                    st.write("Completion Tokens :", metrics.completion_tokens)
                    st.write("Total Tokens :", metrics.total_tokens)
                    st.write("Estimated Cost :", f"${metrics.estimated_cost:.6f}")

                    if is_knowledge:
                        st.divider()
                    
                        st.write("### Retrieved Chunks")
                    

                        for i, item in enumerate(retrieval.retrieved_chunks, 1):
            
                            with st.expander(f"Chunk {i} | Rerank: {item.rerank_score:.2f} | Vector: {item.vector_score:.2f}"):

                                st.write("Vector Score :", item.vector_score)
                                st.write("BM25 Score :", item.bm25_score)
                                st.write("Rerank Score :", item.rerank_score)

                                if hasattr(item, "chunk"):
                                    st.caption(item.chunk.text[:250] + "...")

                    st.divider()       
                    
                    with st.expander("Pipeline Execution"):
                        st.markdown("✅ Query Received")

                        if is_knowledge:
                            st.markdown(f"➡️ Route Selected : **{pipeline['route']}**")
                            st.markdown(f"➡️ Retrieved {len(retrieval.retrieved_chunks)} chunks")
                            st.markdown("➡️ Cross Encoder Reranking")
                            st.markdown(f"➡️ Answer Generated from **{pipeline['answer_source']}**")
                            st.markdown(f"✅ Confidence : **{pipeline['confidence_level']}**")
                        else:
                            st.markdown(
                                f"➡️ Intent : "
                                f"**{pipeline['answer_source']}**"
                            )

                            st.markdown(
                                f"➡️ Answer Generated from "
                                f"**{pipeline['answer_source']}**"
                            )
                        st.divider()

                    if pipeline["web_result"]:
                        st.divider()
                        with st.expander("Web Search"):
                            st.write("**Provider:**", pipeline["web_result"].provider)
                            st.write("**Results:**", pipeline["web_result"].result_count)
                            st.write( "**Latency:**",f"{pipeline['web_result'].search_latency:.3f} sec")

                            st.write("### Sources")

                            for source in pipeline["web_result"].sources:
                                st.markdown(
                                    f"- [{source.title}]({source.url})"
                                )

            # RAG v1 / RAG v2
            elif "content" in message and "|||" in message["content"]:

                parts = message["content"].split("|||")

                st.write(parts[0])

                with st.expander("Sources"):
                    st.caption(parts[1])

                st.caption(parts[2])
            
            else:
                st.write(message["content"])

        else:
            st.write(message["content"])



# ── CHAT INPUT 
query = st.chat_input("Ask a question about your document...")

if query:
    # FIX 1: save only the first question
    if st.session_state.first_question is None:
        st.session_state.first_question = query

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
                    query, embedder, active_store, active_chunks, legacy_reranker
                )
            elif st.session_state.pipeline == "RAG v2 - Agentic":
                answer, faithfulness, relevance = run_rag_v2(
                    query, embedder, active_store, active_chunks
                )
            else:
                state = run_rag_v3(query=query, embedder=embedder, session_id=st.session_state.session_id, reranker=reranker)
                answer = state.answer
                faithfulness = state.faithfulness
                relevance = state.relevance
                pipeline_details = {
                "route": state.route,
                "reason": state.route_reason,
                "confidence": state.route_confidence,
                "confidence_level": state.confidence_level,
                "answer_source": state.answer_source,
                "warning": state.warning,
                "overlap": state.overlap,
                "metrics": state.metrics,
                "retrieval": state.retrieval_result,
                "web_result": state.web_result,
            }


    if st.session_state.pipeline == "RAG v3 - LangChain":

        st.session_state.messages.append({
            "role": "assistant",
            "answer": answer,
            "pipeline": pipeline_details
        })

    else:

        eval_line = f"Faithfulness: {faithfulness} | Relevance: {relevance}"
        sources_line = f"Retrieved from: {active_filename}"

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{answer}|||{sources_line}|||{eval_line}"
        })
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    st.rerun()


# Main Streamlit UI — ChatGPT-style chat interface with sidebar pipeline switcher and document upload. Supports all three RAG versions with evaluation display per answer.
