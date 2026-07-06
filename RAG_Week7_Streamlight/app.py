"""
app.py
------
Streamlit web app for the Document Question Answering (RAG) system.

Pipeline: Upload PDF -> extract text -> chunk -> embed (local HuggingFace
model) -> store in FAISS -> ask question -> retrieve top-k chunks ->
generate grounded answer with Google Gemini.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from modules.pdf_loader import load_document
from modules.text_splitter import split_text
from modules.embedding_store import get_embedding_model, build_vector_store
from modules.rag_pipeline import configure_gemini, answer_question
from utils.helper import timer, document_stats, format_seconds

load_dotenv()

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")
st.title("📄 Document Question Answering System (RAG)")
st.caption("Upload a document, then ask questions grounded in its content.")

# --- Session state setup -----------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "logs" not in st.session_state:
    st.session_state.logs = []

# --- Sidebar: upload + process ------------------------------------------
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or .txt file", type=["pdf", "txt"])

    chunk_size = st.slider("Chunk size (characters)", 200, 1500, 800, 50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 10)
    top_k = st.slider("Chunks to retrieve per question", 1, 10, 4, 1)

    process_clicked = st.button("Process Document", type="primary", disabled=uploaded_file is None)

    if process_clicked and uploaded_file is not None:
        st.session_state.logs = []
        try:
            with timer() as t_load:
                raw_text = load_document(uploaded_file, filename=uploaded_file.name)
            st.session_state.logs.append(f"Loaded document in {format_seconds(t_load.elapsed)}")

            with timer() as t_split:
                chunks = split_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.session_state.logs.append(f"Split into {len(chunks)} chunks in {format_seconds(t_split.elapsed)}")

            with timer() as t_embed:
                embedding_model = get_embedding_model()
                vector_store = build_vector_store(chunks, embedding_model=embedding_model)
            st.session_state.logs.append(f"Embedded + indexed in {format_seconds(t_embed.elapsed)}")

            st.session_state.vector_store = vector_store
            st.session_state.doc_stats = document_stats(raw_text, chunks)
            st.session_state.chat_history = []
            st.success("Document processed! Ask a question below.")
        except Exception as e:
            st.error(f"Error processing document: {e}")

    if st.session_state.doc_stats:
        st.divider()
        st.subheader("📊 Document Stats")
        stats = st.session_state.doc_stats
        st.metric("Characters", stats["characters"])
        st.metric("Words", stats["words"])
        st.metric("Chunks", stats["chunks"])

    if st.session_state.logs:
        st.divider()
        st.subheader("📋 Processing Log")
        for log in st.session_state.logs:
            st.text(log)

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- Main area: ask questions --------------------------------------------
if st.session_state.vector_store is None:
    st.info("👈 Upload a document and click **Process Document** to get started.")
else:
    api_key_present = bool(os.getenv("GOOGLE_API_KEY"))
    if not api_key_present:
        st.warning(
            "No `GOOGLE_API_KEY` found. Add it to a `.env` file in this folder "
            "(see `.env.example`) to enable answer generation with Gemini."
        )

    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        try:
            configure_gemini()
            with timer() as t_answer:
                result = answer_question(st.session_state.vector_store, question, top_k=top_k)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "time": t_answer.elapsed,
            })
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ Error generating answer: {e}",
                "sources": [],
                "time": None,
            })

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                if msg.get("time") is not None:
                    st.caption(f"⏱️ Answered in {format_seconds(msg['time'])}")
                with st.expander("🔍 Retrieved Context"):
                    for i, (chunk, score) in enumerate(msg["sources"], start=1):
                        st.markdown(f"**Chunk {i}** (distance: `{score:.4f}`)")
                        st.text(chunk)
                        st.divider()
