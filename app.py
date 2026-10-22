"""
Streamlit UI for the College Notes Chatbot.
Keep this file in the same folder as notes_chatbot.py.

Run:
    pip install streamlit
    streamlit run app.py
"""
from pathlib import Path

import streamlit as st
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

import notes_chatbot as nb

NOTES_DIR = Path("notes")
NOTES_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="College Notes Chatbot", page_icon="📚", layout="wide")


# Cached models (load once) ---------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformer(nb.EMBED_MODEL)


@st.cache_resource(show_spinner="Loading Llama model...")
def get_llm(path, n_ctx):
    return Llama(model_path=path, n_ctx=n_ctx, verbose=False)


# Session state ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    if Path(nb.INDEX_FILE).exists() and Path(nb.META_FILE).exists():
        st.session_state.index, st.session_state.chunks = nb.load_index()

# Sidebar ---------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Your notes")

    uploads = st.file_uploader(
        "Upload PDF notes", type="pdf", accept_multiple_files=True
    )
    if st.button("Build / rebuild index", type="primary", use_container_width=True):
        for f in uploads or []:
            (NOTES_DIR / f.name).write_bytes(f.getbuffer())
        pages = nb.load_pdfs(NOTES_DIR)
        if not pages:
            st.error("No readable PDF text found. Scanned PDFs need OCR.")
        else:
            with st.spinner("Indexing your notes..."):
                chunks = nb.split_pages(pages)
                index = nb.build_index(chunks, get_embedder())
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.success(f"Indexed {len(pages)} pages ({len(chunks)} chunks).")

    files = sorted(p.name for p in NOTES_DIR.glob("*.pdf"))
    if files:
        st.caption("Files in notes/:")
        for name in files:
            st.write(f"• {name}")

    st.divider()
    st.subheader("Settings")
    model_path = st.text_input(
        "Llama .gguf path", "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    )
    k = st.slider("Passages to retrieve", 1, 8, 4)
    n_ctx = st.select_slider("Context size", [2048, 4096, 8192], value=4096)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat -------------------------------------------------------------------
st.title("College Notes Chatbot")
st.caption("Ask a question and get answers from your own notes, with page references.")


def show_sources(hits):
    with st.expander(f"📄 Sources ({len(hits)})"):
        for h in hits:
            st.markdown(f"**{h['source']} — page {h['page']}** · match {h['score']:.2f}")
            st.write(h["text"])
            st.divider()


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            show_sources(m["hits"])

if st.session_state.index is None:
    st.info("Upload your PDFs in the sidebar and click **Build / rebuild index** to start.")
    st.stop()

if not Path(model_path).exists():
    st.warning("Llama model file not found. Set the correct .gguf path in the sidebar.")
    st.stop()

question = st.chat_input("Ask about your notes...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching notes and thinking..."):
            hits = nb.retrieve(
                question,
                get_embedder(),
                st.session_state.index,
                st.session_state.chunks,
                k=k,
            )
            text, _ = nb.answer(question, get_llm(model_path, n_ctx), hits)
        st.markdown(text)
        if hits:
            show_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "hits": hits}
    )
