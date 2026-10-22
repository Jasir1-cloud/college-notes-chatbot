"""
College Notes Chatbot (RAG)
---------------------------
PDF loader -> text splitter -> all-MiniLM-L6-v2 embeddings -> FAISS
-> similarity search -> local Llama chat model -> answer with sources.

Setup:
    pip install pypdf langchain-text-splitters sentence-transformers faiss-cpu llama-cpp-python numpy
    Download a Llama GGUF chat model, e.g. Llama-3.2-3B-Instruct-Q4_K_M.gguf

Usage:
    python notes_chatbot.py --pdfs notes/ --model models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from llama_cpp import Llama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "notes.faiss"
META_FILE = "notes_meta.json"


# 1. PDF loader ---------------------------------------------------------------
def load_pdfs(folder):
    """Return a list of (text, source_file, page_number)."""
    pages = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        reader = PdfReader(str(pdf))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((text, pdf.name, i))
    return pages


# 2. Text splitter ------------------------------------------------------------
def split_pages(pages, chunk_size=800, overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = []
    for text, source, page_no in pages:
        for piece in splitter.split_text(text):
            chunks.append({"text": piece, "source": source, "page": page_no})
    return chunks


# 3. Embeddings + FAISS index -------------------------------------------------
def build_index(chunks, embedder):
    vectors = embedder.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,  # cosine similarity via inner product
        show_progress_bar=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_FILE)
    Path(META_FILE).write_text(json.dumps(chunks), encoding="utf-8")
    return index


def load_index():
    index = faiss.read_index(INDEX_FILE)
    chunks = json.loads(Path(META_FILE).read_text(encoding="utf-8"))
    return index, chunks


# 4. Similarity search --------------------------------------------------------
def retrieve(question, embedder, index, chunks, k=4, min_score=0.25):
    q = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [
        {**chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i != -1 and s >= min_score
    ]


# 5. Local Llama answer -------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a study assistant. Answer the student's question using ONLY the "
    "note excerpts provided. Be concise and exam-friendly (short points where "
    "helpful). If the answer is not in the excerpts, say: "
    "'I couldn't find that in your notes.'"
)


def answer(question, llm, hits):
    if not hits:
        return "I couldn't find that in your notes.", []
    context = "\n\n".join(
        f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits
    )
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Notes:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=400,
    )
    text = result["choices"][0]["message"]["content"].strip()
    sources = sorted({f"{h['source']} p.{h['page']}" for h in hits})
    return text, sources


# CLI -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", help="Folder of PDF notes (builds/rebuilds index)")
    ap.add_argument("--model", required=True, help="Path to Llama .gguf file")
    ap.add_argument("--k", type=int, default=4, help="Passages to retrieve")
    args = ap.parse_args()

    embedder = SentenceTransformer(EMBED_MODEL)

    if args.pdfs:
        pages = load_pdfs(args.pdfs)
        chunks = split_pages(pages)
        print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks")
        index = build_index(chunks, embedder)
    else:
        index, chunks = load_index()

    llm = Llama(model_path=args.model, n_ctx=4096, verbose=False)

    print("\nAsk about your notes (type 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        hits = retrieve(q, embedder, index, chunks, k=args.k)
        text, sources = answer(q, llm, hits)
        print(f"\nBot: {text}")
        if sources:
            print("Sources:", ", ".join(sources))


if __name__ == "__main__":
    main()
