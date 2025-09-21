
import streamlit as st
import faiss, tempfile, os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from prompt_builder import build_prompt
from llama_wrapper import generate_from_prompt
import pdfplumber


def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            text += "\n"
    return text

def chunk_text(text, max_chars=2000, overlap=200):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)
        if start >= n:
            break
    return chunks

def build_faiss_index(chunks, model):
    embs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embs)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    return index, chunks


st.title("📚 CPU-based RAG with PDF Upload")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("🔍 Enter your query")
k = st.slider("Number of contexts to retrieve (k)", 1, 8, 4)

if "emb_model" not in st.session_state:
    st.session_state["emb_model"] = SentenceTransformer("all-MiniLM-L6-v2")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        index, chunks_list = build_faiss_index(chunks, st.session_state["emb_model"])
        st.session_state["index"] = index
        st.session_state["chunks"] = chunks_list
    st.success(f"✅ Processed {len(st.session_state['chunks'])} chunks from PDF")

if st.button("Get Answer"):
    if "index" not in st.session_state:
        st.error("⚠️ Please upload a PDF first.")
    elif not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving and generating..."):
            q_emb = st.session_state["emb_model"].encode([query], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)
            D, I = st.session_state["index"].search(q_emb, k)
            retrieved = [st.session_state["chunks"][idx] for idx in I[0]]

           
            prompt = build_prompt(query, retrieved)


            answer = generate_from_prompt(prompt)

  
            st.subheader("💡 Answer")
            st.write(answer)

            st.subheader("📖 Retrieved Contexts")
            for i, c in enumerate(retrieved):
                st.markdown(f"**Context {i+1}:**")
                st.write(c[:1000])
