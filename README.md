DocuQuery – CPU-Based RAG with PDF Upload

DocuQuery 🧠📖 is a lightweight Retrieval-Augmented Generation (RAG) pipeline built for CPU-only environments.
It allows you to upload PDFs, ask natural language questions, and get precise answers — all through a simple Streamlit UI.
No GPU required 🚀.

✨ Features :
- 📂 Upload PDFs and instantly extract knowledge.
- 🔍 Efficient document retrieval with FAISS (CPU).
- 🔑 Sentence-Transformers (MiniLM) for generating embeddings.
- 🤖 LLM inference with llama.cpp (quantized models for CPU).
- 🖥️ Streamlit-based interactive UI for seamless Q&A.
- ⚡ Optimized for low memory usage and CPU-only systems.

🛠️ Tech Stack :
- Python 3.12
- FAISS (CPU) – similarity search
- Sentence-Transformers (MiniLM) – embeddings
- llama.cpp – lightweight LLM inference on CPU
- Streamlit – user interface
- pdfplumber – PDF text extraction
