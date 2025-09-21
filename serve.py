
from fastapi import FastAPI
from pydantic import BaseModel
import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer
from prompt_builder import build_prompt
from llama_wrapper import generate_from_prompt

app = FastAPI()

class Query(BaseModel):
    q: str
    k: int = 4


INDEX_PATH = 'faiss_index.idx'
META_PATH = 'docs_meta.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'

print('Loading embedding model...')
emb_model = SentenceTransformer(MODEL_NAME)
print('Loading FAISS index...')
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, 'rb') as f:
    chunks = pickle.load(f)

@app.post('/query')
def query(q: Query):
    q_emb = emb_model.encode([q.q], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, q.k)
    retrieved = [chunks[idx] for idx in I[0]]
    prompt = build_prompt(q.q, retrieved)
    resp = generate_from_prompt(prompt)
    return {'answer': resp, 'contexts': retrieved, 'scores': D[0].tolist()}
