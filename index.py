
import argparse, json, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def load_chunks(jsonl_path):
    docs = []
    meta = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            docs.append(obj['text'])
            meta.append({'source': obj['source'], 'chunk_id': obj['chunk_id']})
    return docs, meta

def build_index(chunks, model_name='all-MiniLM-L6-v2', index_path='faiss_index.idx', meta_path='docs_meta.pkl'):
    model = SentenceTransformer(model_name)
    emb = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    faiss.normalize_L2(emb)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(chunks, f)
    print('Index built and saved:', index_path, meta_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', required=True, help='JSONL of chunks from ingest.py')
    parser.add_argument('--index_path', default='faiss_index.idx')
    parser.add_argument('--meta_path', default='docs_meta.pkl')
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    args = parser.parse_args()
    chunks, meta = load_chunks(args.chunks)
    build_index(chunks, model_name=args.model, index_path=args.index_path, meta_path=args.meta_path)
