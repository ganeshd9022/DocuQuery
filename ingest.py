# ingest.py
# Simple ingestion and chunking of text files into JSONL chunks.
import os, argparse, json
from pathlib import Path

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

def ingest_dir(data_dir, out_file, max_chars=2000, overlap=200):
    p = Path(data_dir)
    items = list(p.glob('**/*.txt'))
    out = []
    for fp in items:
        text = fp.read_text(encoding='utf-8', errors='ignore')
        chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
        for i,c in enumerate(chunks):
            out.append({
                'source': str(fp),
                'chunk_id': f"{fp.name}::{i}",
                'text': c
            })
    with open(out_file, 'w', encoding='utf-8') as f:
        for obj in out:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f"Wrote {len(out)} chunks to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_raw', required=True)
    parser.add_argument('--max_chars', type=int, default=2000)
    parser.add_argument('--overlap', type=int, default=200)
    args = parser.parse_args()
    ingest_dir(args.data_dir, args.out_raw, args.max_chars, args.overlap)
