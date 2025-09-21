
from typing import List

TEMPLATE = """You are a helpful assistant that answers questions using only the information provided in the context below. If the answer is not contained in the context, say 'I don't know'.

Context:
{contexts}

Question: {question}
Answer:"""

def build_prompt(question: str, contexts: List[str], max_chars=3000):

    out = []
    total = 0
    for i,c in enumerate(contexts):
        if total + len(c) > max_chars:
            break
        out.append(f"[{i+1}] " + c)
        total += len(c)
    contexts_joined = "\n\n".join(out)
    return TEMPLATE.format(contexts=contexts_joined, question=question)
