
import json
def read_jsonl(path):
    out = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            out.append(json.loads(line))
    return out
