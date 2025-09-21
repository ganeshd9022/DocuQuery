
import time, requests
def test_query(q, k=4):
    t0 = time.perf_counter()
    r = requests.post('http://localhost:8000/query', json={'q': q, 'k': k}, timeout=120)
    t1 = time.perf_counter()
    return t1 - t0, r.json()

if __name__ == '__main__':
    qs = [
        'What is the objective of the project?',
        'How are documents embedded and stored?',
        'Explain the pipeline architecture.'
    ]
    for q in qs:
        dt, res = test_query(q)
        print(q, '->', f'{dt:.2f}s')
