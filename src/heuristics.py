import numpy as np, random
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)
    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

def candidate_pool(train: list[dict], query_q: str, k_pool: int, emb: Embedder) -> tuple[list[dict], np.ndarray]:
    qs = [x["question"] for x in train]
    qv = emb.encode([query_q])[0]
    tv = emb.encode(qs)
    sims = tv @ qv
    idx = np.argsort(-sims)[:k_pool]
    return [train[i] for i in idx], sims[idx]

def enforce_label_balance(examples: list[dict], k: int) -> list[dict]:
    buckets = {l: [] for l in "ABCD"}
    for e in examples: buckets[e["answer"]].append(e)
    out, i, labels = [], 0, ["A","B","C","D"]
    while len(out) < k and any(buckets.values()):
        lab = labels[i % 4]
        if buckets[lab]: out.append(buckets[lab].pop(0))
        i += 1
        if i > 8*k: break
    if len(out) < k:
        out.extend([e for e in examples if e not in out][:k-len(out)])
    return out[:k]

def order_by_length(examples: list[dict], ascending=True) -> list[dict]:
    key = lambda e: len(e["question"]) + sum(len(v) for v in e["choices"].values())
    return sorted(examples, key=key, reverse=not ascending)

def order_by_similarity(examples: list[dict], sims: np.ndarray, ascending=False) -> list[dict]:
    order = np.argsort(sims if ascending else -sims)
    return [examples[i] for i in order]

def order_random(examples: list[dict], rng: random.Random) -> list[dict]:
    ex = examples[:]; rng.shuffle(ex); return ex
