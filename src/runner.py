import os, re, json, random, numpy as np
from tqdm import tqdm
from src.settings import settings, RunConfig
from src.prompts import build_prompt
from src.cache import KVSqlite, sha256
from src.models.hf_local import HFLocal
from src.heuristics import (
    Embedder, candidate_pool, enforce_label_balance,
    order_random, order_by_similarity, order_by_length
)
from src.data import arc as arc_loader

ANSWER_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)

def parse_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1).upper() if m else None

def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    t, p = np.array(y_true), np.array(y_pred)
    return float((t == p).mean())

def flip_rate(matrix_correct: np.ndarray) -> float:
    varies = (matrix_correct.min(axis=1) != matrix_correct.max(axis=1))
    return float(varies.mean())

def run_experiment(cfg: RunConfig) -> dict:
    random.seed(cfg.seed); np.random.seed(cfg.seed)

    # paths
    os.makedirs(settings.OUT_DIR, exist_ok=True)
    cache = KVSqlite(settings.CACHE_DB)

    # load data
    if cfg.dataset_key == "commonsense":
        train = arc_loader.load_split(settings.DATA_DIR, "train")
        test  = arc_loader.load_split(settings.DATA_DIR, "test")
    else:
        raise ValueError(f"Unknown dataset_key={cfg.dataset_key}")

    if cfg.max_items:
        test = test[:cfg.max_items]

    # models
    lm = HFLocal(settings.MODEL_NAME, device=settings.DEVICE, dtype=settings.DTYPE)
    emb = Embedder()

    gold, vote_preds = [], []
    correctness_cols: list[list[int]] = []

    for item in tqdm(test, desc="items"):
        q = item["question"]
        pool, _ = candidate_pool(train, q, cfg.k_pool, emb) if cfg.k > 0 else ([], None)
        demos_k = enforce_label_balance(pool, cfg.k) if cfg.k > 0 else []

        # permutations / orderings
        orders: list[list[dict]] = []
        if cfg.k <= 1:
            orders = [demos_k]
        else:
            seen = set()
            while len(orders) < cfg.n_permutations:
                if cfg.ordering == "random":
                    o = order_random(demos_k, random)
                elif cfg.ordering == "similarity":
                    qs = [d["question"] for d in demos_k]
                    sims = emb.encode(qs) @ emb.encode([q])[0]
                    o = order_by_similarity(demos_k, sims, ascending=False)
                elif cfg.ordering == "length_asc":
                    o = order_by_length(demos_k, ascending=True)
                elif cfg.ordering == "length_desc":
                    o = order_by_length(demos_k, ascending=False)
                else:
                    o = order_random(demos_k, random)
                key = tuple(id(x) for x in o)
                if key not in seen:
                    orders.append(o); seen.add(key)

        # generate for each order
        preds_this = []
        for demos in orders:
            prompt = build_prompt(item, demos)
            k = sha256(prompt)
            cached = cache.get(k)
            if cached is None:
                text = lm.generate(prompt, max_new_tokens=8)
                cache.put(k, {"text": text})
            else:
                text = cached["text"]
            pred = parse_answer(text) or "Z"
            preds_this.append(pred)

        # vote or first-order
        if cfg.vote_m > 1 and len(preds_this) >= cfg.vote_m:
            vote = max("ABCD", key=preds_this[:cfg.vote_m].count)
        else:
            vote = preds_this[0] if preds_this else "Z"

        gold.append(item["answer"])
        vote_preds.append(vote)
        if preds_this:
            correctness_cols.append([1 if p == item["answer"] else 0 for p in preds_this])

    # metrics
    out = {
        "dataset": cfg.dataset_key,
        "k": cfg.k,
        "ordering": cfg.ordering,
        "n_permutations": cfg.n_permutations,
        "accuracy_vote": accuracy(gold, vote_preds),
    }

    if correctness_cols:
        M = np.array(correctness_cols, dtype=np.int32)
        per_perm = M.mean(axis=0).tolist()
        out["flip_rate"] = flip_rate(M)
        out["best_worst_gap"] = float(np.max(per_perm) - np.min(per_perm))
        out["permutation_accuracies"] = per_perm

    with open(os.path.join(settings.OUT_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out
