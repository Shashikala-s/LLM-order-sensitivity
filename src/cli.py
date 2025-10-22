import argparse, json, os
from src.settings import settings, RunConfig
from src.runner import run_experiment
import sys


def main():
    p = argparse.ArgumentParser(description="ICL Order Sensitivity (modular-lite)")
    p.add_argument("--dataset", default="arc")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n_perm", type=int, default=50)
    p.add_argument("--ordering", choices=["random","similarity","length_asc","length_desc"], default="random")
    p.add_argument("--k_pool", type=int, default=50)
    p.add_argument("--vote_m", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_items", type=int, default=None)
    # we can override settings here too:
    p.add_argument("--model", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--cache_db", default=None)

    # generation params
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    args = p.parse_args()

    # allow overrides without editing files
    if args.model: settings.MODEL_NAME = args.model
    if args.device: settings.DEVICE = args.device
    if args.dtype: settings.DTYPE = args.dtype
    if args.data_dir: settings.DATA_DIR = args.data_dir
    if args.out_dir: settings.OUT_DIR = args.out_dir
    if args.cache_db: settings.CACHE_DB = args.cache_db

    cfg = RunConfig(
        dataset_key=args.dataset, k=args.k, n_permutations=args.n_perm,
        ordering=args.ordering, k_pool=args.k_pool, vote_m=args.vote_m,
        seed=args.seed, max_items=args.max_items, temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens
    )

    import torch
    torch.cuda.empty_cache()


    res = run_experiment(cfg)
    print(json.dumps(res, indent=2))
    print(f"Saved: {os.path.join(settings.OUT_DIR, 'summary.json')}")

if __name__ == "__main__":
    main()
