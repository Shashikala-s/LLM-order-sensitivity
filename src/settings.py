import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    DEVICE: str = os.getenv("DEVICE", "auto")
    DTYPE: str = os.getenv("DTYPE", "bfloat16")
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    OUT_DIR: str = os.getenv("OUT_DIR", "outputs/run")
    CACHE_DB: str = os.getenv("CACHE_DB", "outputs/cache.sqlite")
    DB_URL: str | None = os.getenv("DB_URL") or None

settings = Settings()

@dataclass
class RunConfig:
    dataset_key: str = "arc"
    k: int = 5
    n_permutations: int = 50
    ordering: str = "random"
    k_pool: int = 50
    vote_m: int = 1
    seed: int = 123
    max_items: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 8
