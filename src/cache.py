import sqlite3, os, json, threading, hashlib
from typing import Optional

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

class KVSqlite:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.conn:
            self.conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
    def get(self, k: str) -> Optional[dict]:
        with self.lock:
            row = self.conn.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None
    def put(self, k: str, v: dict) -> None:
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO kv(k,v) VALUES(?,?)", (k, json.dumps(v)))
