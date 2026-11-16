"""Hybrid RAG (BM25 + dense MiniLM, reciprocal-rank fusion) over REAL rail-vehicle maintenance logs.

Data: SMRT-Maintenance-Logs (Hugging Face, ASNIJKDNI/SMRT-Maintenance-Logs): 9,706 train / 1,079 test logs.
Each log = fault report "~~~~~" technician finding + action; label = component/action that fixed it (727 classes).
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
LABELS = None


def tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def split_log(text: str) -> tuple[str, str]:
    head, _, tail = text.partition("~~~~~")
    return head.strip(), tail.strip()


def label_names() -> list[str]:
    cache = HERE / "data" / "labels.txt"
    return cache.read_text().splitlines()


class HybridIndex:
    def __init__(self, texts: list[str], labels: list[int], cache: Path | None = None):
        self.texts, self.labels = texts, np.asarray(labels)
        self.bm25 = BM25Okapi([tok(t) for t in texts])
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        if cache and cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            if cache:
                np.save(cache, self.emb)

    def _bm25_rank(self, q: str, n: int) -> np.ndarray:
        return np.argsort(-self.bm25.get_scores(tok(q)))[:n]

    def _dense_rank(self, q: str, n: int) -> np.ndarray:
        v = self.model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.argsort(-(self.emb @ v))[:n]

    def search(self, q: str, k: int = 5, mode: str = "hybrid", pool: int = 50) -> list[int]:
        if mode == "bm25":
            return self._bm25_rank(q, k).tolist()
        if mode == "dense":
            return self._dense_rank(q, k).tolist()
        score: dict[int, float] = {}
        for ranking in (self._bm25_rank(q, pool), self._dense_rank(q, pool)):
            for r, i in enumerate(ranking):
                score[int(i)] = score.get(int(i), 0.0) + 1.0 / (60 + r)
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])[:k]]

    def timed_search(self, q: str, k: int = 5) -> tuple[list[int], float]:
        t = time.perf_counter()
        hits = self.search(q, k)
        return hits, (time.perf_counter() - t) * 1000


def load_index() -> tuple[HybridIndex, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_parquet(HERE / "data" / "train.parquet")
    te = pd.read_parquet(HERE / "data" / "test.parquet")
    idx = HybridIndex(tr.text.tolist(), tr.label.tolist(), cache=HERE / "data" / "train_emb.npy")
    return idx, tr, te
