"""Tìm kiếm BM25 / BM25F trên inverted index đã nén.

Chi phí mỗi truy vấn tỉ lệ với tổng độ dài posting của các term trong truy vấn, không tỉ lệ
với số tài liệu: chỉ những tài liệu xuất hiện trong posting mới được cộng điểm.

BM25F (Robertson & Zaragoza 2009), với field f, trọng số w_f, chuẩn hóa độ dài b_f:
    tf~(t,d) = Σ_f w_f · tf_f(t,d) / (1 − b_f + b_f · len_f(d) / avglen_f)
    score(q,d) = Σ_t qtf(t) · idf(t) · tf~ · (k1 + 1) / (tf~ + k1)
    idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))      (dạng không âm như Lucene)
Khi chỉ có một field và w = 1 thì công thức trùng BM25 thường.
"""
from __future__ import annotations

import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from analysis.tokenizer import get_tokenizer
from index import codec as codec_mod


class InvertedIndex:
    def __init__(self, path: str | Path, cache_terms: int = 4096):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "index_manifest.json").read_text(encoding="utf-8"))
        self.codec = self.manifest["codec"]
        self.fields: list[str] = self.manifest["fields"]
        self.n_docs: int = self.manifest["n_docs"]
        self.avgdl: dict[str, float] = self.manifest["avgdl"]
        d = pq.read_table(self.path / "dictionary.parquet").to_pydict()
        self.dictionary = {(f, t): (df, off, nb) for f, t, df, off, nb in
                           zip(d["field"], d["term"], d["df"], d["offset"], d["nbytes"])}
        a = pq.read_table(self.path / "terms_any.parquet").to_pydict()
        self.df_any = dict(zip(a["term"], a["df_any"]))
        lens = pq.read_table(self.path / "doc_lengths.parquet")
        self.doc_len = {f: lens.column(f"len_{f}").to_numpy().astype(np.float64) for f in self.fields}
        dm = self.path / "doc_map.parquet"
        self.doc_to_premise = pq.read_table(dm).column("premise_id").to_numpy() if dm.exists() else np.arange(self.n_docs)
        self._postings = np.memmap(self.path / "postings.bin", dtype=np.uint8, mode="r") \
            if (self.path / "postings.bin").stat().st_size else np.zeros(0, np.uint8)
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_terms
        self.stats = {"decoded_postings": 0, "cache_hits": 0}

    def postings(self, field: str, term: str) -> tuple[np.ndarray, np.ndarray] | None:
        key = (field, term)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.stats["cache_hits"] += 1
            return hit
        entry = self.dictionary.get(key)
        if entry is None:
            return None
        df, off, nb = entry
        res = codec_mod.decode_postings(self._postings[off:off + nb].tobytes(), df, self.codec)
        self.stats["decoded_postings"] += df
        if self._cache_size:
            self._cache[key] = res
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return res


@dataclass
class BM25FParams:
    k1: float = 1.2
    b: dict[str, float] = dc_field(default_factory=dict)        # mặc định 0.75 cho mọi field
    weights: dict[str, float] = dc_field(default_factory=dict)  # mặc định 1.0 cho mọi field
    query_tf: str = "count"                                     # "count" hoặc "binary"

    def b_of(self, f: str) -> float:
        return self.b.get(f, 0.75)

    def w_of(self, f: str) -> float:
        return self.weights.get(f, 1.0)


class BM25FSearcher:
    def __init__(self, index: InvertedIndex, params: BM25FParams | None = None, tokenizer: str | None = None):
        self.index = index
        self.params = params or BM25FParams()
        self.tokenize = get_tokenizer(tokenizer or index.manifest["tokenizer"])
        self._norm = {f: 1.0 - self.params.b_of(f) + self.params.b_of(f) * index.doc_len[f] / max(index.avgdl[f], 1e-9)
                      for f in index.fields}

    def idf(self, term: str) -> float:
        df = self.index.df_any.get(term, 0)
        n = self.index.n_docs
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_terms(self, qcounts: Counter) -> tuple[np.ndarray, np.ndarray]:
        """Trả (doc_ids, scores) cho mọi tài liệu chứa ít nhất một term của truy vấn."""
        p = self.params
        ids_all, sc_all = [], []
        for term, qtf in qcounts.items():
            if term not in self.index.df_any:
                continue
            ids_parts, val_parts = [], []
            for f in self.index.fields:
                w = p.w_of(f)
                if w == 0:
                    continue
                post = self.index.postings(f, term)
                if post is None:
                    continue
                d, tf = post
                ids_parts.append(d)
                val_parts.append(w * tf / self._norm[f][d])
            if not ids_parts:
                continue
            if len(ids_parts) == 1:
                ids, tt = ids_parts[0], val_parts[0]
            else:
                ids, inv = np.unique(np.concatenate(ids_parts), return_inverse=True)
                tt = np.bincount(inv, weights=np.concatenate(val_parts))
            weight = (qtf if p.query_tf == "count" else 1) * self.idf(term)
            ids_all.append(ids)
            sc_all.append(weight * tt * (p.k1 + 1.0) / (tt + p.k1))
        if not ids_all:
            return np.zeros(0, np.int64), np.zeros(0)
        ids, inv = np.unique(np.concatenate(ids_all), return_inverse=True)
        return ids, np.bincount(inv, weights=np.concatenate(sc_all))

    def search(self, query: str, k: int = 100, ctx: dict | None = None,
               mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Trả [(premise_id, score)] theo điểm giảm dần; hòa điểm thì premise_id nhỏ trước (ổn định).
        mask: mảng bool theo premise_id; nếu có thì chỉ giữ premise được phép (ví dụ premise truy cập được)."""
        return self.search_full(query, k, mask)[0]

    def search_full(self, query: str, k: int = 100,
                    mask: np.ndarray | None = None) -> tuple[list[tuple[int, float]], np.ndarray, np.ndarray]:
        """Như search, kèm (doc_id, điểm) của mọi tài liệu đã được chấm (sắp theo doc_id) để tra điểm ứng viên khác."""
        all_ids, all_scores = self.score_terms(Counter(self.tokenize(query)))
        return self._topk(all_ids, all_scores, k, mask), all_ids, all_scores

    def score_of(self, premise_ids: np.ndarray, all_ids: np.ndarray, all_scores: np.ndarray) -> np.ndarray:
        """Điểm BM25F của các premise cho trước (0 nếu không chung term nào với truy vấn hoặc không có trong index)."""
        if not hasattr(self, "_premise_to_doc"):
            self._premise_to_doc = {int(p): d for d, p in enumerate(self.index.doc_to_premise)}
        docs = np.array([self._premise_to_doc.get(int(p), -1) for p in premise_ids], dtype=np.int64)
        if all_ids.size == 0:
            return np.zeros(len(docs))
        pos = np.clip(np.searchsorted(all_ids, docs), 0, all_ids.size - 1)
        found = (docs >= 0) & (all_ids[pos] == docs)
        return np.where(found, all_scores[pos], 0.0)

    def _topk(self, ids: np.ndarray, scores: np.ndarray, k: int, mask: np.ndarray | None) -> list[tuple[int, float]]:
        if mask is not None and ids.size:
            keep = mask[self.index.doc_to_premise[ids]]
            ids, scores = ids[keep], scores[keep]
        if ids.size == 0:
            return []
        if ids.size > k:
            top = np.argpartition(-scores, k - 1)[:k]
            ids, scores = ids[top], scores[top]
        order = np.lexsort((ids, -scores))
        pids = self.index.doc_to_premise[ids[order]]
        return [(int(pid), float(s)) for pid, s in zip(pids, scores[order])]
