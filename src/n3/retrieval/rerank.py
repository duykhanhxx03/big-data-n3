"""Hai giai đoạn: BM25F lấy top-K ứng viên → reranker tuyến tính trên đặc trưng cấu trúc.

Config (khóa ``rerank`` trong YAML):
  candidates: 100            số ứng viên lấy từ stage 1
  accessible_filter: false   lọc stage 1 chỉ giữ premise truy cập được (qua import / khai báo trước)
  strategy: random           thống kê phụ thuộc lấy từ train của split này
  extra: {file: 0, dir: 0, popular: 0}
                             (V5) ứng viên bổ sung ngoài BM25F: premise dùng nhiều trong cùng file, cùng thư mục
                             (chỉ tính theorem train-A "có trước"), và premise phổ biến nhất; đều lọc truy cập được
  model: results/models/v4.json   trọng số do scripts/train_reranker.py tạo; bỏ trống = chỉ stage 1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from n3.retrieval.features import PremiseMeta, build_features

ROOT = Path(__file__).resolve().parents[3]
_META: dict[str, PremiseMeta] = {}


def get_meta(strategy: str) -> PremiseMeta:
    if strategy not in _META:
        _META[strategy] = PremiseMeta(ROOT / "data" / "processed", strategy)
    return _META[strategy]


class LinearModel:
    def __init__(self, path: str | Path):
        m = json.loads((ROOT / path).read_text(encoding="utf-8")) if not Path(path).is_absolute() else \
            json.loads(Path(path).read_text(encoding="utf-8"))
        self.features = tuple(m["features"])
        self.mean = np.array(m["mean"])
        self.std = np.array(m["std"])
        self.coef = np.array(m["coef"])
        self.intercept = float(m["intercept"])

    def score(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std) @ self.coef + self.intercept


class StructuralReranker:
    def __init__(self, stage1, cfg: dict):
        self.stage1 = stage1
        self.cfg = cfg
        self.k_cand = int(cfg.get("candidates", 100))
        self.filter = bool(cfg.get("accessible_filter", False))
        extra = cfg.get("extra") or {}
        self.extra = {"n_file": int(extra.get("file", 0)), "n_dir": int(extra.get("dir", 0)),
                      "n_pop": int(extra.get("popular", 0))}
        self.meta = get_meta(cfg.get("strategy", "random"))
        self.model = LinearModel(cfg["model"]) if cfg.get("model") else None
        # Với index con (10K, 100K), ứng viên bổ sung chỉ được lấy trong các premise có trong index.
        self.in_index = None
        if stage1.index.n_docs < self.meta.n:
            self.in_index = np.zeros(self.meta.n, dtype=bool)
            self.in_index[stage1.index.doc_to_premise] = True

    def candidates(self, query: str, ctx: dict | None):
        """Trả (premise_id, điểm BM25F, mặt nạ truy cập được, số ứng viên đến từ stage 1)."""
        ctx = ctx or {}
        access = None
        if self.filter or (self.model and "accessible" in self.model.features):
            access = self.meta.accessible_mask(ctx.get("file_path", ""), ctx.get("theorem_start_line", 10**9))
        hits, all_ids, all_scores = self.stage1.search_full(query, k=self.k_cand, mask=access if self.filter else None)
        cand = np.array([p for p, _ in hits], dtype=np.int64)
        s1 = np.array([s for _, s in hits], dtype=np.float64)
        n1 = len(cand)
        if any(self.extra.values()):
            allowed = access if self.filter else None
            if self.in_index is not None:
                allowed = self.in_index if allowed is None else allowed & self.in_index
            more = self.meta.extra_candidates(ctx.get("file_path", ""), ctx.get("theorem_start_line", 10**9),
                                              allowed, set(cand.tolist()), **self.extra)
            if more:
                more = np.array(more, dtype=np.int64)
                cand = np.concatenate([cand, more])
                s1 = np.concatenate([s1, self.stage1.score_of(more, all_ids, all_scores)])
        return cand, s1, access, n1

    def search(self, query: str, k: int = 100, ctx: dict | None = None, mask=None) -> list[tuple[int, float]]:
        cand, s1, access, n1 = self.candidates(query, ctx)
        if self.model is None or cand.size == 0:
            return list(zip(cand.tolist()[:k], s1.tolist()[:k]))
        X = build_features(self.meta, cand, s1, query, ctx or {}, self.model.features, access, n_stage1=n1)
        z = self.model.score(X)
        order = np.lexsort((np.arange(cand.size), -z))[:k]
        return [(int(cand[i]), float(z[i])) for i in order]
