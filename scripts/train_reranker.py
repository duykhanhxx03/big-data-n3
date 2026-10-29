"""Huấn luyện reranker tuyến tính (logistic regression) trên nửa B của train.

Dùng:  python scripts/train_reranker.py --config configs/v4_rerank_dep.yaml [--features a,b,c] [--out results/models/x.json]
Stage 1 và bộ lọc lấy đúng như trong config, để phân phối ứng viên lúc huấn luyện giống lúc chạy.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

from eval.run_eval import PROCESSED, ROOT, ensure_index, for_strategy, load_config, make_searcher, query_contexts
from retrieval.features import ALL_FEATURES, build_features
from retrieval.search import InvertedIndex
from splits import train_half

_S: dict = {}


def _init(cfg: dict, features: tuple[str, ...]):
    stage_cfg = {**cfg, "rerank": {**cfg["rerank"], "model": None}}
    _S["r"] = make_searcher(stage_cfg, InvertedIndex(ensure_index(cfg["index"])))
    _S["features"] = features


def _one(item):
    text, ctx, gold = item
    r = _S["r"]
    cand, s1, access, n1 = r.candidates(text, ctx)
    if cand.size == 0:
        return None
    X = build_features(r.meta, cand, s1, text, ctx, _S["features"], access, n_stage1=n1)
    y = np.isin(cand, list(gold)).astype(np.int8)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", default=None, help="danh sách đặc trưng, mặc định lấy từ config rerank.features")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strategy", default="random", choices=["random", "novel_premises"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    cfg = for_strategy(load_config(args.config), args.strategy)
    feats = tuple((args.features or ",".join(cfg["rerank"].get("features", ALL_FEATURES))).split(","))
    unknown = set(feats) - set(ALL_FEATURES)
    if unknown:
        raise SystemExit(f"đặc trưng không tồn tại: {unknown}")
    strategy = cfg["rerank"].get("strategy", "random")

    q = pq.read_table(PROCESSED / "queries.parquet", columns=["query_id", "split_strategy", "split", "state_before",
                                                              "theorem_full_name", "file_path", "theorem_start_line"]).to_pandas()
    q = q[(q.split_strategy == strategy) & (q.split == "train")]
    q = q[q.theorem_full_name.map(train_half) == "B"]
    q = q.sample(n=min(args.n_train, len(q)), random_state=args.seed).sort_values("query_id")
    qr = pq.read_table(PROCESSED / "qrels.parquet").to_pandas()
    gold = qr[qr.query_id.isin(set(q.query_id))].groupby("query_id").premise_id.apply(set).to_dict()
    items = [(t, c, gold[qid]) for qid, t, c in zip(q.query_id, q.state_before, query_contexts(q))]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(cfg, feats)) as ex:
        res = [r for r in ex.map(_one, items, chunksize=50) if r is not None]
    X = np.vstack([r[0] for r in res])
    y = np.concatenate([r[1] for r in res])
    t_feat = time.perf_counter() - t0
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    clf = LogisticRegression(C=args.C, max_iter=2000)
    clf.fit((X - mean) / std, y)
    model = {
        "type": "logistic_regression", "features": list(feats), "mean": mean.tolist(), "std": std.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "C": args.C,
        "train": {"strategy": strategy, "half": "B", "n_queries": len(items), "n_pairs": int(len(y)),
                  "positives": int(y.sum()), "candidates": cfg["rerank"].get("candidates", 100),
                  "accessible_filter": cfg["rerank"].get("accessible_filter", False), "seed": args.seed,
                  "extra": cfg["rerank"].get("extra"),
                  "stage1_config": cfg["name"], "feature_time_s": round(t_feat, 1)},
    }
    out = Path(args.out or cfg["rerank"]["model"])
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), **model["train"],
                      "coef": dict(zip(feats, np.round(clf.coef_[0], 3).tolist()))}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
