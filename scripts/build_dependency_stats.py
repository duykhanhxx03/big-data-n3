"""Thống kê phụ thuộc premise từ tập train: độ phổ biến, PageRank và bảng sử dụng theo file (nối Chương 5).

Để tránh rò rỉ khi huấn luyện reranker, theorem của train được chia đôi cố định theo băm tên:
- nửa A: tính thống kê (popularity, PageRank) → đặc trưng cho reranker
- nửa B: dùng để huấn luyện reranker (scripts/train_reranker.py)
Val/test không bao giờ tham gia.

Đồ thị: nút là premise; cạnh T → P nếu theorem T (bản thân là một premise trong corpus)
dùng P trong một tactic. PageRank hệ số tắt dần 0,85, nút cụt phân phối đều.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from splits import train_half


def pagerank(src: np.ndarray, dst: np.ndarray, n: int, d: float = 0.85, iters: int = 100, tol: float = 1e-10):
    outdeg = np.bincount(src, minlength=n).astype(np.float64)
    dangling = outdeg == 0
    x = np.full(n, 1.0 / n)
    for it in range(iters):
        contrib = x[src] / outdeg[src]
        new = (1 - d) / n + d * (np.bincount(dst, weights=contrib, minlength=n) + x[dangling].sum() / n)
        if np.abs(new - x).sum() < tol:
            return new, it + 1
        x = new
    return x, iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    args = ap.parse_args()
    root = Path(args.processed)
    prem = pq.read_table(root / "premises.parquet", columns=["premise_id", "full_name", "path"]).to_pandas()
    n = len(prem)
    pid_by_key = {(f, p): i for i, f, p in zip(prem.premise_id, prem.full_name, prem.path)}
    q = pq.read_table(root / "queries.parquet", columns=["query_id", "split_strategy", "split", "theorem_full_name",
                                                         "file_path", "theorem_start_line"]).to_pandas()
    q = q[q.split == "train"].copy()
    q["half"] = q.theorem_full_name.map(train_half)
    qr = pq.read_table(root / "qrels.parquet").to_pandas()
    info = {}
    for strategy in ("random", "novel_premises"):
        qa = q[(q.split_strategy == strategy) & (q.half == "A")]
        edges = qr[qr.query_id.isin(set(qa.query_id))].merge(
            qa[["query_id", "theorem_full_name", "file_path", "theorem_start_line"]], on="query_id")
        # popularity = số theorem khác nhau (nửa A) dùng premise
        uses = edges.drop_duplicates(["theorem_full_name", "premise_id"])
        pop = np.bincount(uses.premise_id.to_numpy(), minlength=n)
        src_ids = [pid_by_key.get((t, f)) for t, f in zip(uses.theorem_full_name, uses.file_path)]
        mask = np.array([s is not None for s in src_ids])
        src = np.array([s for s in src_ids if s is not None], dtype=np.int64)
        dst = uses.premise_id.to_numpy()[mask]
        pr, it = pagerank(src, dst, n)
        out = pa.table({"premise_id": np.arange(n, dtype=np.int32), "popularity": pop.astype(np.int32),
                        "pagerank": pr.astype(np.float64)})
        pq.write_table(out, root / f"dep_stats_{strategy}.parquet")
        # Ai dùng gì, ở đâu: nguồn ứng viên và đặc trưng file_usage/dir_usage của V5 (chỉ đếm theorem "có trước")
        pq.write_table(pa.table({"file_path": uses.file_path.to_numpy(), "line": uses.theorem_start_line.to_numpy(),
                                 "theorem": uses.theorem_full_name.to_numpy(),
                                 "premise_id": uses.premise_id.to_numpy().astype(np.int32)}),
                       root / f"dep_usage_{strategy}.parquet")
        info[strategy] = {"train_A_queries": int(len(qa)), "theorems_A": int(qa.theorem_full_name.nunique()),
                          "edges": int(len(dst)), "theorem_nodes_in_corpus": int(mask.sum()),
                          "premises_with_use": int((pop > 0).sum()), "pagerank_iters": it}
    (root / "dep_stats_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
