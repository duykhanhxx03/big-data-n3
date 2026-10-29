"""Tạo tập truy vấn đánh giá cố định và các index con lồng nhau (benchmark_spec.md, mục 4–5).

- eval_sets/<strategy>_<split>_all.npy     : toàn bộ query_id của split (dùng cho kết quả chất lượng chính)
- eval_sets/scale_<strategy>_test_<n>.npy  : mẫu n truy vấn test (seed) dùng cho thí nghiệm quy mô
- subsets/premises_<size>.npy              : premise_id của index con; 10K ⊂ 100K ⊂ toàn bộ (181K),
                                             luôn chứa toàn bộ premise đúng của mẫu truy vấn quy mô
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SIZES = (10_000, 100_000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale-queries", type=int, default=500)
    args = ap.parse_args()
    root = Path(args.processed)
    (root / "eval_sets").mkdir(exist_ok=True)
    (root / "subsets").mkdir(exist_ok=True)

    q = pq.read_table(root / "queries.parquet", columns=["query_id", "split_strategy", "split"]).to_pandas()
    qrels = pq.read_table(root / "qrels.parquet").to_pandas()
    n_prem = pq.read_metadata(root / "premises.parquet").num_rows
    info = {"seed": args.seed, "n_premises": n_prem, "eval_sets": {}, "subsets": {}}

    for (strategy, split), g in q[q.split != "train"].groupby(["split_strategy", "split"]):
        ids = np.array(sorted(g.query_id))
        np.save(root / "eval_sets" / f"{strategy}_{split}_all.npy", ids)
        info["eval_sets"][f"{strategy}_{split}_all"] = len(ids)

    rng = np.random.default_rng(args.seed)
    test_ids = np.array(sorted(q[(q.split_strategy == "random") & (q.split == "test")].query_id))
    scale_ids = np.sort(rng.choice(test_ids, size=args.scale_queries, replace=False))
    np.save(root / "eval_sets" / f"scale_random_test_{args.scale_queries}.npy", scale_ids)
    info["eval_sets"][f"scale_random_test_{args.scale_queries}"] = len(scale_ids)

    gold = np.unique(qrels[qrels.query_id.isin(scale_ids)].premise_id.to_numpy())
    others = np.setdiff1d(np.arange(n_prem), gold)
    order = rng.permutation(others)  # thứ tự ngẫu nhiên cố định → các tập con lồng nhau
    for size in SIZES:
        need = size - gold.size
        ids = np.sort(np.concatenate([gold, order[:need]]))
        np.save(root / "subsets" / f"premises_{size // 1000}k.npy", ids)
        info["subsets"][f"{size // 1000}k"] = {"n": int(ids.size), "gold_in_subset": int(gold.size),
                                              "gold_ratio": round(gold.size / ids.size, 4)}
    info["subsets"]["full"] = {"n": n_prem, "gold_in_subset": int(gold.size), "gold_ratio": round(gold.size / n_prem, 4)}
    (root / "subsets" / "subsets_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
