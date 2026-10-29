"""Tập holdout mới để xác nhận V5 sau khi V5 được thiết kế từ phân tích lỗi trên tập test.

Lấy từ train nửa B của split random: nửa B không dùng để tính thống kê phụ thuộc (nửa A), và chỉ 6.000 truy vấn của
nửa B được dùng huấn luyện reranker. Holdout loại mọi theorem có ít nhất một truy vấn trong mẫu huấn luyện đó (lấy mẫu
đúng như scripts/train_reranker.py, seed 42), rồi chọn ngẫu nhiên theo theorem (seed 42) cho tới khi đủ khoảng 2.500 truy
vấn. Không bước thiết kế hay chọn cấu hình nào đã nhìn tập này.

Dùng:  python scripts/make_holdout.py
Ra:    data/processed/eval_sets/random_holdoutB.npy, data/processed/eval_sets/random_holdoutB.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from splits import train_half  # noqa: E402

P = ROOT / "data" / "processed"
TARGET, SEED, N_TRAIN = 2500, 42, 6000


def main() -> None:
    cols = ["query_id", "split_strategy", "split", "state_before", "theorem_full_name", "file_path", "theorem_start_line"]
    q = pq.read_table(P / "queries.parquet", columns=cols).to_pandas()
    q = q[(q.split_strategy == "random") & (q.split == "train")]
    q = q[q.theorem_full_name.map(train_half) == "B"]
    train_sample = q.sample(n=min(N_TRAIN, len(q)), random_state=SEED)   # đúng như train_reranker.py
    used = set(zip(train_sample.file_path, train_sample.theorem_full_name))
    rest = q[[k not in used for k in zip(q.file_path, q.theorem_full_name)]]
    theorems = rest[["file_path", "theorem_full_name"]].drop_duplicates()
    order = np.random.default_rng(SEED).permutation(len(theorems))
    by_thm = rest.groupby(["file_path", "theorem_full_name"]).query_id.apply(list)
    chosen, n = [], 0
    for i in order:
        key = tuple(theorems.iloc[i])
        chosen.extend(by_thm[key])
        n += 1
        if len(chosen) >= TARGET:
            break
    ids = np.array(sorted(chosen))
    np.save(P / "eval_sets" / "random_holdoutB.npy", ids)
    info = {"source": "random/train, nửa B", "exclude": f"theorem có truy vấn trong mẫu huấn luyện reranker ({N_TRAIN}, seed {SEED})",
            "n_queries": int(len(ids)), "n_theorems": n, "seed": SEED,
            "theorems_in_half_B": int(len(q[["file_path", "theorem_full_name"]].drop_duplicates())),
            "theorems_excluded": len(used)}
    (P / "eval_sets" / "random_holdoutB.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
