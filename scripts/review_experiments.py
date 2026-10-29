"""Thí nghiệm bổ sung sau bản đánh giá báo cáo lần 1.

1. Cùng ngân sách ứng viên: V4 với 240 ứng viên BM25F (V4-K240) so với V5 (khoảng 232 đến 239 ứng viên).
2. Tách tín hiệu phụ thuộc: V4 chỉ giữ popularity, V4 chỉ giữ PageRank.
3. Tách trường namespace của BM25F: V2 với trọng số namespace bằng 0.
4. Holdout mới (scripts/make_holdout.py) cho B0, V2, V4, V4-K240, V5: xác nhận V5 trên dữ liệu chưa từng nhìn.

Dùng:  python scripts/review_experiments.py
Ra:    results/raw/eval/*.json (tag "review"), results/models/review/*.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

TRAIN = [("configs/review/v4k240.yaml", "random"), ("configs/review/v4k240.yaml", "novel_premises"),
         ("configs/review/v4_poponly.yaml", "random"), ("configs/review/v4_pronly.yaml", "random")]
EVALS = [
    ("configs/review/v4k240.yaml", ["random_val_all", "random_test_all", "novel_premises_test_all", "random_holdoutB"]),
    ("configs/review/v2_nons.yaml", ["random_val_all", "random_test_all", "novel_premises_test_all"]),
    ("configs/review/v4_poponly.yaml", ["random_val_all", "random_test_all"]),
    ("configs/review/v4_pronly.yaml", ["random_val_all", "random_test_all"]),
    ("configs/b0_bm25_default.yaml", ["random_holdoutB"]),
    ("configs/v2_bm25f_lean.yaml", ["random_holdoutB"]),
    ("configs/v4_rerank_dep.yaml", ["random_holdoutB"]),
    ("configs/v5_rerank_expand.yaml", ["random_holdoutB"]),
]


def main() -> None:
    for cfg, strategy in TRAIN:
        out = subprocess.run([PY, "scripts/train_reranker.py", "--config", cfg, "--strategy", strategy, "--workers", "8"],
                             cwd=ROOT, check=True, env=ENV, capture_output=True, text=True)
        d = json.loads(out.stdout)
        print("[train]", cfg, strategy, d["n_pairs"], d["positives"], flush=True)
    for cfg, evals in EVALS:
        for ev in evals:
            out = subprocess.run([PY, "-m", "eval.run_eval", "--config", cfg, "--eval", ev, "--tag", "review"],
                                 cwd=ROOT, check=True, env=ENV, capture_output=True, text=True)
            d = json.loads(out.stdout.strip().splitlines()[-1])
            print("[eval]", cfg, ev, {k: round(d[k], 4) for k in ("R@1", "R@10", "R@50", "MRR")}, flush=True)


if __name__ == "__main__":
    main()
