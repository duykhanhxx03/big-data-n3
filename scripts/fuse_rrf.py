"""Trộn hai (hoặc nhiều) run file bằng Reciprocal Rank Fusion và đánh giá (tuần 9: sparse + dense với Nhóm 4).

    score(d) = Σ_r  w_r / (k + rank_r(d))      (k = 60 theo Cormack và cộng sự, 2009)

Dùng:  python scripts/fuse_rrf.py --runs results/runs/<nhom3>.tsv <nhom4>.tsv --eval random_test_all [--k 60] [--weights 1,1]
Run file theo docs/benchmark_spec.md: query_id, premise_id, rank, score, system (TSV, có header).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

import runlog
from eval import metrics
from eval.run_eval import ROOT, load_eval


def fuse(runs: list[pd.DataFrame], k: int = 60, weights: list[float] | None = None, depth: int = 100) -> dict[str, list[int]]:
    weights = weights or [1.0] * len(runs)
    scores: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for run, w in zip(runs, weights):
        for qid, pid, rank in zip(run.query_id, run.premise_id, run["rank"]):
            scores[qid][int(pid)] += w / (k + rank)
    return {qid: [p for p, _ in sorted(s.items(), key=lambda x: (-x[1], x[0]))[:depth]] for qid, s in scores.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval", default="random_test_all")
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--name", default="RRF")
    args = ap.parse_args()
    runs = [pd.read_csv(p, sep="\t") for p in args.runs]
    weights = [float(x) for x in args.weights.split(",")] if args.weights else None
    q, gold = load_eval(args.eval)
    missing = [set(q.query_id) - set(r.query_id) for r in runs]
    if any(missing):
        print("Cảnh báo: số truy vấn thiếu trong từng run:", [len(m) for m in missing])
    fused = fuse(runs, args.k, weights)
    rows = [metrics.per_query(fused.get(qid, []), gold[qid]) for qid in q.query_id]
    agg = metrics.aggregate(rows)
    rec = runlog.write_record("fusion", parameters={"runs": args.runs, "eval_set": args.eval, "k": args.k,
                                                    "weights": weights, "name": args.name}, metrics=agg)
    print(json.dumps({k_: round(v, 4) for k_, v in agg.items()}), "→", Path(rec).relative_to(ROOT))


if __name__ == "__main__":
    main()
