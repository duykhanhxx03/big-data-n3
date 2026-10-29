"""CLI tìm premise.

Dùng:  python -m serving.cli "n m : ℕ ⊢ n + m = m + n" -k 10 --mode V4
       python -m serving.cli --query-id "random:test:12:0" --mode V4   (lấy proof state và ngữ cảnh từ tập dữ liệu)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

from eval.run_eval import PROCESSED, ROOT, ensure_index, load_config, make_searcher, query_contexts
from retrieval.search import InvertedIndex

MODES = {"B0": "b0_bm25_default", "B0t": "b0t_bm25_tuned", "V1": "v1_bm25_lean", "V2": "v2_bm25f_lean",
         "V2a": "v2a_bm25f_accessible", "V3": "v3_rerank_struct", "V4": "v4_rerank_dep"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default=None)
    ap.add_argument("--query-id", default=None)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--mode", default="V4", choices=sorted(MODES))
    ap.add_argument("--file-path", default="")
    ap.add_argument("--theorem", default="")
    ap.add_argument("--line", type=int, default=10**9)
    args = ap.parse_args()
    ctx = {"theorem_full_name": args.theorem, "file_path": args.file_path, "theorem_start_line": args.line}
    gold: set[int] = set()
    text = args.query
    if args.query_id:
        q = pq.read_table(PROCESSED / "queries.parquet").to_pandas()
        q = q[q.query_id == args.query_id]
        if q.empty:
            raise SystemExit(f"không có query_id {args.query_id}")
        text, ctx = q.state_before.iat[0], query_contexts(q)[0]
        qr = pq.read_table(PROCESSED / "qrels.parquet").to_pandas()
        gold = set(qr[qr.query_id == args.query_id].premise_id)
        print(f"Theorem: {ctx['theorem_full_name']}  ({ctx['file_path']}:{ctx['theorem_start_line']})")
    if not text:
        raise SystemExit("cần proof state hoặc --query-id")
    cfg = load_config(ROOT / "configs" / f"{MODES[args.mode]}.yaml")
    searcher = make_searcher(cfg, InvertedIndex(ensure_index(cfg["index"])))
    t = time.perf_counter()
    hits = searcher.search(text, k=args.k, ctx=ctx)
    ms = (time.perf_counter() - t) * 1000
    prem = pq.read_table(PROCESSED / "premises.parquet", columns=["full_name", "statement"]).to_pandas()
    print(f"Proof state:\n{text}\n\nTop {args.k} ({args.mode}, {ms:.1f} ms):")
    for i, (pid, score) in enumerate(hits, 1):
        mark = "  ✓ đúng" if pid in gold else ""
        print(f"{i:>3}. {prem.full_name.iat[pid]}  [{score:.3f}]{mark}")
        print(f"     {prem.statement.iat[pid][:160]}")
    if gold:
        print("\nPremise đúng:", ", ".join(prem.full_name.iat[g] for g in sorted(gold)))


if __name__ == "__main__":
    main()
