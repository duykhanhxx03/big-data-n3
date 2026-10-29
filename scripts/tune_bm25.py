"""Quét tham số BM25/BM25F trên tập val (không bao giờ dùng test để tinh chỉnh).

Dùng:  python scripts/tune_bm25.py --index-config configs/v2_bm25f_lean.yaml --grid configs/grid_v2.yaml
Mỗi cấu hình chạy trong một tiến trình riêng; kết quả ghi results/raw/tuning/<tên>.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from eval import metrics
from eval.run_eval import ROOT, ensure_index, load_config, load_eval
from retrieval.search import BM25FParams, BM25FSearcher, InvertedIndex

_STATE: dict = {}


def _init(index_path: str, eval_name: str):
    _STATE["idx"] = InvertedIndex(index_path, cache_terms=20000)
    _STATE["q"], _STATE["gold"] = load_eval(eval_name)


def _eval(params: dict) -> dict:
    idx = _STATE["idx"]
    b = params["b"] if isinstance(params["b"], dict) else {f: params["b"] for f in idx.fields}
    s = BM25FSearcher(idx, BM25FParams(k1=params["k1"], b=b, weights=params.get("weights", {}),
                                       query_tf=params.get("query_tf", "count")))
    rows = [metrics.per_query([p for p, _ in s.search(t, k=100)], _STATE["gold"][qid])
            for qid, t in zip(_STATE["q"].query_id, _STATE["q"].state_before)]
    return {**{k: json.dumps(v) if isinstance(v, dict) else v for k, v in params.items()}, **metrics.aggregate(rows)}


def expand(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-config", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--eval", default="random_val_all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    cfg = load_config(args.index_config)
    grid = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))
    combos = expand(grid)
    path = ensure_index(cfg["index"])
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(path), args.eval)) as ex:
        res = list(ex.map(_eval, combos))
    df = pd.DataFrame(res).sort_values("R@10", ascending=False)
    out = ROOT / "results" / "raw" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{cfg['name']}__{Path(args.grid).stem}__{args.eval}.csv"
    df.to_csv(out / name, index=False)
    print(df.head(12).to_string(index=False))
    print(f"{len(combos)} cấu hình, {time.perf_counter() - t0:.0f}s → {out / name}")


if __name__ == "__main__":
    main()
