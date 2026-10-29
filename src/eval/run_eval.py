"""Chạy một thí nghiệm truy hồi từ file config: build index (nếu chưa có) → truy vấn → metric → run record.

Dùng:  python -m eval.run_eval --config configs/b0_bm25_default.yaml --eval random_test_all [--subset 10k]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

import runlog
from eval import metrics
from index import build_seq
from retrieval.search import BM25FParams, BM25FSearcher, InvertedIndex

ROOT = runlog.ROOT
PROCESSED = ROOT / "data" / "processed"


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("name", Path(path).stem)
    return cfg


def index_dir(icfg: dict, subset: str) -> Path:
    name = f"{icfg['tokenizer']}__{'+'.join(icfg['fields'])}__{icfg.get('codec', 'vb')}__{subset}"
    return ROOT / "indexes" / name


def ensure_index(icfg: dict, subset: str = "full") -> Path:
    out = index_dir(icfg, subset)
    if (out / "index_manifest.json").exists():
        return out
    limit = None if subset == "full" else np.load(PROCESSED / "subsets" / f"premises_{subset}.npy")
    docs, n, source = build_seq.iter_premises(PROCESSED / "premises.parquet", icfg["fields"], limit)
    pids = source.pop("premise_ids")
    source["subset"] = subset
    build_seq.build(docs, n, out, fields=icfg["fields"], tokenizer=icfg["tokenizer"], codec=icfg.get("codec", "vb"),
                    source=source)
    import pyarrow as pa
    pq.write_table(pa.table({"doc_id": np.arange(n, dtype=np.int32), "premise_id": pids.astype(np.int32)}),
                   out / "doc_map.parquet")
    return out


def load_eval(eval_name: str) -> tuple[pd.DataFrame, dict[str, set[int]]]:
    ids = np.load(PROCESSED / "eval_sets" / f"{eval_name}.npy", allow_pickle=False)
    q = pq.read_table(PROCESSED / "queries.parquet", columns=["query_id", "state_before", "theorem_full_name",
                                                                "file_path", "theorem_start_line", "n_gold"]).to_pandas()
    q = q[q.query_id.isin(set(ids))].sort_values("query_id").reset_index(drop=True)
    qr = pq.read_table(PROCESSED / "qrels.parquet").to_pandas()
    qr = qr[qr.query_id.isin(set(ids))]
    gold = qr.groupby("query_id").premise_id.apply(set).to_dict()
    return q, gold


def query_contexts(q: pd.DataFrame) -> list[dict]:
    """Thông tin có sẵn lúc truy vấn ngoài proof state: theorem đang chứng minh, file và vị trí."""
    return [{"theorem_full_name": t, "file_path": f, "theorem_start_line": int(l)}
            for t, f, l in zip(q.theorem_full_name, q.file_path, q.theorem_start_line)]


def make_searcher(cfg: dict, idx: InvertedIndex):
    s = cfg.get("search", {})
    params = BM25FParams(k1=s.get("k1", 1.2), b=s.get("b", {}), weights=s.get("weights", {}),
                         query_tf=s.get("query_tf", "count"))
    if isinstance(params.b, (int, float)):
        params.b = {f: float(params.b) for f in idx.fields}
    searcher = BM25FSearcher(idx, params)
    if cfg.get("rerank"):
        from retrieval.rerank import StructuralReranker
        return StructuralReranker(searcher, cfg["rerank"])
    return searcher


def for_strategy(cfg: dict, strategy: str) -> dict:
    """Thống kê phụ thuộc và reranker phải lấy từ train của cùng split_strategy với tập đánh giá:
    train của random chứa theorem thuộc test của novel_premises nên dùng chéo sẽ rò rỉ."""
    if not cfg.get("rerank") or strategy == "random":
        return cfg
    rr = dict(cfg["rerank"], strategy=strategy)
    if rr.get("model"):
        rr["model"] = rr["model"].replace(".json", f"_{strategy}.json")
    return {**cfg, "rerank": rr}


def run(cfg: dict, eval_name: str, subset: str = "full", k: int = 100, write_run_file: bool = True,
        tag: str | None = None) -> dict:
    cfg = for_strategy(cfg, "novel_premises" if eval_name.startswith("novel_premises") else "random")
    t0 = time.perf_counter()
    path = ensure_index(cfg["index"], subset)
    build_s = time.perf_counter() - t0
    idx = InvertedIndex(path)
    searcher = make_searcher(cfg, idx)
    queries, gold = load_eval(eval_name)
    run_id = runlog.new_run_id(f"eval-{cfg['name']}")
    rows, lat, run_lines = [], [], []
    for qid, text, ctx in zip(queries.query_id, queries.state_before, query_contexts(queries)):
        t = time.perf_counter()
        hits = searcher.search(text, k=k, ctx=ctx)
        lat.append((time.perf_counter() - t) * 1000)
        ranked = [pid for pid, _ in hits]
        r = metrics.per_query(ranked, gold[qid])
        r["query_id"] = qid
        rows.append(r)
        if write_run_file:
            run_lines.extend(f"{qid}\t{pid}\t{rank}\t{score:.6f}\t{cfg['name']}" for rank, (pid, score) in enumerate(hits, 1))
    per_q = pd.DataFrame(rows)
    agg = metrics.aggregate([{k_: v for k_, v in r.items() if k_ != "query_id"} for r in rows])
    lat_arr = np.array(lat)
    agg.update({"latency_ms_p50": float(np.percentile(lat_arr, 50)), "latency_ms_p95": float(np.percentile(lat_arr, 95)),
                "latency_ms_mean": float(lat_arr.mean()), "qps_sequential": float(1000 / lat_arr.mean()),
                "n_queries": len(rows)})
    out_dir = ROOT / "results" / "raw" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_q["latency_ms"] = lat_arr
    per_q.to_parquet(out_dir / f"{run_id}.perquery.parquet", index=False)
    run_file = None
    if write_run_file:
        run_dir = ROOT / "results" / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_file = run_dir / f"{run_id}.tsv"
        run_file.write_text("query_id\tpremise_id\trank\tscore\tsystem\n" + "\n".join(run_lines) + "\n", encoding="utf-8")
    man = idx.manifest
    rec = runlog.write_record(
        "eval", parameters={"config": cfg, "eval_set": eval_name, "subset": subset, "k": k, "tag": tag},
        metrics=agg, seed=42, run_id=run_id,
        extra={"index": {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "n_docs": man["n_docs"],
                         "n_postings": man["n_postings"], "postings_bytes": man["postings_bytes"],
                         "build": man["build"], "built_this_run_s": round(build_s, 2)},
               "per_query_file": f"results/raw/eval/{run_id}.perquery.parquet",
               "run_file": str(run_file.relative_to(ROOT)).replace("\\", "/") if run_file else None})
    return {"run_id": run_id, "record": str(rec), **agg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval", default="random_val_all")
    ap.add_argument("--subset", default="full")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    res = run(load_config(args.config), args.eval, args.subset, args.k, tag=args.tag)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
