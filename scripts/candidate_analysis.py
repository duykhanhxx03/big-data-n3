"""Phân rã giới hạn của hệ thống hai giai đoạn: mất ở bước tạo ứng viên và mất ở bước xếp hạng.

Với truy vấn q, tập premise đúng G_q và tập ứng viên C_q trước khi xếp hạng lại:
  OracleRecall@k(q) = min(k, |G_q ∩ C_q|) / |G_q|     trần mà một reranker hoàn hảo đạt được ở top k
  CandidateLoss(q)  = 1 - |G_q ∩ C_q| / |G_q|        phần nhãn không vào được tập ứng viên
  RankingGap@k(q)   = OracleRecall@k(q) - Recall@k(q)  phần mất do xếp hạng
Lấy trung bình theo truy vấn như metric chính. Với V5, ghi thêm premise đúng trong tập ứng viên đến từ nguồn nào.
Recall@k lấy từ lần chạy đã ghim (results/figures/report_manifest.json).

Dùng:  python scripts/candidate_analysis.py
Ra:    results/figures/robustness/candidates.csv, candidate_sources.csv, và phần cuối của robustness.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.run_eval import ensure_index, for_strategy, load_config, load_eval, make_searcher, query_contexts  # noqa: E402
from retrieval.search import InvertedIndex  # noqa: E402

CONFIGS = {"V4": "configs/v4_rerank_dep.yaml", "V4-K240": "configs/review/v4k240.yaml", "V5": "configs/v5_rerank_expand.yaml"}
EVALS = ["random_test_all", "novel_premises_test_all", "random_holdoutB"]
OUT = ROOT / "results" / "figures" / "robustness"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((ROOT / "results" / "figures" / "report_manifest.json").read_text(encoding="utf-8"))
    rows, src_rows = [], []
    for ev in EVALS:
        strategy = "novel_premises" if ev.startswith("novel") else "random"
        queries, gold = load_eval(ev)
        ctxs = query_contexts(queries)
        for name, path in CONFIGS.items():
            cfg = for_strategy(load_config(ROOT / path), strategy)
            r = make_searcher({**cfg, "rerank": {**cfg["rerank"], "model": None}}, InvertedIndex(ensure_index(cfg["index"])))
            pq_ = pd.read_parquet(ROOT / man[f"{name}|{ev}"]["per_query_file"]).set_index("query_id")
            per = []
            src_count = {"BM25F": 0, "tệp": 0, "thư mục": 0, "phổ biến": 0}
            for qid, text, ctx in zip(queries.query_id, queries.state_before, ctxs):
                cand, _, access, n1 = r.candidates(text, ctx)
                g = gold[qid]
                inter = g & set(cand.tolist())
                per.append({"pool": len(cand), "cover": len(inter) / len(g),
                            "oracle10": min(10, len(inter)) / len(g), "oracle50": min(50, len(inter)) / len(g),
                            "r10": pq_.loc[qid, "R@10"], "r50": pq_.loc[qid, "R@50"]})
                if name == "V5" and inter:
                    bm = set(cand[:n1].tolist())
                    meta, ex = r.meta, r.extra
                    allowed = access if r.filter else None
                    f = set(meta.extra_candidates(ctx["file_path"], ctx["theorem_start_line"], allowed, bm,
                                                  n_file=ex["n_file"]))
                    d = set(meta.extra_candidates(ctx["file_path"], ctx["theorem_start_line"], allowed, bm | f,
                                                  n_dir=ex["n_dir"]))
                    for p in inter:
                        key = "BM25F" if p in bm else "tệp" if p in f else "thư mục" if p in d else "phổ biến"
                        src_count[key] += 1
            d = pd.DataFrame(per)
            rows.append({"tập": ev, "hệ_thống": name, "ứng_viên_TB": d.pool.mean(), "độ_phủ": d.cover.mean(),
                         "mất_ở_ứng_viên": 1 - d.cover.mean(), "Oracle@10": d.oracle10.mean(), "R@10": d.r10.mean(),
                         "mất_ở_xếp_hạng@10": (d.oracle10 - d.r10).mean(), "Oracle@50": d.oracle50.mean(),
                         "R@50": d.r50.mean(), "mất_ở_xếp_hạng@50": (d.oracle50 - d.r50).mean()})
            if name == "V5":
                tot = sum(src_count.values())
                src_rows.append({"tập": ev, **{k: v / tot for k, v in src_count.items()}, "nhãn_trong_tập_ứng_viên": tot})
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()},
                             ensure_ascii=False), flush=True)
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "candidates.csv", index=False)
    s = pd.DataFrame(src_rows)
    s.to_csv(OUT / "candidate_sources.csv", index=False)
    print(s.to_string())


if __name__ == "__main__":
    main()
