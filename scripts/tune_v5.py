"""Chọn số ứng viên bổ sung của V5 trên random_val_all (không chạm tập test).

Với mỗi tổ hợp (file, dir, popular): huấn luyện lại reranker trên train nửa B, đánh giá trên val, và đo độ phủ của
tập ứng viên (tỉ lệ premise đúng nằm trong tập ứng viên trước khi xếp hạng lại), là trần recall của reranker.
Tổ hợp (0, 0, 0) giữ nguyên ứng viên của V4 nhưng thêm hai đặc trưng mới, để tách tác dụng của đặc trưng và của ứng viên.

Dùng:  python scripts/tune_v5.py [--workers 8]
Ra:    results/raw/tuning/V5__extra__random_val_all.csv, configs/v5_rerank_expand.yaml cập nhật khóa rerank.extra
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
GRID = [(0, 0, 0), (50, 0, 0), (0, 50, 0), (0, 0, 50), (50, 50, 0), (50, 50, 50), (100, 100, 50), (100, 100, 100)]
BASE = ROOT / "configs" / "v5_rerank_expand.yaml"


def pool_recall(cfg: dict, eval_name: str) -> tuple[float, float]:
    """Độ phủ trung bình theo truy vấn của tập ứng viên, và số ứng viên trung bình."""
    from eval.run_eval import load_eval, make_searcher, ensure_index, query_contexts
    from retrieval.search import InvertedIndex
    stage_cfg = {**cfg, "rerank": {**cfg["rerank"], "model": None}}
    r = make_searcher(stage_cfg, InvertedIndex(ensure_index(cfg["index"])))
    q, gold = load_eval(eval_name)
    rec, size = [], []
    for qid, text, ctx in zip(q.query_id, q.state_before, query_contexts(q)):
        cand = set(r.candidates(text, ctx)[0].tolist())
        g = gold[qid]
        rec.append(len(g & cand) / len(g))
        size.append(len(cand))
    return float(np.mean(rec)), float(np.mean(size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval", default="random_val_all")
    args = ap.parse_args()
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    out_dir = ROOT / "configs" / "v5_grid"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for f, d, p in GRID:
        cfg = copy.deepcopy(base)
        cfg["name"] = f"V5-f{f}-d{d}-p{p}"
        cfg["rerank"]["extra"] = {"file": f, "dir": d, "popular": p}
        cfg["rerank"]["model"] = f"results/models/v5_grid/{cfg['name']}.json"
        path = out_dir / f"{cfg['name']}.yaml"
        path.write_text("# Sinh bởi scripts/tune_v5.py\n" + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
        subprocess.run([PY, "scripts/train_reranker.py", "--config", str(path), "--workers", str(args.workers)],
                       cwd=ROOT, check=True, env=ENV, stdout=subprocess.DEVNULL)
        out = subprocess.run([PY, "-m", "eval.run_eval", "--config", str(path), "--eval", args.eval, "--tag", "v5-tune"],
                             cwd=ROOT, check=True, env=ENV, capture_output=True, text=True)
        res = json.loads(out.stdout.strip().splitlines()[-1])
        res.update(json.loads(Path(res["record"]).read_text(encoding="utf-8"))["metrics"])
        cov, size = pool_recall(cfg, args.eval)
        row = {"file": f, "dir": d, "popular": p, "R@1": res["R@1"], "R@10": res["R@10"], "R@50": res["R@50"],
               "MRR": res["MRR"], "pool_recall": cov, "pool_size": size, "p95_ms": res["latency_ms_p95"]}
        rows.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "raw" / "tuning" / f"V5__extra__{args.eval}.csv"
    df.to_csv(out, index=False)
    best = df.sort_values(["R@10", "MRR"], ascending=False).iloc[0]
    base["rerank"]["extra"] = {"file": int(best["file"]), "dir": int(best["dir"]), "popular": int(best["popular"])}
    text = BASE.read_text(encoding="utf-8")
    head = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
    BASE.write_text(head + yaml.safe_dump(base, allow_unicode=True, sort_keys=False, default_flow_style=None),
                    encoding="utf-8")
    print("chọn:", base["rerank"]["extra"], "| bảng:", out)


if __name__ == "__main__":
    main()
