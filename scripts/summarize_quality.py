"""Sinh bảng chất lượng và kiểm định paired bootstrap từ results/raw/eval/*.json (không chỉnh tay).

Đầu ra: results/figures/quality_<eval>.csv, results/figures/significance_<eval>.csv, results/figures/quality.md
Với mỗi (config, eval_set, subset) lấy lần chạy mới nhất.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eval.metrics import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["B0", "B0t", "V1", "V2", "V3", "V2a", "V4", "V5"]
PAIRS = [("B0", "B0t"), ("B0t", "V1"), ("V1", "V2"), ("V2", "V3"), ("V2", "V2a"), ("V2a", "V4"), ("V3", "V4"), ("B0", "V4"),
         ("V4", "V5"), ("B0", "V5")]
METRICS = ["R@1", "R@5", "R@10", "R@50", "MRR", "nDCG@10"]


def latest_runs() -> pd.DataFrame:
    rows = []
    for p in (ROOT / "results" / "raw" / "eval").glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        prm = r["parameters"]
        rows.append({"name": prm["config"]["name"], "eval_set": prm["eval_set"], "subset": prm.get("subset", "full"),
                     "timestamp": r["timestamp"], "run_id": r["run_id"], "per_query": r.get("per_query_file"),
                     **{m: r["metrics"][m] for m in METRICS}, "n_queries": r["metrics"]["n_queries"]})
    df = pd.DataFrame(rows).sort_values("timestamp")
    return df.groupby(["name", "eval_set", "subset"]).tail(1)


def main() -> None:
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    runs = latest_runs()
    runs = runs[runs.subset == "full"]
    md = ["# Chất lượng truy hồi (sinh tự động bởi scripts/summarize_quality.py)\n"]
    for ev in sorted(runs.eval_set.unique()):
        t = runs[runs.eval_set == ev].set_index("name").reindex([n for n in ORDER if n in set(runs[runs.eval_set == ev].name)])
        t[METRICS + ["n_queries", "run_id"]].to_csv(out / f"quality_{ev}.csv")
        md.append(f"\n## {ev} (n = {int(t.n_queries.iloc[0])})\n")
        md.append("| Biến thể | " + " | ".join(METRICS) + " |")
        md.append("|---" * (len(METRICS) + 1) + "|")
        for name, row in t.iterrows():
            md.append(f"| {name} | " + " | ".join(f"{row[m]:.3f}" for m in METRICS) + " |")
        sig = []
        for a, b in PAIRS:
            if a in t.index and b in t.index:
                pa = pd.read_parquet(ROOT / t.loc[a, "per_query"]).sort_values("query_id")
                pb = pd.read_parquet(ROOT / t.loc[b, "per_query"]).sort_values("query_id")
                for m in ("R@10", "MRR"):
                    res = paired_bootstrap(pa[m].to_numpy(), pb[m].to_numpy())
                    sig.append({"from": a, "to": b, "metric": m, **res})
        if sig:
            s = pd.DataFrame(sig)
            s.to_csv(out / f"significance_{ev}.csv", index=False)
            md.append(f"\nPaired bootstrap (10.000 mẫu), chênh lệch = to − from:\n")
            md.append("| Từ → đến | Metric | Δ | CI 95% | p |")
            md.append("|---|---|---|---|---|")
            for _, r in s.iterrows():
                md.append(f"| {r['from']} → {r['to']} | {r['metric']} | {r['delta']:+.3f} | "
                          f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}] | {r['p_value']:.4f} |")
    (out / "quality.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
