"""Error analysis tuần 8: phân loại truy vấn thất bại, lát cắt theo chủ đề/độ dài/ký hiệu, chọn 30 failure case.

Phân loại tự động chỉ là GỢI Ý để nhóm gán nhãn thủ công (cột "Nhãn thủ công" trong failure_cases.md).

Dùng:  python scripts/error_analysis.py   (dùng lần chạy mới nhất của B0 và V4 trên random_test_all)
Đầu ra: results/figures/error_slices.md, results/figures/failure_cases.md, results/raw/error_analysis.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analysis.tokenizer import lean_tokens
from retrieval.features import PremiseMeta

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
EVAL = "random_test_all"


def latest(name: str) -> dict:
    best = None
    for p in (ROOT / "results" / "raw" / "eval").glob(f"eval-{name}-*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["parameters"]["eval_set"] == EVAL and r["parameters"].get("subset", "full") == "full" and r.get("run_file"):
            if best is None or r["timestamp"] > best["timestamp"]:
                best = r
    if best is None:
        raise SystemExit(f"chưa có lần chạy {name} trên {EVAL}")
    return best


def words(text: str) -> set[str]:
    return {t for t in lean_tokens(text) if any(c.isalpha() for c in t)}


def main() -> None:
    v4, b0 = latest("V4"), latest("B0")
    run = pd.read_csv(ROOT / v4["run_file"], sep="\t")
    ranks = {(q, p): r for q, p, r in zip(run.query_id, run.premise_id, run["rank"])}
    top = run[run["rank"] <= 3].groupby("query_id").premise_id.apply(list).to_dict()
    pq_v4 = pd.read_parquet(ROOT / v4["per_query_file"]).set_index("query_id")
    pq_b0 = pd.read_parquet(ROOT / b0["per_query_file"]).set_index("query_id")
    q = pq.read_table(P / "queries.parquet").to_pandas().set_index("query_id").loc[pq_v4.index]
    qr = pq.read_table(P / "qrels.parquet").to_pandas()
    qr = qr[qr.query_id.isin(set(q.index))]
    prem = pq.read_table(P / "premises.parquet", columns=["full_name", "statement", "decl_kind", "library"]).to_pandas()
    meta = PremiseMeta(P, "random")
    pop = pq.read_table(P / "dep_stats_random.parquet").to_pandas().popularity.to_numpy()
    pop_p99 = np.percentile(pop[pop > 0], 99)

    q["area"] = q.file_path.map(lambda f: "/".join(f.split("/")[:2]) if f.startswith("Mathlib/") else f.split("/")[0])
    q["len_tokens"] = q.state_before.map(lambda s: len(lean_tokens(s)))
    q["sym_density"] = q.state_before.map(lambda s: np.mean([not any(c.isalnum() for c in t) for t in lean_tokens(s)] or [0]))
    q["R@10_V4"], q["R@10_B0"] = pq_v4["R@10"], pq_b0["R@10"]

    # ---- từng cặp (truy vấn, premise đúng) mà V4 không đưa vào top-10 ----
    rows = []
    for qid, pid in zip(qr.query_id, qr.premise_id):
        rank = ranks.get((qid, pid), 10**6)
        if rank <= 10:
            continue
        st = q.loc[qid, "state_before"]
        overlap = len(words(st) & words(prem.full_name.iat[pid] + " " + prem.statement.iat[pid]))
        kind = prem.decl_kind.iat[pid]
        cats = []
        if rank > 100:
            cats.append("ngoai_top100")
        if overlap == 0:
            cats.append("khong_trung_tu_vung")
        if pop[pid] >= pop_p99:
            cats.append("premise_tong_quat")
        if kind in ("def", "abbrev", "instance", "structure", "class", "inductive"):
            cats.append("dinh_nghia_instance")
        if q.loc[qid, "n_gold"] >= 3:
            cats.append("nhieu_premise_dung")
        if q.loc[qid, "len_tokens"] > q.len_tokens.quantile(0.9):
            cats.append("truy_van_dai")
        rows.append({"query_id": qid, "premise_id": pid, "rank_v4": rank if rank < 10**6 else None, "overlap": overlap,
                     "gold_popularity": int(pop[pid]), "gold_kind": kind, "gold_library": prem.library.iat[pid],
                     "categories": ",".join(cats) or "khac", "area": q.loc[qid, "area"]})
    fails = pd.DataFrame(rows)
    fails.to_parquet(ROOT / "results" / "raw" / "error_analysis.parquet", index=False)

    md = ["# Error analysis (sinh bởi scripts/error_analysis.py)\n",
          f"Run V4: `{v4['run_id']}`, B0: `{b0['run_id']}`, tập {EVAL} (n = {len(q)}).",
          f"Cặp (truy vấn, premise đúng) V4 không đưa vào top-10: {len(fails)} / {len(qr)}.\n",
          "## Phân loại tự động (một cặp có thể thuộc nhiều nhóm)\n", "| Nhóm | Số cặp | Tỉ lệ |", "|---|---|---|"]
    cat_counts = fails.categories.str.split(",").explode().value_counts()
    for c, n in cat_counts.items():
        md.append(f"| {c} | {n} | {n / len(fails):.1%} |")
    md += ["\nTỉ lệ nằm ngoài top-100 (giới hạn của stage 1, reranker không cứu được): "
           f"{(fails.rank_v4.isna()).mean():.1%}.\n"]

    def slice_table(col, title, bins=None, labels=None):
        s = pd.cut(q[col], bins=bins, labels=labels) if bins is not None else q[col]
        g = q.groupby(s, observed=True).agg(n=("R@10_V4", "size"), B0=("R@10_B0", "mean"), V4=("R@10_V4", "mean"))
        g = g[g.n >= 30].sort_values("n", ascending=False)
        out = [f"\n## Lát cắt theo {title} (chỉ nhóm ≥ 30 truy vấn)\n", "| Nhóm | n | R@10 B0 | R@10 V4 | Δ |", "|---|---|---|---|---|"]
        out += [f"| {i} | {int(r.n)} | {r.B0:.3f} | {r.V4:.3f} | {r.V4 - r.B0:+.3f} |" for i, r in g.iterrows()]
        return out

    md += slice_table("area", "chủ đề (thư mục mathlib)")
    md += slice_table("len_tokens", "độ dài truy vấn (số token Lean)", [0, 20, 40, 80, 160, 10**6],
                      ["≤20", "21–40", "41–80", "81–160", ">160"])
    md += slice_table("sym_density", "mật độ ký hiệu toán trong truy vấn", [-0.01, 0.2, 0.35, 0.5, 1.0],
                      ["≤0,20", "0,20–0,35", "0,35–0,50", ">0,50"])
    md += slice_table("n_gold", "số premise đúng", [0, 1, 2, 4, 100], ["1", "2", "3–4", "≥5"])
    (ROOT / "results" / "figures" / "error_slices.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- 30 failure case, phân tầng theo nhóm chính ----
    rng = np.random.default_rng(42)
    fails["main"] = fails.categories.str.split(",").str[0]
    picks = []
    for c, g in fails.groupby("main"):
        k = max(1, round(30 * len(g) / len(fails)))
        picks.append(g.loc[rng.choice(g.index.to_numpy(), size=min(k, len(g)), replace=False)])
    sample = pd.concat(picks).head(30)
    cases = ["# 30 failure case của V4 (random_test_all)\n",
             "Cột *Nhãn thủ công* để trống: hai thành viên gán nhãn độc lập rồi đối chiếu.\n"]
    for i, r in enumerate(sample.itertuples(), 1):
        st = q.loc[r.query_id, "state_before"]
        goal = st[st.rfind("⊢"):] if "⊢" in st else st[-300:]
        cases += [f"## {i}. `{r.query_id}`",
                  f"- Theorem: `{q.loc[r.query_id, 'theorem_full_name']}` ({r.area})",
                  f"- Goal: `{goal[:300].replace(chr(10), ' ')}`",
                  f"- Premise đúng: `{prem.full_name.iat[r.premise_id]}` ({r.gold_kind}, dùng {r.gold_popularity} lần trong train-A) — "
                  f"hạng V4: {int(r.rank_v4) if pd.notna(r.rank_v4) else '>100'}",
                  f"  - `{prem.statement.iat[r.premise_id][:200].replace(chr(10), ' ')}`",
                  f"- Top-3 V4: " + ", ".join(f"`{prem.full_name.iat[p]}`" for p in top.get(r.query_id, [])),
                  f"- Gợi ý tự động: {r.categories}",
                  "- Nhãn thủ công: ", ""]
    (ROOT / "results" / "figures" / "failure_cases.md").write_text("\n".join(cases), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
