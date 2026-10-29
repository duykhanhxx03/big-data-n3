"""Phân tích độ bền của kết luận từ các lần chạy đã ghim (results/figures/report_manifest.json), không chạy lại truy hồi.

1. Luồng lọc dữ liệu: theorem gốc, tactic, tactic có premise, truy vấn giữ lại, theorem còn truy vấn, proof state khác nhau.
2. Trung bình theo truy vấn (query-macro, metric chính) và theo theorem (theorem-macro, kiểm tra độ bền).
3. Khoảng tin cậy: paired bootstrap theo cụm theorem (lấy mẫu theorem, mang theo các truy vấn của nó), 10.000 mẫu.
   Giá trị p: kiểm định hoán vị ghép cặp theo cụm theorem (đổi nhãn hai hệ thống cho mọi truy vấn của một theorem),
   10.000 phép hoán vị, p hai phía có hiệu chỉnh Monte Carlo. Họ so sánh chính là 7 bước của chuỗi cải tiến
   B0 → B0t → V1 → V2 → V3, V2 → V2a → V4 → V5, hiệu chỉnh Holm trong từng tập, từng metric và từng cách lấy trung
   bình. Các đối chứng là phân tích phụ, báo p chưa hiệu chỉnh.
4. Premise đã thấy và chưa thấy: một premise đúng là "chưa thấy" nếu không là nhãn của truy vấn train nào cùng split.

Dùng:  python scripts/robustness.py
Ra:    results/figures/robustness/*.csv và robustness.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.metrics import holm, paired_cluster_bootstrap, paired_cluster_permutation  # noqa: E402

P = ROOT / "data" / "processed"
OUT = ROOT / "results" / "figures" / "robustness"
EVALS = ["random_test_all", "novel_premises_test_all", "random_holdoutB"]
CHAIN = [("B0", "B0t"), ("B0t", "V1"), ("V1", "V2"), ("V2", "V3"), ("V2", "V2a"), ("V2a", "V4"), ("V4", "V5")]
EXTRA = [("B0", "V4"), ("B0", "V5"), ("V4", "V4-K240"), ("V4-K240", "V5"), ("V2-nons", "V2"), ("V4-poponly", "V4"),
         ("V4-pronly", "V4")]
SYSTEMS = ["B0", "B0t", "V1", "V2", "V2-nons", "V3", "V2a", "V4", "V4-poponly", "V4-pronly", "V4-K240", "V5"]


def md_table(df: pd.DataFrame) -> str:
    def cell(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def fmt_p(p: float) -> str:
    return f"{p:.4f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((ROOT / "results" / "figures" / "report_manifest.json").read_text(encoding="utf-8"))
    q = pq.read_table(P / "queries.parquet", columns=["query_id", "split_strategy", "split", "theorem_full_name",
                                                       "file_path", "state_before"]).to_pandas()
    q["theorem"] = q.file_path + "::" + q.theorem_full_name
    qr = pq.read_table(P / "qrels.parquet").to_pandas()
    md = ["# Phân tích độ bền (sinh bởi scripts/robustness.py từ results/figures/report_manifest.json)", ""]

    # ---- 1. luồng lọc dữ liệu
    stats = json.loads((P / "extract_stats.json").read_text(encoding="utf-8"))["splits"]
    raw = ROOT / "data" / "raw" / "leandojo_benchmark_4_v10" / "leandojo_benchmark_4"
    rows = []
    for strategy in ("random", "novel_premises"):
        for split in ("train", "val", "test"):
            s = stats[f"{strategy}/{split}"]
            x = q[(q.split_strategy == strategy) & (q.split == split)]
            data = json.loads((raw / strategy / f"{split}.json").read_text(encoding="utf-8"))
            with_tac = sum(1 for t in data if t["traced_tactics"])
            with_prem = sum(1 for t in data if any(a["annotated_tactic"][1] for a in t["traced_tactics"]))
            del data
            rows.append({"split": strategy, "tập": split, "theorem_gốc": s["theorems"], "theorem_có_tactic": with_tac,
                         "theorem_có_tactic_có_premise": with_prem, "tactic": s["tactics"],
                         "tactic_có_premise": s["tactics"] - s.get("tactics_without_premise", 0),
                         "nhãn_không_có_trong_corpus": s.get("gold_not_in_corpus", 0),
                         "truy_vấn_bị_loại": s.get("queries_dropped_no_gold", 0), "truy_vấn_giữ_lại": len(x),
                         "theorem_còn_truy_vấn": x.theorem.nunique(),
                         "tỉ_lệ_theorem_giữ_lại": round(x.theorem.nunique() / s["theorems"], 3),
                         "proof_state_khác_nhau": x.state_before.nunique()})
    flow = pd.DataFrame(rows)
    flow.to_csv(OUT / "data_flow.csv", index=False)
    total_unique = q.drop_duplicates(["split_strategy", "state_before"]).groupby("split_strategy").size().to_dict()
    md += ["## 1. Luồng lọc dữ liệu", "", md_table(flow), "",
           f"Tổng số lượt truy vấn của hai split là {len(q):,}; số proof state khác nhau trong từng split: {total_unique}.", ""]

    # ---- 2 và 3. query-macro, theorem-macro, kiểm định theo cụm
    def per_query(system: str, ev: str) -> pd.DataFrame | None:
        e = man.get(f"{system}|{ev}")
        if not e:
            return None
        d = pd.read_parquet(ROOT / e["per_query_file"])
        return d.merge(q[["query_id", "theorem"]], on="query_id", validate="one_to_one").set_index("query_id").sort_index()

    macro, sig = [], []
    for ev in EVALS:
        frames = {s: per_query(s, ev) for s in SYSTEMS}
        frames = {s: d for s, d in frames.items() if d is not None}
        for s, d in frames.items():
            th = d.groupby("theorem")[["R@10", "MRR"]].mean()
            macro.append({"tập": ev, "hệ_thống": s, "n_truy_vấn": len(d), "n_theorem": len(th),
                          "R@10_query": d["R@10"].mean(), "R@10_theorem": th["R@10"].mean(),
                          "MRR_query": d["MRR"].mean(), "MRR_theorem": th["MRR"].mean()})
        for family, pairs in (("chính", CHAIN), ("phụ", EXTRA)):
            for a, b in pairs:
                if a not in frames or b not in frames:
                    continue
                da, db = frames[a], frames[b]
                assert da.index.equals(db.index)
                for metric in ("R@10", "MRR"):
                    for est in ("query", "cluster"):
                        args = (da[metric].to_numpy(), db[metric].to_numpy(), da.theorem.to_numpy())
                        r = paired_cluster_bootstrap(*args, estimand=est)
                        r["p"] = paired_cluster_permutation(*args, estimand=est)["p"]
                        sig.append({"tập": ev, "họ": family, "từ": a, "đến": b, "metric": metric,
                                    "trung_bình": "truy vấn" if est == "query" else "theorem", **r})
    macro = pd.DataFrame(macro)
    macro.to_csv(OUT / "macro.csv", index=False)
    sig = pd.DataFrame(sig)
    sig["p_holm"] = np.nan
    for (ev, metric, est), g in sig[(sig["họ"] == "chính")].groupby(["tập", "metric", "trung_bình"]):
        sig.loc[g.index, "p_holm"] = holm(g.p.tolist())
    sig.to_csv(OUT / "significance_cluster.csv", index=False)

    md += ["## 2. Trung bình theo truy vấn và theo theorem", "",
           "| Tập | Hệ thống | Truy vấn | Theorem | R@10 (truy vấn) | R@10 (theorem) | MRR (truy vấn) | MRR (theorem) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in macro.itertuples():
        md.append(f"| {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.4f} | {r[8]:.4f} |")
    md += ["", "## 3. Kiểm định bootstrap theo cụm theorem (10.000 mẫu)", "",
           "KTC từ bootstrap theo cụm; p hai phía từ kiểm định hoán vị theo cụm. Họ chính: Holm trong từng tập, metric và cách lấy "
           "trung bình. Họ phụ: không hiệu chỉnh.",
           "", "| Tập | Họ | So sánh | Metric | Trung bình | Δ | KTC 95% | p | p Holm |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in sig.itertuples():
        md.append(f"| {r.tập} | {r.họ} | {r.từ} → {r.đến} | {r.metric} | {r.trung_bình} | {r.delta:+.4f} | "
                  f"[{r.ci95_low:+.4f}; {r.ci95_high:+.4f}] | {fmt_p(r.p)} | "
                  f"{'' if np.isnan(r.p_holm) else fmt_p(r.p_holm)} |")

    # ---- 4. premise đã thấy và chưa thấy
    seen_rows, unseen_sig = [], []
    for ev, strategy in (("random_test_all", "random"), ("novel_premises_test_all", "novel_premises")):
        train_q = set(q[(q.split_strategy == strategy) & (q.split == "train")].query_id)
        seen = set(qr[qr.query_id.isin(train_q)].premise_id)
        test = q[(q.split_strategy == strategy) & (q.split == ev.replace(f"{strategy}_", "").replace("_all", ""))]
        g = qr[qr.query_id.isin(set(test.query_id))]
        gold = g.groupby("query_id").premise_id.apply(set).to_dict()
        unseen = {k: {p for p in v if p not in seen} for k, v in gold.items()}
        seenset = {k: {p for p in v if p in seen} for k, v in gold.items()}
        q_u = [k for k, v in unseen.items() if v]
        thm = test.set_index("query_id").theorem
        info = {"tập": ev, "cặp_nhãn": len(g), "cặp_chưa_thấy": int(sum(len(v) for v in unseen.values())),
                "premise_chưa_thấy_khác_nhau": len(set().union(*unseen.values())),
                "truy_vấn_có_nhãn_chưa_thấy": len(q_u), "theorem_có_nhãn_chưa_thấy": thm[q_u].nunique(),
                "theorem_được_chấm": test.theorem.nunique()}
        per_sys = {}
        for s in ("B0", "V2", "V4", "V4-K240", "V5"):
            e = man.get(f"{s}|{ev}")
            if not e or not e.get("run_file"):
                continue
            run = pd.read_csv(ROOT / e["run_file"], sep="\t")
            ranked = run.sort_values(["query_id", "rank"]).groupby("query_id").premise_id.apply(list).to_dict()
            row = {**info, "hệ_thống": s}
            vals = {}
            for k in (10, 50):
                ru = [len(unseen[qid] & set(ranked.get(qid, [])[:k])) / len(unseen[qid]) for qid in q_u]
                q_s = [qid for qid, v in seenset.items() if v]
                rs = [len(seenset[qid] & set(ranked.get(qid, [])[:k])) / len(seenset[qid]) for qid in q_s]
                row[f"R@{k}_chưa_thấy"] = float(np.mean(ru))
                row[f"R@{k}_đã_thấy"] = float(np.mean(rs))
                vals[k] = np.array(ru)
            per_sys[s] = vals
            seen_rows.append(row)
        for a, b in (("V4", "V5"), ("V4", "V4-K240"), ("V4-K240", "V5")):
            if a in per_sys and b in per_sys:
                args = (per_sys[a][10], per_sys[b][10], thm[q_u].to_numpy())
                r = paired_cluster_bootstrap(*args)
                r["p"] = paired_cluster_permutation(*args)["p"]
                unseen_sig.append({"tập": ev, "từ": a, "đến": b, "metric": "R@10_chưa_thấy", **r})
    su = pd.DataFrame(seen_rows)
    su.to_csv(OUT / "seen_unseen.csv", index=False)
    us = pd.DataFrame(unseen_sig)
    us.to_csv(OUT / "unseen_significance.csv", index=False)
    md += ["", "## 4. Premise đã thấy và chưa thấy trong train cùng split", "",
           md_table(su), "", md_table(us)]

    # ---- 5. theorem ít và nhiều truy vấn: mức tăng của V5 tập trung ở đâu
    buckets = []
    for ev in EVALS:
        frames = {s: per_query(s, ev) for s in ("V4", "V4-K240", "V5")}
        if any(d is None for d in frames.values()):
            continue
        size = frames["V4"].groupby("theorem").size()
        grp = pd.cut(size, [0, 1, 4, 10_000], labels=["1", "2-4", ">=5"]).astype(str)
        for g in ("1", "2-4", ">=5"):
            th = set(grp[grp == g].index)
            row = {"tập": ev, "nhóm_số_truy_vấn": g, "n_theorem": len(th)}
            for s, d in frames.items():
                x = d[d.theorem.isin(th)]
                row["n_truy_vấn"] = len(x)
                row[f"R@10_{s}"] = x["R@10"].mean()
            buckets.append(row)
    bk = pd.DataFrame(buckets)
    bk.to_csv(OUT / "theorem_buckets.csv", index=False)
    md += ["", "## 5. Theo số truy vấn của theorem (R@10 trung bình theo truy vấn trong nhóm)", "", md_table(bk)]
    (ROOT / "results" / "figures" / "robustness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
