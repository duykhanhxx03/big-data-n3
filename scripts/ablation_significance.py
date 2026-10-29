"""Kiểm định ablation: so sánh từng biến thể với abl_full trên metric từng truy vấn.

Khoảng tin cậy từ paired bootstrap theo cụm theorem; p hai phía từ kiểm định hoán vị ghép cặp theo cụm theorem
(eval.metrics.paired_cluster_permutation), hiệu chỉnh Holm trên 9 phép so sánh của từng metric.
Đọc các lần chạy ablation đã ghim trong results/figures/report_records.json (scripts/pin_runs.py); nếu chưa ghim thì
lấy run record mới nhất có tag "ablation" của mỗi biến thể, cùng file per-query tương ứng.

Dùng:  python scripts/ablation_significance.py
Ra:    results/figures/ablation_significance.md, results/figures/ablation_significance.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq  # noqa: E402

from eval.metrics import holm, paired_cluster_bootstrap, paired_cluster_permutation  # noqa: E402

ORDER = ["abl_full", "abl_no_bm25", "abl_no_namespace", "abl_no_location", "abl_no_symbols", "abl_no_kind",
         "abl_no_dependency", "abl_no_filter", "abl_k50", "abl_k200"]
WHAT = {"abl_full": "V4 đầy đủ (tham chiếu)", "abl_no_bm25": "bỏ bm25_rel, bm25_rr",
        "abl_no_namespace": "bỏ ns_in_query, ns_thm", "abl_no_location": "bỏ same_file, dir_overlap",
        "abl_no_symbols": "bỏ sym_jaccard", "abl_no_kind": "bỏ 4 đặc trưng kind_*",
        "abl_no_dependency": "bỏ popularity, pagerank", "abl_no_filter": "tắt bộ lọc truy cập được",
        "abl_k50": "K = 50 ứng viên", "abl_k200": "K = 200 ứng viên"}
METRICS = ["R@10", "MRR"]


def latest() -> dict[str, dict]:
    pinned = ROOT / "results" / "figures" / "report_records.json"
    if pinned.exists():
        ids = json.loads(pinned.read_text(encoding="utf-8")).get("ablation", {})
        if ids:
            return {n: json.loads((ROOT / "results" / "raw" / "eval" / f"{rid}.json").read_text(encoding="utf-8"))
                    for n, rid in ids.items()}
    recs: dict[str, dict] = {}
    for f in (ROOT / "results" / "raw" / "eval").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        p = r["parameters"]
        if p.get("tag") == "ablation" and p["eval_set"] == "random_test_all":
            name = p["config"]["name"]
            if name not in recs or r["timestamp"] > recs[name]["timestamp"]:
                recs[name] = r
    return recs


def main() -> None:
    recs = latest()
    missing = [n for n in ORDER if n not in recs]
    if missing:
        raise SystemExit(f"thiếu run ablation: {missing} (chạy scripts/ablation.py trước)")
    per = {n: pd.read_parquet(ROOT / recs[n]["per_query_file"]).set_index("query_id").sort_index() for n in ORDER}
    base = per["abl_full"]
    q = pq.read_table(ROOT / "data" / "processed" / "queries.parquet",
                      columns=["query_id", "file_path", "theorem_full_name"]).to_pandas().set_index("query_id")
    clusters = (q.file_path + "::" + q.theorem_full_name).reindex(base.index).to_numpy()
    rows = []
    for n in ORDER:
        d = per[n]
        assert d.index.equals(base.index)
        row = {"variant": n, "change": WHAT[n], "n": len(d)}
        for m in METRICS:
            row[m] = float(d[m].mean())
            if n != "abl_full":
                # các truy vấn cùng một chứng minh không độc lập: KTC bootstrap và hoán vị đều theo cụm theorem
                args = (base[m].to_numpy(), d[m].to_numpy(), clusters)
                bs = paired_cluster_bootstrap(*args)
                row.update({f"d_{m}": bs["delta"], f"lo_{m}": bs["ci95_low"], f"hi_{m}": bs["ci95_high"],
                            f"p_{m}": paired_cluster_permutation(*args)["p"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    for m in METRICS:   # Holm trên họ 9 phép so sánh của từng metric
        mask = df.variant != "abl_full"
        df.loc[mask, f"pholm_{m}"] = holm(df.loc[mask, f"p_{m}"].tolist())
    out = ROOT / "results" / "figures"
    df.to_csv(out / "ablation_significance.csv", index=False)

    def verdict(r) -> str:
        if r["variant"] == "abl_full":
            return ""
        if r["pholm_R@10"] < 0.05:
            return "tốt hơn có ý nghĩa" if r["d_R@10"] > 0 else "kém hơn có ý nghĩa"
        return "không phân biệt được sau hiệu chỉnh"

    lines = ["# Ablation kèm kiểm định (sinh bởi scripts/ablation_significance.py)", "",
             f"random_test_all, n = {len(base)}. Δ = biến thể trừ V4 đầy đủ; KTC 95% từ paired bootstrap theo cụm theorem, "
             "p hai phía từ kiểm định hoán vị ghép cặp theo cụm theorem (10.000 phép hoán vị, seed 42, p = (1 + số lần cực "
             "trị) / 10.001); p Holm hiệu chỉnh trên 9 phép so sánh của từng metric. Mỗi biến thể huấn luyện lại reranker "
             "trên cùng 6.000 truy vấn train nửa B.", "",
             "| Biến thể | Thay đổi | R@10 | ΔR@10 [KTC 95%] | p (Holm) | MRR | ΔMRR [KTC 95%] | p (Holm) | Kết luận theo R@10 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in df.to_dict("records"):
        if r["variant"] == "abl_full":
            lines.append(f"| {r['variant']} | {r['change']} | {r['R@10']:.4f} |  |  | {r['MRR']:.4f} |  |  |  |")
            continue
        cells = []
        for m in METRICS:
            p_, ph = r[f"p_{m}"], r[f"pholm_{m}"]
            cells += [f"{r[m]:.4f}", f"{r[f'd_{m}']:+.4f} [{r[f'lo_{m}']:+.4f}, {r[f'hi_{m}']:+.4f}]",
                      f"{p_:.4f} ({ph:.4f})"]
        lines.append(f"| {r['variant']} | {r['change']} | " + " | ".join(cells) + f" | {verdict(r)} |")
    (out / "ablation_significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
