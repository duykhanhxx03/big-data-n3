"""Ví dụ xuyên suốt cho báo cáo: một truy vấn validation đi qua từng giai đoạn của hệ thống.

Truy vấn được chọn theo quy tắc cố định, không theo kết quả: truy vấn random:val:36:0 là ví dụ định nghĩa benchmark ở
Mục 3.3 (một gold premise, proof state ngắn). Số liệu lấy từ tệp kết quả top 100 của các lần chạy validation đã ghim
(results/figures/report_manifest.json), không chạy lại truy hồi.

Dùng:  python scripts/running_example.py [--query-id random:val:36:0]
Ra:    report/latex/tables/example.tex, results/figures/running_example.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis.tokenizer import default_tokens, lean_tokens  # noqa: E402

SYSTEMS = [("B0", "BM25, tokenizer mặc định"), ("V2", "BM25F ba trường, tokenizer Lean"),
           ("V2a", "V2 và bộ lọc accessible premise"), ("V4", "V2a và reranker 13 đặc trưng"),
           ("V5", "V4 và candidate từ lịch sử sử dụng")]


def tex(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%") \
        .replace("#", "\\#").replace("{", "\\{").replace("}", "\\}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-id", default="random:val:36:0")
    args = ap.parse_args()
    man = json.loads((ROOT / "results" / "figures" / "report_manifest.json").read_text(encoding="utf-8"))
    P = ROOT / "data" / "processed"
    needed = [P / "queries.parquet"] + [ROOT / man[f"{s}|random_val_all"]["run_file"] for s, _ in SYSTEMS]
    if not all(x.exists() for x in needed):
        # bản checkout không có dữ liệu đầy đủ và tệp run (không đưa vào Git): giữ nguyên tables/example.tex đã sinh
        print("thiếu data/processed hoặc results/runs; giữ nguyên report/latex/tables/example.tex")
        return
    q = pq.read_table(P / "queries.parquet").to_pandas().set_index("query_id").loc[args.query_id]
    prem = pq.read_table(P / "premises.parquet", columns=["premise_id", "full_name"]).to_pandas().set_index("premise_id")
    qr = pq.read_table(P / "qrels.parquet").to_pandas()
    gold = set(qr[qr.query_id == args.query_id].premise_id)
    gold_names = ", ".join(prem.loc[sorted(gold), "full_name"])

    rows, md = [], [f"# Ví dụ xuyên suốt: {args.query_id}", "", "```", q.state_before, "```", "",
                    f"Premise đúng: {gold_names}", "", f"Token mặc định: {default_tokens(q.state_before)}",
                    f"Token Lean: {lean_tokens(q.state_before)}", "", "| Hệ thống | Hạng gold premise | Top 3 |",
                    "|---|---|---|"]
    for sys_name, desc in SYSTEMS:
        e = man[f"{sys_name}|random_val_all"]
        run = pd.read_csv(ROOT / e["run_file"], sep="\t")
        run = run[run.query_id == args.query_id].sort_values("rank")
        hit = run[run.premise_id.isin(gold)]
        rank = str(int(hit["rank"].min())) if len(hit) else "ngoài top 100"
        top3 = [prem.loc[p, "full_name"] for p in run.premise_id.head(3)]
        rows.append((sys_name, desc, rank, top3))
        md.append(f"| {sys_name} | {rank} | {', '.join(top3)} |")

    body = ["% Sinh bởi scripts/running_example.py, không sửa tay.", "\\begin{table}[htbp]", "\\centering",
            f"\\caption[Ví dụ xuyên suốt]{{Ví dụ xuyên suốt: truy vấn validation \\code{{{tex(args.query_id)}}}, gold premise "
            f"\\code{{{tex(gold_names)}}}. Hạng lấy từ kết quả top 100 của từng biến thể; ba kết quả đầu theo thứ tự hạng}}"
            "\\label{tab:example}", "\\begin{tabularx}{\\linewidth}{@{}l>{\\raggedright\\arraybackslash}p{4.2cm}r"
            ">{\\raggedright\\arraybackslash}X@{}}", "\\toprule", "Biến thể & Thay đổi & Hạng & Ba kết quả đầu \\\\",
            "\\midrule"]
    for sys_name, desc, rank, top3 in rows:
        body.append(f"{sys_name} & {desc} & {rank} & " + "; ".join(f"\\code{{{tex(t)}}}" for t in top3) + " \\\\")
    body += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
    out = ROOT / "report" / "latex" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "example.tex").write_text("\n".join(body), encoding="utf-8")
    (ROOT / "results" / "figures" / "running_example.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
