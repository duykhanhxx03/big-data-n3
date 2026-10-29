"""Sinh các bảng số liệu của báo cáo LaTeX từ kết quả thô, không chép tay.

Mỗi bảng là một tệp report/latex/tables/<tên>.tex (môi trường table đầy đủ, có caption và label), được main.tex nạp bằng
\\input. Số liệu chất lượng lấy từ các lần chạy đã ghim trong results/figures/report_manifest.json (scripts/pin_runs.py);
các bảng khác lấy từ CSV và run record ghi rõ trong report/latex/tables/SOURCES.md.

Dùng:  python scripts/make_report_tables.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from runlog import file_sha256  # noqa: E402
from shared import EXPECTED_COUNTS  # noqa: E402

FIG = ROOT / "results" / "figures"
RAW = ROOT / "results" / "raw"
OUT = ROOT / "report" / "latex" / "tables"
SOURCES: list[tuple[str, str]] = []
INPUTS: set[Path] = set()


def inp(path: Path) -> Path:
    """Ghi nhận một tệp đầu vào để liệt kê kèm SHA-256 trong SOURCES.md."""
    INPUTS.add(path)
    return path


def pinned_records() -> dict:
    f = FIG / "report_records.json"
    return json.loads(inp(f).read_text(encoding="utf-8"))["records"] if f.exists() else {}


# ---------- định dạng số kiểu tiếng Việt ----------
def f(x: float, nd: int = 3) -> str:
    s = f"{abs(x):.{nd}f}".replace(".", ",")
    return ("$-$" if x < 0 and round(abs(x), nd) > 0 else "") + s


def sg(x: float, nd: int = 3) -> str:
    """Có dấu: $+$0,038 hoặc $-$0,051."""
    s = f"{abs(x):.{nd}f}".replace(".", ",")
    return ("$-$" if x < 0 else "$+$") + s


def ci(lo: float, hi: float, nd: int = 3) -> str:
    """Khoảng tin cậy; cận sát 0 (làm tròn về 0 ở nd chữ số) được in thêm một chữ số để thấy rõ dấu của nó."""
    def b(x: float) -> str:
        k = nd + 1 if x != 0 and round(abs(x), nd) == 0 else nd
        return ("$-$" if x < 0 else "") + f"{abs(x):.{k}f}".replace(".", ",")
    return f"[{b(lo)}; {b(hi)}]"


def n(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def p(x: float) -> str:
    return f(x, 4)


def write(name: str, caption: str, label: str, body: list[str], colspec: str, sources: str, short: str | None = None,
          env: str = "tabular", sep: str | None = None, note: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cap = f"\\caption[{short}]{{{caption}}}" if short else f"\\caption{{{caption}}}"
    tex = ["% Sinh bởi scripts/make_report_tables.py, không sửa tay.", "\\begin{table}[htbp]", "\\centering",
           *([f"\\setlength{{\\tabcolsep}}{{{sep}}}"] if sep else []),
           f"{cap}\\label{{{label}}}",
           f"\\begin{{{env}}}{{{colspec}}}" if env == "tabular" else f"\\begin{{{env}}}{{\\linewidth}}{{{colspec}}}",
           "\\toprule", *body, "\\bottomrule", f"\\end{{{env}}}",
           *([f"\\par\\smallskip{{\\footnotesize\\raggedright Ghi chú: {note}\\par}}"] if note else []), "\\end{table}", ""]
    (OUT / f"{name}.tex").write_text("\n".join(tex), encoding="utf-8")
    SOURCES.append((name, sources))


def load_records(kind: str) -> list[dict]:
    """Run record của một loại; nếu đã ghim (scripts/pin_runs.py) thì chỉ đọc đúng danh sách đã ghim."""
    ids = pinned_records().get(kind)
    files = [RAW / kind / f"{i}.json" for i in ids] if ids is not None else sorted((RAW / kind).glob("*.json"))
    return [json.loads(inp(x).read_text(encoding="utf-8")) for x in files]


# ---------- bảng ----------
def t_dataflow():
    d = pd.read_csv(inp(FIG / "robustness" / "data_flow.csv"))
    cols = d.to_dict("records")
    steps = [("theorem_gốc", "Theorem gốc"), ("theorem_có_tactic", "\\quad có ít nhất một tactic"),
             ("theorem_có_tactic_có_premise", "\\quad có tactic được chú thích premise"),
             ("theorem_còn_truy_vấn", "\\quad còn ít nhất một truy vấn"),
             ("tỉ_lệ_theorem_giữ_lại", "\\quad tỉ lệ so với theorem gốc"), (None, None), ("tactic", "Tactic"),
             ("tactic_có_premise", "\\quad được chú thích premise"),
             ("truy_vấn_bị_loại", "\\quad bị loại vì mọi gold premise nằm ngoài corpus"),
             ("truy_vấn_giữ_lại", "Truy vấn giữ lại"), ("proof_state_khác_nhau", "\\quad proof state khác nhau"),
             (None, None), ("nhãn_không_có_trong_corpus", "Lượt gold premise nằm ngoài corpus")]
    body = ["& \\multicolumn{3}{c}{\\code{random}} & \\multicolumn{3}{c}{\\code{novel\\_premises}} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}", "Bước & train & val & test & train & val & test \\\\", "\\midrule"]
    for key, lab in steps:
        if key is None:
            body.append("\\midrule")
            continue
        if key == "tỉ_lệ_theorem_giữ_lại":
            cells = [f"{f(100 * r[key], 1)}\\%" for r in cols]
        else:
            cells = [n(r[key]) for r in cols]
        body.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("dataflow", "Luồng lọc dữ liệu theo split và tập", "tab:dataflow", body, "@{}lrrrrrr@{}", note=
          "phần lớn theorem bị loại vì được chứng minh không qua tactic (chứng minh dạng term) nên không sinh truy vấn nào.",
          sources=
          "results/figures/robustness/data_flow.csv (scripts/robustness.py)", short="Luồng lọc dữ liệu theo split")


def manifest() -> dict:
    return json.loads(inp(FIG / "report_manifest.json").read_text(encoding="utf-8"))


def t_quality():
    m = manifest()
    systems = ["B0", "B0t", "V1", "V2", "V3", "V2a", "V4", "V5"]
    body = ["& \\multicolumn{6}{c}{\\code{random}} & \\multicolumn{3}{c}{\\code{novel\\_premises}} \\\\",
            "\\cmidrule(lr){2-7}\\cmidrule(l){8-10}",
            "Biến thể & R@1 & R@5 & R@10 & R@50 & MRR & nDCG@10 & R@10 & R@50 & MRR \\\\", "\\midrule"]
    cols = [("random_test_all", k) for k in ("R@1", "R@5", "R@10", "R@50", "MRR", "nDCG@10")] + \
           [("novel_premises_test_all", k) for k in ("R@10", "R@50", "MRR")]
    vals = {s: [m[f"{s}|{ev}"]["metrics"][k] for ev, k in cols] for s in systems}
    best = [max(round(vals[s][j], 3) for s in systems) for j in range(len(cols))]
    for s in systems:
        cells = [f"\\textbf{{{f(v)}}}" if round(v, 3) == best[j] else f(v) for j, v in enumerate(vals[s])]
        body.append(f"{s} & " + " & ".join(cells) + " \\\\")
    write("quality", "Chất lượng truy hồi trên toàn bộ tập test, query-macro", "tab:quality", body, "@{}l" + "r" * 9 + "@{}",
          note="\\num{2641} truy vấn \\code{random}, \\num{4141} truy vấn \\code{novel\\_premises}. Chữ đậm là giá trị cao "
          "nhất của từng cột; cấu hình đề xuất dùng chung là V4 (Mục~\\ref{sec:robust}).", sources=
          "results/figures/report_manifest.json (các khóa <hệ thống>|random_test_all, |novel_premises_test_all)",
          short="Chất lượng truy hồi trên toàn bộ tập test")


def t_signif():
    s = pd.read_csv(inp(FIG / "robustness" / "significance_cluster.csv"))
    s = s[(s.metric == "R@10") & (s.trung_bình == "truy vấn")]
    pairs = [("B0", "B0t"), ("B0t", "V1"), ("V1", "V2"), ("V2", "V3"), ("V2", "V2a"), ("V2a", "V4"), ("V4", "V5"),
             ("B0", "V5")]
    body = ["& \\multicolumn{2}{c}{\\code{random}} & \\multicolumn{2}{c}{\\code{novel\\_premises}} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
            "So sánh & $\\Delta$Recall@10 [KTC 95\\%] & $p$ Holm & $\\Delta$Recall@10 [KTC 95\\%] & $p$ Holm \\\\", "\\midrule"]
    for a, b in pairs:
        if (a, b) == ("B0", "V5"):
            body.append("\\midrule")
        cells = []
        for ev in ("random_test_all", "novel_premises_test_all"):
            r = s[(s.tập == ev) & (s.từ == a) & (s.đến == b)].iloc[0]
            pv = r.p_holm if not np.isnan(r.p_holm) else r.p
            cells += [f"{sg(r.delta)} \\ {ci(r.ci95_low, r.ci95_high)}", p(pv) + ("" if not np.isnan(r.p_holm) else "*")]
        body.append(f"{b} so với {a} & " + " & ".join(cells) + " \\\\")
    write("signif", "Chênh lệch Recall@10 giữa các bước cải tiến trên tập test, query-macro", "tab:signif", body,
          "@{}lllll@{}", note="KTC 95\\% và $p$ hai phía theo Mục~\\ref{sec:metrics}; $p$ hiệu chỉnh Holm trên 7 bước của "
          "chuỗi trong từng split, giá trị nhỏ nhất có thể là $1/\\num{10001}$. Dấu * là $p$ chưa hiệu chỉnh.", sources=
          "results/figures/robustness/significance_cluster.csv (scripts/robustness.py)",
          short="Chênh lệch Recall@10 giữa các bước cải tiến")


def t_robust():
    mac = pd.read_csv(inp(FIG / "robustness" / "macro.csv")).set_index(["tập", "hệ_thống"])
    s = pd.read_csv(inp(FIG / "robustness" / "significance_cluster.csv"))
    s = s[s.metric == "R@10"]
    names = {"random_test_all": "\\code{random}", "novel_premises_test_all": "\\code{novel\\_premises}",
             "random_holdoutB": "Holdout"}
    body = ["Tập & Macro & V4 & V4-K240 & V5 & V5 $-$ V4 & V5 $-$ V4-K240 \\\\", "\\midrule"]
    for ev, label in names.items():
        for est, col in (("truy vấn", "R@10_query"), ("theorem", "R@10_theorem")):
            vals = [f(mac.loc[(ev, sy), col]) for sy in ("V4", "V4-K240", "V5")]
            ds = []
            for a in ("V4", "V4-K240"):
                r = s[(s.tập == ev) & (s.từ == a) & (s.đến == "V5") & (s.trung_bình == est)].iloc[0]
                ds.append(f"{sg(r.delta)} {ci(r.ci95_low, r.ci95_high)}")
            body.append(f"{label if est == 'truy vấn' else ''} & {'query' if est == 'truy vấn' else 'theorem'} & "
                        + " & ".join(vals + ds) + " \\\\")
        if ev != "random_holdoutB":
            body.append("\\midrule")
    write("robust", "Recall@10 của V4, V4-K240 và V5 trên hai tập test và holdout, theo query-macro và theorem-macro",
          "tab:robust", body, "@{}llrrrll@{}", sep="3.5pt", note="KTC 95\\% từ bootstrap theo cụm theorem. Holdout gồm "
          "\\num{2500} truy vấn của \\num{1138} theorem; số candidate của V4-K240 so với V5 xem Mục~\\ref{sec:setup}.",
          sources=
          "results/figures/robustness/macro.csv, significance_cluster.csv; data/processed/eval_sets/random_holdoutB.json",
          short="Độ bền của kết quả V5")


def t_unseen():
    d = pd.read_csv(inp(FIG / "robustness" / "seen_unseen.csv"))
    d = d[d.tập == "novel_premises_test_all"].set_index("hệ_thống")
    info = d.iloc[0]
    body = ["Hệ thống & R@10, unseen & R@50, unseen & R@10, seen & R@50, seen \\\\", "\\midrule"]
    for sy in ("B0", "V2", "V4", "V4-K240", "V5"):
        if sy in d.index:
            r = d.loc[sy]
            body.append(f"{sy} & {f(r['R@10_chưa_thấy'])} & {f(r['R@50_chưa_thấy'])} & {f(r['R@10_đã_thấy'])} & "
                        f"{f(r['R@50_đã_thấy'])} \\\\")
    write("unseen", "Recall trên unseen premise và seen premise, tập test \\code{novel\\_premises}", "tab:unseen", body,
          "@{}lrrrr@{}", note=f"unseen premise chiếm {n(info['cặp_chưa_thấy'])} trên {n(info['cặp_nhãn'])} cặp nhãn, "
          f"thuộc {n(info['truy_vấn_có_nhãn_chưa_thấy'])} truy vấn và {n(info['theorem_có_nhãn_chưa_thấy'])} trên "
          f"{n(info['theorem_được_chấm'])} theorem; mỗi cột chỉ lấy trung bình trên các truy vấn có loại nhãn tương ứng.",
          sources= "results/figures/robustness/seen_unseen.csv (scripts/robustness.py)",
          short="Recall trên unseen premise và seen premise")


def t_controls():
    m = manifest()
    s = pd.read_csv(inp(FIG / "robustness" / "significance_cluster.csv"))
    s = s[(s.trung_bình == "truy vấn") & (s.tập == "random_test_all")]
    rows = [("V2", None, "V2"), ("V2-nons", "V2", "V2-nons"), (None, None, None), ("V4", None, "V4"),
            ("V4-poponly", "V4", "V4-poponly"), ("V4-pronly", "V4", "V4-pronly")]
    body = ["& \\multicolumn{2}{c}{Validation} & \\multicolumn{4}{c}{Test \\code{random}} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(l){4-7}",
            "Cấu hình & R@10 & MRR & R@10 & MRR & $\\Delta$R@10 & $\\Delta$MRR \\\\", "\\midrule"]
    for sy, ref, lab in rows:
        if sy is None:
            body.append("\\midrule")
            continue
        a, b = m[f"{sy}|random_val_all"]["metrics"], m[f"{sy}|random_test_all"]["metrics"]
        cells = [f(a["R@10"]), f(a["MRR"]), f(b["R@10"]), f(b["MRR"])]
        for metric in ("R@10", "MRR"):
            if ref is None:
                cells.append("")
                continue
            # bảng significance lưu chiều (biến thể rút gọn -> đầy đủ); đổi dấu để đọc là "rút gọn trừ đầy đủ"
            r = s[(s.từ == sy) & (s.đến == ref) & (s.metric == metric)].iloc[0]
            cells.append(f"{sg(-r.delta)} {ci(-r.ci95_high, -r.ci95_low)}")
        body.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("controls", "Đối chứng cho trường namespace và hai đặc trưng phụ thuộc", "tab:controls", body,
          "@{}lrrrrll@{}", sep="5pt", note="V2-nons bỏ trường namespace khỏi BM25F; V4-poponly và V4-pronly chỉ giữ "
          "popularity hoặc PageRank. $\\Delta$ là cấu hình rút gọn trừ cấu hình đầy đủ trên tập test, kèm KTC 95\\%.",
          sources=
          "results/figures/report_manifest.json; results/figures/robustness/significance_cluster.csv",
          short="Đối chứng namespace và đặc trưng phụ thuộc")


def t_candidates():
    c = pd.read_csv(inp(FIG / "robustness" / "candidates.csv"))
    names = {"random_test_all": "\\code{random}", "novel_premises_test_all": "\\code{novel\\_premises}",
             "random_holdoutB": "Holdout"}
    body = ["Tập & Hệ thống & Candidate & Candidate recall & Oracle@10 & R@10 & Ranking gap & R@50 \\\\", "\\midrule"]
    for ev, label in names.items():
        for i, r in enumerate(c[c.tập == ev].to_dict("records")):
            body.append(f"{label if i == 0 else ''} & {r['hệ_thống']} & {f(r['ứng_viên_TB'], 0)} & {f(r['độ_phủ'])} & "
                        f"{f(r['Oracle@10'])} & {f(r['R@10'])} & {f(r['mất_ở_xếp_hạng@10'])} & {f(r['R@50'])} \\\\")
        if ev != "random_holdoutB":
            body.append("\\midrule")
    write("candidates", "Candidate recall, Oracle@10 và ranking gap, query-macro", "tab:candidates", body,
          "@{}llrrrrrr@{}", sep="5pt", note="định nghĩa ở Mục~\\ref{sec:metrics}; Oracle@50 trùng candidate recall đến ba "
          "chữ số nên không ghi.", sources=
          "results/figures/robustness/candidates.csv (scripts/candidate_analysis.py)",
          short="Phân rã giới hạn: candidate và xếp hạng")


def t_v5grid():
    d = pd.read_csv(inp(RAW / "tuning" / "V5__extra__random_val_all.csv"))
    m = manifest()["V4|random_val_all"]["metrics"]
    best = d.sort_values(["R@10", "MRR"], ascending=False).index[0]
    body = ["Tệp & Thư mục & Phổ biến & Số candidate & Candidate recall & R@10 & R@50 & MRR \\\\", "\\midrule"]
    for i, r in d.iterrows():
        cells = [str(int(r.file)), str(int(r.dir)), str(int(r.popular)), f(r.pool_size, 0), f(r.pool_recall), f(r["R@10"]),
                 f(r["R@50"]), f(r.MRR)]
        if i == best:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        body.append(" & ".join(cells) + " \\\\")
    write("v5grid", f"Tinh chỉnh số candidate bổ sung của V5 trên validation \\code{{random}} (\\num{{2536}} truy vấn). V4 "
          f"trên cùng tập: Recall@10 = {f(m['R@10'])}; Recall@50 = {f(m['R@50'])}; MRR = {f(m['MRR'])}", "tab:v5grid",
          body, "@{}rrrrrrrr@{}", "results/raw/tuning/V5__extra__random_val_all.csv (scripts/tune_v5.py); manifest V4|random_val_all",
          short="Tinh chỉnh số candidate bổ sung của V5")


def t_leandojo():
    m = manifest()
    # Giữ đúng độ chính xác công bố: R@1, R@10 là phần trăm một chữ số thập phân, MRR hai chữ số
    pub = {"bm25_all": ("0,019", "0,119", "0,08", "0,021", "0,124", "0,08"),
           "bm25_acc": ("0,067", "0,172", "0,15", "0,059", "0,155", "0,14"),
           "reprover": ("0,128", "0,347", "0,29", "0,098", "0,321", "0,24")}

    def ours(s):
        a, b = m[f"{s}|random_test_all"]["metrics"], m[f"{s}|novel_premises_test_all"]["metrics"]
        return " & ".join(f(x) for x in (a["R@1"], a["R@10"], a["MRR"], b["R@1"], b["R@10"], b["MRR"]))

    def pubrow(k):
        return " & ".join(pub[k])

    body = ["& \\multicolumn{3}{c}{\\code{random}} & \\multicolumn{3}{c}{\\code{novel\\_premises}} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}", "Hệ thống & R@1 & R@10 & MRR & R@1 & R@10 & MRR \\\\", "\\midrule",
            f"BM25, toàn bộ premise (LeanDojo, Bảng 1, Lean 3) & {pubrow('bm25_all')} \\\\", f"B0 của nhóm & {ours('B0')} \\\\",
            "\\midrule", f"BM25, accessible premise (LeanDojo, Bảng 1, Lean 3) & {pubrow('bm25_acc')} \\\\",
            f"V2a của nhóm & {ours('V2a')} \\\\", f"V4 của nhóm & {ours('V4')} \\\\", f"V5 của nhóm & {ours('V5')} \\\\",
            "\\midrule", f"ReProver (LeanDojo, Bảng B, Lean 4, snapshot cũ hơn v10) & {pubrow('reprover')} \\\\"]
    write("leandojo", "Số liệu công bố trong bài báo LeanDojo đặt cạnh kết quả của nhóm, chỉ để làm bối cảnh: các dòng của "
          "bài báo đo trên benchmark và snapshot khác nên không so sánh trực tiếp được", "tab:leandojo", body,
          "@{}>{\\raggedright\\arraybackslash}X" + "r" * 6 + "@{}",
          "Yang và cộng sự (2023), arXiv:2306.15626, Bảng 1 và Bảng B; manifest cho các dòng của nhóm",
          short="Bối cảnh: số liệu công bố của LeanDojo", env="tabularx")


def t_shared():
    """Benchmark chung N3/N4 (gói n4_shared): lần chạy scripts/eval_shared.py đã ghim, mới nhất nếu có nhiều lần."""
    rec = max(load_records("shared"), key=lambda r: r["timestamp"])
    m, sl = rec["metrics"], {"all": "toàn bộ", "signature": "không chứa chứng minh"}
    n_all, n_sig = (next(iter(m.values()))[s]["n"] for s in sl)
    body = [f"& \\multicolumn{{4}}{{c}}{{Toàn bộ ({n(n_all)} truy vấn)}} & "
            f"\\multicolumn{{2}}{{c}}{{Không chứa chứng minh ({n(n_sig)})}} \\\\",
            "\\cmidrule(lr){2-5}\\cmidrule(l){6-7}", "Hệ thống & R@1 & R@10 & R@50 & MRR & R@10 & MRR \\\\", "\\midrule"]
    body += [f"{s} & " + " & ".join(f(v["all"][k]) for k in ("R@1", "R@10", "R@50", "MRR")) + " & "
             + " & ".join(f(v["signature"][k]) for k in ("R@10", "MRR")) + " \\\\" for s, v in m.items()]
    steps = "; ".join(f"trên tập {sl[s]}, " + ", ".join(
        f"{c['a']} sang {c['b']} {sg(c['delta_R@10'])} (p Holm {p(c['p_holm'])})"
        for c in rec["comparisons"] if c["slice"] == s) for s in sl)
    write("shared", f"Kết quả trên benchmark chung với Nhóm 4: gói \\code{{n4\\_shared}} gồm "
          f"{n(EXPECTED_COUNTS['corpus.jsonl'])} premise định danh bằng \\code{{full\\_name}} và "
          f"{n(EXPECTED_COUNTS['queries.jsonl'])} theorem của \\code{{random/test}}, mỗi truy vấn là statement của một "
          "theorem", "tab:shared", body, "@{}l" + "r" * 6 + "@{}",
          f"results/raw/shared/{rec['run_id']}.json (đã ghim trong report_records.json); src/shared.py",
          short="Kết quả trên benchmark chung với Nhóm 4",
          note="Mọi hệ thống loại chính theorem truy vấn trước khi cắt top-k; V4 và V5 lọc thêm accessible premise. "
               f"{n(rec['n_queries'] - rec['n_scored'])} truy vấn không có premise đúng trong corpus không được chấm. "
               f"Chênh lệch Recall@10 {steps}.")


def t_lucene():
    recs = {}
    for r in load_records("lucene"):
        recs[r["parameters"]["variant"]] = r   # bản ghi sau cùng của mỗi biến thể
    m = manifest()
    b0 = m["B0|random_test_all"]
    label = {"lucene-default": "Lucene, bộ phân tích tiếng Anh, $k_1 = 0{,}9$, $b = 0{,}4$",
             "lucene-english-b0params": "Lucene, bộ phân tích tiếng Anh, $k_1 = 1{,}2$, $b = 0{,}75$",
             "lucene-b0tokens": "Lucene, token của B0, $k_1 = 1{,}2$, $b = 0{,}75$"}
    body = ["Hệ thống & R@10 & R@50 & MRR & Trùng top 10 & ms/truy vấn \\\\", "\\midrule"]
    for k, lab in label.items():
        if k in recs:
            x = recs[k]["metrics"]
            body.append(f"{lab} & {f(x['R@10'], 4)} & {f(x['R@50'], 4)} & {f(x['MRR'], 4)} & "
                        f"{f(x.get('top10_overlap_b0', float('nan')), 3)} & {f(x['ms_per_query'], 1)} \\\\")
    x = b0["metrics"]
    b0ms = recs["b0-inprocess"]["metrics"]["ms_per_query"]   # đo một luồng cùng quy tắc với Lucene
    body.append(f"B0 của nhóm, $k_1 = 1{{,}}2$, $b = 0{{,}}75$ & {f(x['R@10'], 4)} & {f(x['R@50'], 4)} & {f(x['MRR'], 4)} & "
                f"1 & {f(b0ms, 1)} \\\\")
    write("lucene", "BM25 của Lucene 10.5 (Anserini 2.3.0) so với B0 trên tập test \\code{random}", "tab:lucene", body,
          note="trường statement, một thread. Trùng top 10 là tỉ lệ phần tử chung giữa top 10 của Lucene và của B0, "
          "query-macro; thời gian là trung bình ở trạng thái ổn định (bỏ 100 truy vấn đầu), không gồm khởi động JVM.",
          colspec=
          "@{}>{\\raggedright\\arraybackslash}Xrrrrr@{}", env="tabularx", sources=
          "results/raw/lucene/*.json (scripts/lucene_baseline.py); manifest B0|random_test_all",
          short="BM25 của Lucene so với BM25 tự cài đặt")


def loadtest_rows(prefix: str) -> pd.DataFrame:
    rows = []
    for r in load_records("loadtest"):
        tag = r["parameters"].get("tag") or ""
        if not tag.startswith(prefix):
            continue
        for lv in r["metrics"]["levels"]:
            rows.append({"mode": r["parameters"]["mode"], "tag": tag, "clients": lv["clients"], "p95": lv["e2e_ms_p95"],
                         "qps": lv["qps"]})
    return pd.DataFrame(rows)


def t_v5load():
    rows = []
    for r in load_records("loadtest"):
        tag = r["parameters"].get("tag") or ""
        if tag.startswith("v5cmp2-"):
            for lv in r["metrics"]["levels"]:
                rows.append({"mode": r["parameters"]["mode"], "clients": lv["clients"], "p95": lv["e2e_ms_p95"],
                             "qps": lv["qps"], "err": lv.get("errors", 0), "ok": lv.get("requests_ok", 0)})
    d = pd.DataFrame(rows)
    if d.empty:
        return
    body = ["Biến thể & Client & P95 (ms) [nhỏ nhất; lớn nhất] & QPS [nhỏ nhất; lớn nhất] & Yêu cầu lỗi \\\\", "\\midrule"]
    for k, mode in enumerate(("V4", "V4-K240", "V5")):
        for c in (1, 10, 50):
            g = d[(d["mode"] == mode) & (d.clients == c)]
            body.append(f"{mode if c == 1 else ''} & {c} & {f(g.p95.median(), 1)} [{f(g.p95.min(), 1)}; {f(g.p95.max(), 1)}] & "
                        f"{f(g.qps.median(), 1)} [{f(g.qps.min(), 1)}; {f(g.qps.max(), 1)}] & "
                        f"{int(g.err.sum())} / {n(g.ok.sum() + g.err.sum())} \\\\")
        if k < 2:
            body.append("\\midrule")
    write("v5load", "V4, V4-K240 và V5 dưới tải trong cùng một phiên đo", "tab:v5load", body, "@{}lrlll@{}",
          note="một server 4 worker nạp cả ba cấu hình, thứ tự đo xoay vòng giữa 3 lần lặp, mỗi lần 400 yêu cầu ở mỗi mức "
          "client. P95 tính trên các yêu cầu thành công; QPS tính trên toàn bộ thời gian chạy, kể cả thời gian chờ yêu cầu "
          "bị timeout. Mỗi ô là trung vị [nhỏ nhất; lớn nhất] của 3 lần.", sources=
          "results/raw/loadtest/*.json, tag v5cmp2-r* (run_scale.py --part serve_v5_k240)",
          short="V4, V4-K240 và V5 dưới tải cùng phiên")


def t_workers():
    d = loadtest_rows("workers-")
    d["workers"] = d.tag.map(lambda t: int(t.split("-")[1]))
    mem = defaultdict(list)
    for r in load_records("serve-memory"):
        mem[r["parameters"]["workers"]].append(r["metrics"]["server_rss_mb"])
    body = ["& & & \\multicolumn{2}{c}{10 client} & \\multicolumn{2}{c}{50 client} & \\multicolumn{2}{c}{100 client} \\\\",
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
            "Biến thể & Worker & RAM (GB) & P95 & QPS & P95 & QPS & P95 & QPS \\\\", "\\midrule"]
    for mode in ("B0", "V4"):
        for w in sorted(d.workers.unique()):
            cells = []
            for c in (10, 50, 100):
                g = d[(d["mode"] == mode) & (d.workers == w) & (d.clients == c)]
                cells += [n(g.p95.median()), n(g.qps.median())]
            body.append(f"{mode} & {w} & {f(np.median(mem[w]) / 1024, 1)} & " + " & ".join(cells) + " \\\\")
        if mode == "B0":
            body.append("\\midrule")
    write("workers", "Kiểm tra tải trên chỉ mục đầy đủ theo số worker (P95 tính bằng ms và QPS, trung vị của 3 lần; RAM là "
          "tổng của mọi tiến trình server)", "tab:workers", body, "@{}lrrrrrrrr@{}",
          "results/raw/loadtest/*.json tag workers-*; results/raw/serve-memory/*.json (run_scale.py --part serve2)",
          short="Mở rộng dịch vụ theo số worker")


def build_groups() -> dict:
    g = defaultdict(list)
    for r in load_records("build"):
        if r["status"] == "ok":
            g[(r["parameters"]["label"], r["parameters"]["size"])].append(r["metrics"])
    return g


def t_build():
    g = build_groups()
    med = lambda lab, size: float(np.median([m["wall_s"] for m in g[(lab, size)]]))  # noqa: E731
    body = ["Nền tảng và chỉ mục & Cách xây & 10K & 181K & 905K & 3,6 triệu \\\\", "\\midrule"]
    blocks = [("Windows, ba trường", "win-lean"), ("Linux, ba trường", "linux-lean"), ("Windows, B0", "win-b0"),
              ("Linux, B0", "linux-b0")]
    sizes = ("10k", "full", "x5", "x20")
    for i, (label, pre) in enumerate(blocks):
        seq = [med(f"{pre}-seq", s) for s in sizes]
        sp = [med(f"{pre}-arrow8", s) for s in sizes]
        body.append(f"\\multirow{{3}}{{*}}{{{label}}} & Tuần tự & " + " & ".join(f(x, 1) for x in seq) + " \\\\")
        body.append(" & Spark local[8] & " + " & ".join(f(x, 1) for x in sp) + " \\\\")
        sp_cells = [(f"\\textbf{{{f(a / b, 2)}}}" if a / b > 1 else f(a / b, 2)) for a, b in zip(seq, sp)]
        body.append(" & Tăng tốc & " + " & ".join(sp_cells) + " \\\\")
        if i < len(blocks) - 1:
            body.append("\\midrule")
    write("build", "Thời gian xây chỉ mục (giây, trung vị của 3 lần, gồm cả thời gian khởi động Spark). Tăng tốc là thời gian "
          "tuần tự chia cho thời gian Spark local[8] trên cùng nền tảng; hai mức cuối là corpus nhân bản 5 và 20 lần",
          "tab:build", body, "@{}llrrrr@{}", "results/raw/build/*.json, experiment build2 (run_scale.py --part build2)",
          short="Thời gian xây chỉ mục bằng engine Arrow")


def t_engines():
    g = build_groups()

    def cell(lab, size, key):
        ms = g.get((lab, size))
        if not ms:
            return ""
        v = float(np.median([m[key] for m in ms]))
        return f(v, 1) if key == "wall_s" else f(v / 1024, 1)

    body = ["& & \\multicolumn{2}{c}{Thời gian (s)} & \\multicolumn{2}{c}{RAM đỉnh (GB)} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}", "Kích thước & Engine & Windows & Linux & Windows & Linux \\\\", "\\midrule"]
    rows = [("905K", "RDD", "spark4", None, "x5"), ("905K", "Arrow", "win-b0-arrow4", "linux-b0-arrow4", "x5"),
            ("3,6 triệu", "RDD", "win-b0-rdd4", "linux-b0-rdd4", "x20"),
            ("3,6 triệu", "Arrow", "win-b0-arrow4", "linux-b0-arrow4", "x20")]
    for size, eng, win, lin, sz in rows:
        body.append(f"{size} & {eng} & {cell(win, sz, 'wall_s')} & {cell(lin, sz, 'wall_s') if lin else ''} & "
                    f"{cell(win, sz, 'peak_rss_mb')} & {cell(lin, sz, 'peak_rss_mb') if lin else ''} \\\\")
    write("engines", "So sánh hai engine Spark, chỉ mục B0, Spark local[4] (trung vị của 3 lần; engine RDD ở 3,6 triệu bản "
          "ghi chạy 1 lần trên mỗi nền tảng; engine RDD ở 905K chỉ đo trên Windows)", "tab:engines", body, "@{}llrrrr@{}",
          "results/raw/build/*.json (nhãn spark4, *-b0-arrow4, *-b0-rdd4)", short="So sánh hai engine Spark")


def t_profile():
    d = pd.read_parquet(inp(RAW / "profile" / "latency_profile.parquet"))
    d = d[d["pass"] == "nóng"]
    steps = [("tokenize", "Tokenize"), ("bm25_score", "Cộng điểm BM25/BM25F"), ("accessible_mask", "Tính mặt nạ accessible premise"),
             ("filter_topk", "Lọc và chọn top 100"), ("features", "Tính 13 đặc trưng"), ("rerank_sort", "Rerank và sắp xếp"),
             ("total", "Tổng")]
    body = ["Bước & B0 & V2 & V4 \\\\", "\\midrule"]
    for key, lab in steps:
        if key == "total":
            body.append("\\midrule")
        cells = []
        for cfg in ("B0", "V2", "V4"):
            x = d[d.config == cfg][key]
            # bước không có ở biến thể này chỉ đo được chi phí gọi hàm rỗng (dưới 0,005 ms): để trống
            cells.append("" if x.median() < 0.005 else f"{f(x.median(), 2)} ({f(x.quantile(0.95), 2)})")
        body.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    share = [f"{f(100 * d[d.config == c].bm25_score.sum() / d[d.config == c].total.sum(), 0)}\\%" for c in ("B0", "V2", "V4")]
    body.append("Tỉ lệ thời gian của BM25 & " + " & ".join(share) + " \\\\")
    write("profile", "Thời gian theo bước của một truy vấn (ms, trung vị và P95 trong ngoặc), một tiến trình, chỉ mục đầy "
          "đủ, 500 truy vấn, bộ đệm đã nóng", "tab:profile", body, "@{}lccc@{}",
          "results/raw/profile/latency_profile.parquet (scripts/profile_latency.py)",
          short="Thời gian theo bước của một truy vấn")


def t_errors():
    e = pd.read_parquet(inp(RAW / "error_analysis.parquet"))
    labels = [("ngoai_top100", "Gold premise nằm ngoài 100 candidate của giai đoạn 1"),
              ("nhieu_premise_dung", "Truy vấn có nhiều gold premise"),
              ("premise_tong_quat", "Premise tổng quát, thuộc 1\\% premise được dùng nhiều nhất"),
              ("khong_trung_tu_vung", "Premise không có từ nào chung với truy vấn"),
              ("dinh_nghia_instance", "Premise là định nghĩa, instance hoặc structure"),
              ("truy_van_dai", "Truy vấn dài hơn 160 token"), ("khac", "Không thuộc nhóm nào ở trên")]
    cats = e.categories.str.split(",")
    body = ["Nhóm lỗi & Số cặp & Tỉ lệ \\\\", "\\midrule"]
    for key, lab in labels:
        k = int(cats.map(lambda c: key in c).sum())
        body.append(f"{lab} & {n(k)} & {f(100 * k / len(e), 1)}\\% \\\\")
    write("errors", f"Phân loại tự động {n(len(e))} cặp (truy vấn, gold premise) mà V4 bỏ lỡ ở top 10 trên tập test "
          "\\code{random}; một cặp có thể thuộc nhiều nhóm", "tab:errors", body, "@{}Xrr@{}",
          "results/raw/error_analysis.parquet (scripts/error_analysis.py)", short="Phân loại tự động các lỗi của V4",
          env="tabularx")


def t_ablation():
    d = pd.read_csv(inp(FIG / "ablation_significance.csv")).set_index("variant")
    full = d.loc["abl_full"]
    lab = [("abl_no_filter", "Tắt bộ lọc accessible premise"), ("abl_no_bm25", "Bỏ điểm và hạng BM25"),
           ("abl_no_dependency", "Bỏ độ phổ biến và PageRank"), ("abl_no_symbols", "Bỏ độ trùng ký hiệu"),
           ("abl_no_kind", "Bỏ loại khai báo"), ("abl_no_location", "Bỏ cùng tệp, cùng thư mục"),
           ("abl_no_namespace", "Bỏ namespace"), (None, None), ("abl_k50", "50 candidate thay vì 100"),
           ("abl_k200", "200 candidate thay vì 100")]
    body = ["Thay đổi so với V4 & $\\Delta$Recall@10 [KTC 95\\%] & $p$ Holm & $\\Delta$MRR [KTC 95\\%] & $p$ Holm \\\\",
            "\\midrule"]
    for k, l in lab:
        if k is None:
            body.append("\\midrule")
            continue
        r = d.loc[k]
        body.append(f"{l} & {sg(r['d_R@10'])} \\ {ci(r['lo_R@10'], r['hi_R@10'])} & {p(r['pholm_R@10'])} & "
                    f"{sg(r['d_MRR'])} \\ {ci(r['lo_MRR'], r['hi_MRR'])} & {p(r['pholm_MRR'])} \\\\")
    write("ablation", "Ablation của reranker V4 trên tập test \\code{random}", "tab:ablation", body, "@{}lllll@{}",
          note=f"\\num{{2641}} truy vấn; V4 đầy đủ có Recall@10 = {f(full['R@10'])}, MRR = {f(full['MRR'])}. $\\Delta$ là "
          "biến thể trừ V4. Khoảng tin cậy từ bootstrap theo cụm theorem; $p$ hai phía từ kiểm định hoán vị ghép cặp theo "
          "cụm theorem, hiệu chỉnh Holm trên 9 phép so sánh của từng metric.", sources= "results/figures/ablation_significance.csv (scripts/ablation_significance.py)",
          short="Ablation của reranker", sep="4pt")


def main() -> None:
    for fn in (t_dataflow, t_quality, t_signif, t_robust, t_unseen, t_controls, t_candidates, t_v5grid, t_leandojo, t_shared, t_lucene, t_v5load,
               t_workers, t_build, t_engines, t_profile, t_errors, t_ablation):
        fn()
    lines = ["# Nguồn của từng bảng (sinh bởi scripts/make_report_tables.py)", "",
             "Mọi bảng số liệu của báo cáo được sinh từ các tệp dưới đây; không sửa tay tệp .tex trong thư mục này.", "",
             "| Tệp bảng | Nguồn |", "|---|---|"] + [f"| `tables/{a}.tex` | {b} |" for a, b in SOURCES] + \
            ["| `tables/example.tex` | scripts/running_example.py (top 100 của các lần chạy validation đã ghim) |"]
    man = manifest()
    lines += ["", "## Lần chạy đánh giá đã ghim (results/figures/report_manifest.json)", "", "| Hệ thống và tập | run_id |",
              "|---|---|"] + [f"| {k} | `{v['run_id']}` |" for k, v in man.items()]
    lines += ["", "## Tệp đầu vào và SHA-256 tại thời điểm sinh bảng", "", "| Tệp | SHA-256 |", "|---|---|"]
    for x in sorted(INPUTS):
        lines.append(f"| `{x.relative_to(ROOT).as_posix()}` | `{file_sha256(x)}` |")
    (OUT / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(SOURCES)} bảng trong {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
