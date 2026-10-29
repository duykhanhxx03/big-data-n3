"""20 truy vấn kiểm tra (tuần 1): kiểm tra tay định nghĩa truy vấn / premise đúng trong benchmark_spec.

Chọn 20 truy vấn random/val (seed 42), mỗi nhóm thư mục Mathlib/<lĩnh vực> nhiều nhất 2 truy vấn, có cả truy vấn
nhiều premise đúng và truy vấn bị bỏ bớt premise đúng vì không có trong corpus. Với mỗi truy vấn, đọc lại tactic gốc
trong JSON của LeanDojo (tactic không được đưa vào queries.parquet) và kiểm tra tự động:
  C1  mọi premise đúng xuất hiện (được đánh dấu <a>) trong tactic gốc
  C2  mọi premise đúng truy cập được từ theorem (qua import hoặc khai báo xong trước nó trong cùng file)
  C3  theorem đang chứng minh không nằm trong premise đúng
  C4  premise chú thích nhưng không có trong corpus được ghi vào dead_letter, không lẳng lặng mất
Cuối cùng ghi hạng của premise đúng theo B0 và V4 (val được phép dùng để xem).

Dùng:  python scripts/make_check_queries.py
Ra:    docs/check_queries.md (có cột để người kiểm tra ký xác nhận), data_sample/check_queries.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval import run_eval  # noqa: E402
from retrieval.rerank import get_meta  # noqa: E402
from retrieval.search import InvertedIndex  # noqa: E402

RAW = ROOT / "data" / "raw" / "leandojo_benchmark_4_v10" / "leandojo_benchmark_4"
PROCESSED = ROOT / "data" / "processed"
N, SEED, PER_AREA = 20, 42, 2


def area(path: str) -> str:
    parts = path.split("/")
    if parts[0] == ".lake" and len(parts) > 2:      # .lake/packages/<thư viện>/...
        return parts[2]
    return "/".join(parts[:2]) if parts[0] == "Mathlib" and len(parts) > 2 else parts[0]


def pick(q: pd.DataFrame, dropped: set[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    q = q.assign(area=q.file_path.map(area), partial=q.query_id.isin(dropped))
    q = q.iloc[rng.permutation(len(q))]
    chosen: list[int] = []
    # Bắt buộc có: 3 truy vấn nhiều premise đúng, 2 truy vấn bị bỏ bớt premise đúng.
    for cond, need in ((q.n_gold >= 3, 3), (q.partial, 2)):
        chosen += [i for i in q[cond].index if i not in chosen][:need]
    counts = q.loc[chosen].area.value_counts().to_dict()
    for i, a in zip(q.index, q.area):
        if len(chosen) >= N:
            break
        if i not in chosen and counts.get(a, 0) < PER_AREA:
            chosen.append(i)
            counts[a] = counts.get(a, 0) + 1
    return q.loc[chosen].sort_values("query_id", key=lambda s: s.map(lambda x: tuple(map(int, x.split(":")[2:]))))


def ranks(searcher, text: str, ctx: dict, gold: set[int]) -> dict[int, int | None]:
    hits = [p for p, _ in searcher.search(text, k=100, ctx=ctx)]
    pos = {p: r for r, p in enumerate(hits, 1)}
    return {g: pos.get(g) for g in gold}


def main() -> None:
    q = pq.read_table(PROCESSED / "queries.parquet").to_pandas()
    q = q[(q.split_strategy == "random") & (q.split == "val")]
    dead = pq.read_table(PROCESSED / "dead_letter.parquet").to_pandas()
    dead_gold = dead[(dead.entity == "gold") & dead.key.str.startswith("random:val:")]
    dropped_by_q: dict[str, list[str]] = {}
    for k in dead_gold.key:
        a, b, c, d, name = k.split(":", 4)
        dropped_by_q.setdefault(f"{a}:{b}:{c}:{d}", []).append(name)
    sel = pick(q, set(dropped_by_q))

    raw = json.loads((RAW / "random" / "val.json").read_text(encoding="utf-8"))
    prem = pq.read_table(PROCESSED / "premises.parquet",
                         columns=["premise_id", "full_name", "path", "decl_kind", "statement"]).to_pandas()
    assert (prem.premise_id.to_numpy() == np.arange(len(prem))).all()
    qrels = pq.read_table(PROCESSED / "qrels.parquet").to_pandas()
    gold_of = qrels[qrels.query_id.isin(set(sel.query_id))].groupby("query_id").premise_id.apply(set).to_dict()
    meta = get_meta("random")
    searchers = {}
    for name in ("b0_bm25_default", "v4_rerank_dep"):
        cfg = run_eval.load_config(ROOT / "configs" / f"{name}.yaml")
        searchers[cfg["name"]] = run_eval.make_searcher(cfg, InvertedIndex(run_eval.ensure_index(cfg["index"], "full")))

    out, md = [], []
    for row in sel.itertuples():
        thm = raw[row.theorem_idx]
        tac = thm["traced_tactics"][row.tactic_idx]
        assert thm["full_name"] == row.theorem_full_name and tac["state_before"] == row.state_before
        marked = re.findall(r"<a>(.*?)</a>", tac["annotated_tactic"][0])
        ann = [a["full_name"] for a in tac["annotated_tactic"][1]]
        gold = gold_of[row.query_id]
        gnames = {g: prem.full_name[g] for g in gold}
        ctx = {"theorem_full_name": row.theorem_full_name, "file_path": row.file_path,
               "theorem_start_line": int(row.theorem_start_line)}
        access = meta.accessible_mask(row.file_path, int(row.theorem_start_line))
        c1 = all(any(gnames[g] == m or gnames[g].endswith("." + m) for m in marked) for g in gold)
        c2 = bool(access[list(gold)].all())
        c3 = row.theorem_full_name not in gnames.values()
        dropped = dropped_by_q.get(row.query_id, [])
        c4 = sorted(set(ann) - set(gnames.values())) == sorted(set(dropped))
        rk = {s: ranks(sr, row.state_before, ctx, gold) for s, sr in searchers.items()}
        rec = {"query_id": row.query_id, "theorem_full_name": row.theorem_full_name, "file_path": row.file_path,
               "area": row.area, "state_before": row.state_before,
               "tactic_annotated": tac["annotated_tactic"][0], "annotated_premises": ann,
               "gold_premise_ids": sorted(gold), "gold_full_names": [gnames[g] for g in sorted(gold)],
               "dropped_not_in_corpus": dropped, "checks": {"C1": c1, "C2": c2, "C3": c3, "C4": c4},
               "ranks": {s: {gnames[g]: r for g, r in v.items()} for s, v in rk.items()}}
        out.append(rec)

        goal = row.state_before.strip()
        goal = goal if len(goal) <= 400 else goal[:400] + " …"
        md += [f"### {len(out)}. `{row.query_id}` — {row.area}", "",
               f"- Theorem: `{row.theorem_full_name}` (`{row.file_path}`, dòng {row.theorem_start_line})",
               "- Tactic gốc (chỉ để kiểm tra, không đưa vào truy vấn):", "", "```lean", tac["annotated_tactic"][0], "```", "",
               "- Proof state (truy vấn):", "", "```lean", goal, "```", "",
               "- Premise đúng:"]
        for g in sorted(gold):
            p = prem.loc[g]
            st = " ".join(p.statement.split())
            st = st if len(st) <= 160 else st[:160] + " …"
            md.append(f"  - `{p.full_name}` ({p.decl_kind}, `{p.path}`) — hạng B0: {rk['B0'][g] or '>100'}, "
                      f"V4: {rk['V4'][g] or '>100'}  \n    `{st.replace('`', 'ˋ')}`")
        if dropped:
            md.append(f"- Chú thích nhưng không có trong corpus (đã ghi dead_letter): "
                      + ", ".join(f"`{d}`" for d in dropped))
        md += [f"- Kiểm tra tự động: C1 {'✓' if c1 else '✗'} · C2 {'✓' if c2 else '✗'} · "
               f"C3 {'✓' if c3 else '✗'} · C4 {'✓' if c4 else '✗'}",
               "- Người kiểm tra: ______ · Kết luận: ☐ đúng định nghĩa ☐ có vấn đề (ghi rõ): ______", ""]

    (ROOT / "data_sample" / "check_queries.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    n_pass = {c: sum(r["checks"][c] for r in out) for c in ("C1", "C2", "C3", "C4")}
    found = {s: sum(any(v is not None and v <= 10 for v in r["ranks"][s].values()) for r in out) for s in searchers}
    head = ["# 20 truy vấn kiểm tra (sinh bởi scripts/make_check_queries.py)", "",
            "Mục đích: kiểm tra bằng mắt định nghĩa truy vấn và premise đúng trong `docs/benchmark_spec.md` trước khi "
            "chạy thí nghiệm. 20 truy vấn lấy từ `random/val` (seed 42), mỗi lĩnh vực nhiều nhất 2 truy vấn, bắt buộc "
            "có truy vấn nhiều premise đúng và truy vấn bị bỏ bớt premise đúng. Dữ liệu máy đọc: "
            "`data_sample/check_queries.jsonl`.", "",
            "Kiểm tra tự động:", "",
            "- **C1** mọi premise đúng được đánh dấu `<a>` trong tactic gốc",
            "- **C2** mọi premise đúng truy cập được từ theorem (import hoặc khai báo trước trong cùng file)",
            "- **C3** theorem đang chứng minh không phải premise đúng",
            "- **C4** premise chú thích mà không có trong corpus đều nằm trong `dead_letter.parquet`", "",
            "| Kiểm tra | C1 | C2 | C3 | C4 |", "|---|---|---|---|---|",
            f"| Số truy vấn đạt / {len(out)} | " + " | ".join(str(n_pass[c]) for c in ("C1", "C2", "C3", "C4")) + " |", "",
            f"Có ít nhất một premise đúng trong top-10: B0 {found['B0']}/{len(out)}, V4 {found['V4']}/{len(out)}.", "",
            "Người kiểm tra điền hai dòng cuối mỗi truy vấn. Mọi truy vấn đánh dấu *có vấn đề* phải mở issue và, nếu "
            "đổi định nghĩa, nâng phiên bản benchmark spec.", ""]
    (ROOT / "docs" / "check_queries.md").write_text("\n".join(head + md), encoding="utf-8")
    print("\n".join(head))


if __name__ == "__main__":
    main()
