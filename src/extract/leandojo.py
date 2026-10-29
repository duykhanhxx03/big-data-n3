"""Trích LeanDojo Benchmark 4 sang các bảng Parquet theo data contract (docs/benchmark_spec.md).

Đầu ra trong data/processed/:
- premises.parquet   premise_id, full_name, name, path, namespace, short_name, decl_kind, statement, code, library, start_line, end_line
- queries.parquet    query_id, split_strategy, split, theorem_full_name, file_path, theorem_idx, theorem_start_line,
                     tactic_idx, state_before, n_gold
- qrels.parquet      query_id, premise_id
- files.parquet      path, imports (để tính premise truy cập được nếu cần)
- dead_letter.parquet  entity, key, reason (bản ghi bị loại và lý do)
- extract_stats.json

Định danh premise = (full_name, path): có 57 tên trùng giữa Lean core và Mathlib.
premise_id = thứ tự của (full_name, path) sau khi sắp xếp, nên ổn định với cùng snapshot.
Không lưu text của tactic vào queries để tránh rò rỉ tên premise.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

STRATEGIES = ("random", "novel_premises")
SPLITS = ("train", "val", "test")
_OPEN = "([{⟨⦃"
_CLOSE = ")]}⟩⦄"
_MODIFIERS = {"private", "protected", "noncomputable", "nonrec", "partial", "unsafe", "scoped", "local"}
_DECL_RE = re.compile(r"\b(theorem|lemma|def|instance|abbrev|structure|class|inductive|opaque|irreducible_def|axiom|example)\b")


def _strip_attributes(code: str) -> str:
    """Bỏ docstring /-- -/ và các khối @[...] ở đầu (có thể lồng ngoặc)."""
    s = code.lstrip()
    while True:
        if s.startswith("/--"):
            end = s.find("-/")
            s = s[end + 2:].lstrip() if end >= 0 else ""
            continue
        if s.startswith("@["):
            depth, i = 0, 1
            while i < len(s):
                if s[i] == "[":
                    depth += 1
                elif s[i] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            s = s[i + 1:].lstrip()
            continue
        return s


def decl_kind(code: str) -> str:
    m = _DECL_RE.search(_strip_attributes(code))
    return m.group(1) if m else "other"


def statement_of(code: str) -> str:
    """Phần phát biểu/chữ ký: cắt ở ':=' , 'where' hoặc nhánh '|' đầu tiên ở độ sâu ngoặc 0."""
    s = _strip_attributes(code)
    depth, i, n = 0, 0, len(s)
    while i < n:
        ch = s[i]
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth = max(0, depth - 1)
        elif depth == 0:
            if s.startswith(":=", i):
                return s[:i].rstrip()
            if s.startswith("where", i) and (i == 0 or s[i - 1].isspace()) and (i + 5 == n or not s[i + 5].isalnum()):
                return s[:i].rstrip()
            if ch == "|":
                line_start = s.rfind("\n", 0, i)
                if line_start >= 0 and not s[line_start + 1:i].strip():  # nhánh pattern matching đầu dòng
                    return s[:i].rstrip()
        i += 1
    return s.rstrip()


def library_of(path: str) -> str:
    if path.startswith(".lake/packages/"):
        return path.split("/")[2]
    return path.split("/")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Trích LeanDojo Benchmark 4 sang Parquet")
    ap.add_argument("--src", default="data/raw/leandojo_benchmark_4_v10/leandojo_benchmark_4")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    dead: list[dict] = []

    # ---- premises ----
    rows, files = {}, {"path": [], "imports": []}
    with (src / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            files["path"].append(d["path"])
            files["imports"].append(d["imports"])
            for p in d["premises"]:
                key = (p["full_name"], d["path"])
                if key in rows:
                    dead.append({"entity": "premise", "key": f"{key[0]}@{key[1]}", "reason": "duplicate_full_name_and_path"})
                    continue
                rows[key] = (p, d["path"])
    keys = sorted(rows)
    pid_of = {k: i for i, k in enumerate(keys)}
    by_name: dict[str, list[int]] = {}
    for (name, _), i in pid_of.items():
        by_name.setdefault(name, []).append(i)
    prem = {c: [] for c in ("premise_id", "full_name", "name", "path", "namespace", "short_name", "decl_kind", "statement",
                            "code", "library", "start_line", "end_line")}
    for i, (name, path) in enumerate(keys):
        p, _ = rows[(name, path)]
        ns, _, short = name.rpartition(".")
        prem["premise_id"].append(i)
        prem["full_name"].append(name)
        prem["name"].append(name)  # field "name" cho index (cùng giá trị full_name)
        prem["path"].append(path)
        prem["namespace"].append(ns)
        prem["short_name"].append(short)
        prem["decl_kind"].append(decl_kind(p["code"]))
        prem["statement"].append(statement_of(p["code"]))
        prem["code"].append(p["code"])
        prem["library"].append(library_of(path))
        prem["start_line"].append(p["start"][0])
        prem["end_line"].append(p["end"][0])
    pq.write_table(pa.table(prem), out / "premises.parquet", compression="zstd")
    pq.write_table(pa.table(files), out / "files.parquet", compression="zstd")
    stats = {"n_premises": len(keys), "n_files": len(files["path"]),
             "decl_kind": dict(Counter(prem["decl_kind"]).most_common()),
             "library": dict(Counter(prem["library"]).most_common())}
    print(f"premises: {len(keys)} ({time.perf_counter() - t0:.1f}s)")

    # ---- queries + qrels ----
    q = {c: [] for c in ("query_id", "split_strategy", "split", "theorem_full_name", "file_path", "theorem_idx",
                         "theorem_start_line", "tactic_idx", "state_before", "n_gold")}
    qrels = {"query_id": [], "premise_id": []}
    split_stats = {}
    for strategy in args.strategies.split(","):
        for split in SPLITS:
            data = json.loads((src / strategy / f"{split}.json").read_text(encoding="utf-8"))
            c = Counter(theorems=len(data))
            for ti, thm in enumerate(data):
                for ki, tac in enumerate(thm["traced_tactics"]):
                    c["tactics"] += 1
                    ann = tac["annotated_tactic"][1]
                    if not ann:
                        c["tactics_without_premise"] += 1
                        continue
                    qid = f"{strategy}:{split}:{ti}:{ki}"
                    gold = set()
                    for a in ann:
                        pid = pid_of.get((a["full_name"], a["def_path"]))
                        if pid is None and len(by_name.get(a["full_name"], [])) == 1:
                            pid = by_name[a["full_name"]][0]
                        if pid is None:
                            c["gold_not_in_corpus"] += 1
                            dead.append({"entity": "gold", "key": f"{qid}:{a['full_name']}", "reason": "premise_not_in_corpus"})
                        else:
                            gold.add(pid)
                    if not gold:
                        c["queries_dropped_no_gold"] += 1
                        dead.append({"entity": "query", "key": qid, "reason": "no_gold_in_corpus"})
                        continue
                    q["query_id"].append(qid)
                    q["split_strategy"].append(strategy)
                    q["split"].append(split)
                    q["theorem_full_name"].append(thm["full_name"])
                    q["file_path"].append(thm["file_path"])
                    q["theorem_idx"].append(ti)
                    q["theorem_start_line"].append(thm["start"][0])
                    q["tactic_idx"].append(ki)
                    q["state_before"].append(tac["state_before"])
                    q["n_gold"].append(len(gold))
                    for pid in sorted(gold):
                        qrels["query_id"].append(qid)
                        qrels["premise_id"].append(pid)
                    c["queries"] += 1
            split_stats[f"{strategy}/{split}"] = dict(c)
            print(strategy, split, dict(c), f"({time.perf_counter() - t0:.1f}s)")
            del data
    pq.write_table(pa.table(q), out / "queries.parquet", compression="zstd")
    pq.write_table(pa.table(qrels), out / "qrels.parquet", compression="zstd")
    pq.write_table(pa.table({k: [r[k] for r in dead] for k in ("entity", "key", "reason")}),
                   out / "dead_letter.parquet", compression="zstd")
    stats["splits"] = split_stats
    stats["dead_letter"] = dict(Counter(r["reason"] for r in dead))
    stats["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (out / "extract_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats["dead_letter"]), f"tổng {stats['elapsed_s']}s")


if __name__ == "__main__":
    main()
