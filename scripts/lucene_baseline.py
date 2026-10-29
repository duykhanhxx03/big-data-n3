"""Baseline BM25 của Lucene (qua Anserini) để đối chiếu với BM25 tự cài đặt (B0).

Hai biến thể, cùng chỉ đánh chỉ mục trường statement như B0:
  lucene-default   văn bản gốc, bộ phân tích tiếng Anh mặc định của Anserini (tách từ, bỏ stopword, Porter stemming),
                   BM25 mặc định của Anserini (k1 = 0,9; b = 0,4). Đây là kết quả "dùng Lucene nguyên bản".
  lucene-english-b0params  như lucene-default nhưng k1 = 1,2; b = 0,75: so với lucene-default tách riêng tác động của
                   tham số, so với lucene-b0tokens tách riêng tác động của bộ phân tích.
  lucene-b0tokens  văn bản đã tách bằng đúng tokenizer của B0 (-pretokenized), BM25 k1 = 1,2; b = 0,75 như B0.
                   Biến thể này kiểm tra cài đặt BM25 của nhóm có tương đương Lucene không và so tốc độ cùng điều kiện.
Truy vấn: toàn bộ random_test_all; lấy top 100; tìm kiếm một luồng để đo thời gian trung bình mỗi truy vấn.
B0 được đo lại trong cùng script theo cùng quy tắc (một tiến trình, một lượt, bỏ 100 truy vấn đầu), vì độ trễ ghi trong
lần đánh giá đã ghim có thể đo khi nhiều đánh giá chạy song song.
Lucene lưu độ dài tài liệu dạng nén 1 byte nên điểm và thứ hạng có thể lệch nhẹ so với cài đặt chính xác.

Cần Java 21 và tệp jar của Anserini (mặc định tools/anserini/anserini-2.3.0-fatjar.jar trong kho, đổi bằng --jar).
Dùng:  python scripts/lucene_baseline.py [--eval random_test_all] [--b0-only]
Ra:    results/raw/lucene/*, results/figures/lucene.md, run record kind "lucene"
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import runlog  # noqa: E402
from analysis.tokenizer import get_tokenizer  # noqa: E402
from eval import metrics  # noqa: E402
from eval.run_eval import ensure_index, load_config, load_eval, make_searcher, query_contexts  # noqa: E402
from retrieval.search import InvertedIndex  # noqa: E402

OUT = ROOT / "results" / "raw" / "lucene"
VARIANTS = {
    "lucene-default": {"pretok": False, "k1": 0.9, "b": 0.4},
    "lucene-english-b0params": {"pretok": False, "k1": 1.2, "b": 0.75},
    "lucene-b0tokens": {"pretok": True, "k1": 1.2, "b": 0.75},
}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def java(jar: Path, cls: str, args: list[str], log: Path) -> tuple[float, str]:
    t0 = time.perf_counter()
    p = subprocess.run(["java", "-cp", str(jar), cls, *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    wall = time.perf_counter() - t0
    log.write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
    if p.returncode != 0:
        raise SystemExit(f"{cls} lỗi, xem {log}")
    return wall, p.stdout + p.stderr


def time_b0(queries, eval_name: str, warmup: int = 100) -> dict:
    """B0 trong một tiến trình, một lượt qua các truy vấn; trung bình từ truy vấn thứ warmup + 1 như với Lucene."""
    cfg = load_config(ROOT / "configs" / "b0_bm25_default.yaml")
    searcher = make_searcher(cfg, InvertedIndex(ensure_index(cfg["index"])))
    lat = []
    for text, ctx in zip(queries.state_before, query_contexts(queries)):
        t = time.perf_counter()
        searcher.search(text, k=100, ctx=ctx)
        lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat)
    row = {"variant": "b0-inprocess", "ms_per_query": round(float(lat[warmup:].mean()), 2),
           "ms_per_query_all": round(float(lat.mean()), 2), "ms_p50": round(float(np.median(lat[warmup:])), 2),
           "n_queries": len(lat)}
    runlog.write_record("lucene", parameters={"variant": "b0-inprocess", "eval_set": eval_name, "config": cfg["name"]},
                        metrics=row, seed=None)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jar", default=str(ROOT / "tools" / "anserini" / "anserini-2.3.0-fatjar.jar"))
    ap.add_argument("--eval", default="random_test_all")
    ap.add_argument("--b0-only", action="store_true", help="chỉ đo lại B0, dùng kết quả Lucene đã ghi")
    args = ap.parse_args()
    jar = Path(args.jar)
    OUT.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer("default")
    prem = pq.read_table(ROOT / "data" / "processed" / "premises.parquet", columns=["premise_id", "statement"]).to_pandas()
    queries, gold = load_eval(args.eval)
    man = json.loads((ROOT / "results" / "figures" / "report_manifest.json").read_text(encoding="utf-8"))
    b0_entry = man[f"B0|{args.eval}"]
    b0_run: dict[str, list[int]] = {}
    for line in (ROOT / b0_entry["run_file"]).read_text(encoding="utf-8").splitlines()[1:]:
        qid, pid, *_ = line.split("\t")
        b0_run.setdefault(qid, []).append(int(pid))

    rows = []
    if args.b0_only:   # lấy bản ghi Lucene gần nhất của mỗi biến thể
        recs = {}
        for f in sorted((ROOT / "results" / "raw" / "lucene").glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r["parameters"]["variant"] in VARIANTS and r["parameters"].get("eval_set") == args.eval:
                recs[r["parameters"]["variant"]] = r["metrics"]
        rows = [recs[k] for k in VARIANTS if k in recs]
    for name, v in ({} if args.b0_only else VARIANTS).items():
        work = OUT / name
        if work.exists():
            shutil.rmtree(work)
        (work / "docs").mkdir(parents=True)
        with (work / "docs" / "docs.jsonl").open("w", encoding="utf-8") as fh:
            for pid, st in zip(prem.premise_id, prem.statement):
                text = " ".join(tok(st or "")) if v["pretok"] else clean(st)
                fh.write(json.dumps({"id": str(pid), "contents": text}, ensure_ascii=False) + "\n")
        with (work / "topics.tsv").open("w", encoding="utf-8") as fh:
            for qid, text in zip(queries.query_id, queries.state_before):
                q = " ".join(tok(text)) if v["pretok"] else clean(text)
                fh.write(f"{qid}\t{q}\n")
        idx_args = ["-collection", "JsonCollection", "-input", str(work / "docs"), "-index", str(work / "index"),
                    "-generator", "DefaultLuceneDocumentGenerator", "-threads", "6"]
        t_index, _ = java(jar, "io.anserini.index.IndexCollection", idx_args + (["-pretokenized"] if v["pretok"] else []),
                          work / "index.log")
        index_bytes = sum(p.stat().st_size for p in (work / "index").glob("*") if p.is_file())
        search_args = ["-index", str(work / "index"), "-topics", str(work / "topics.tsv"), "-topicReader", "TsvString",
                       "-output", str(work / "run.txt"), "-bm25", "-bm25.k1", str(v["k1"]), "-bm25.b", str(v["b"]),
                       "-hits", "100", "-threads", "1"]
        t_search, log = java(jar, "io.anserini.search.SearchCollection",
                             search_args + (["-pretokenized"] if v["pretok"] else []), work / "search.log")
        # Thời gian mỗi truy vấn ở trạng thái ổn định: Anserini ghi mốc sau mỗi 100 truy vấn, lấy khoảng giữa mốc đầu
        # và mốc cuối (bỏ khởi động JVM, mở index và 100 truy vấn làm nóng đầu tiên). Wall time đầy đủ vẫn được ghi lại.
        marks = [(int(n), hh * 3600 + mm * 60 + ss + ms / 1000) for hh, mm, ss, ms, n in
                 ((int(a), int(b), int(c), int(d), e) for a, b, c, d, e in
                  re.findall(r"(\d+):(\d+):(\d+),(\d+) INFO.*?(\d+) queries processed", log))]
        if len(marks) >= 2:
            ms_q = 1000 * (marks[-1][1] - marks[0][1]) / (marks[-1][0] - marks[0][0])
        else:
            ms_q = 1000 * t_search / len(queries)
        ranked: dict[str, list[int]] = {}
        for line in (work / "run.txt").read_text(encoding="utf-8").splitlines():
            qid, _, doc, rank, *_ = line.split()
            ranked.setdefault(qid, []).append(int(doc))
        per = [metrics.per_query(ranked.get(qid, []), gold[qid]) for qid in queries.query_id]
        agg = metrics.aggregate(per)
        # Độ trùng với B0 trên từng truy vấn: tỉ lệ phần tử chung của hai top 10, top 10 trùng nguyên tập, top 1 trùng
        ov = [len(set(ranked.get(qid, [])[:10]) & set(b0_run.get(qid, [])[:10])) / 10 for qid in queries.query_id]
        same10 = [set(ranked.get(qid, [])[:10]) == set(b0_run.get(qid, [])[:10]) for qid in queries.query_id]
        same1 = [ranked.get(qid, [None])[:1] == b0_run.get(qid, [None])[:1] for qid in queries.query_id]
        row = {"variant": name, **{k: agg[k] for k in ("R@1", "R@10", "R@50", "MRR", "nDCG@10")},
               "index_s": round(t_index, 1), "index_mb": round(index_bytes / 2**20, 1),
               "search_wall_s": round(t_search, 1), "ms_per_query_with_startup": round(1000 * t_search / len(queries), 2),
               "ms_per_query": round(ms_q, 2), "k1": v["k1"], "b": v["b"],
               "top10_overlap_b0": float(np.mean(ov)), "same_top10_b0": float(np.mean(same10)),
               "same_top1_b0": float(np.mean(same1)),
               "n_queries": len(queries)}
        rows.append(row)
        runlog.write_record("lucene", parameters={"variant": name, "eval_set": args.eval, "jar": jar.name, **v},
                            metrics=row, seed=None, log_path=str((work / "search.log").relative_to(ROOT)).replace("\\", "/"))
        shutil.rmtree(work / "docs")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    # B0 của nhóm trên cùng tập: chất lượng từ lần chạy đã ghim, thời gian đo lại một luồng ở trên
    b0 = json.loads((ROOT / "results" / "raw" / "eval" / f"{b0_entry['run_id']}.json").read_text(encoding="utf-8"))
    b0t = time_b0(queries, args.eval)
    lines = ["# BM25 của Lucene (Anserini 2.3.0) so với BM25 tự cài đặt (sinh bởi scripts/lucene_baseline.py)", "",
             f"Tập {args.eval}, {len(queries)} truy vấn, top 100, trường statement. Thời gian tìm kiếm một luồng, "
             "trung bình mỗi truy vấn ở trạng thái ổn định (Lucene: từ mốc 100 truy vấn đến mốc cuối trong log của "
             "Anserini; trong ngoặc là wall time chia số truy vấn, gồm khởi động JVM và mở index).", "",
             "| Hệ thống | R@1 | R@10 | R@50 | MRR | ms/truy vấn | Chỉ mục (MB) | Trùng top-10 với B0 | "
             "Top-10 trùng nguyên tập | Top-1 trùng |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} (k1 = {r['k1']}, b = {r['b']}) | {r['R@1']:.3f} | {r['R@10']:.3f} | "
                     f"{r['R@50']:.3f} | {r['MRR']:.3f} | {r['ms_per_query']:.2f} ({r['ms_per_query_with_startup']:.2f}) | "
                     f"{r['index_mb']:.1f} | {r['top10_overlap_b0']:.3f} | {r['same_top10_b0']:.3f} | "
                     f"{r['same_top1_b0']:.3f} |")
    if b0:
        m = b0["metrics"]
        lines.append(f"| B0 của nhóm (k1 = 1.2, b = 0.75) | {m['R@1']:.3f} | {m['R@10']:.3f} | {m['R@50']:.3f} | "
                     f"{m['MRR']:.3f} | {b0t['ms_per_query']:.2f} ({b0t['ms_per_query_all']:.2f}) | "
                     f"{b0['index']['postings_bytes'] / 2**20:.1f} (postings) | 1 | 1 | 1 |")
    (ROOT / "results" / "figures" / "lucene.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
