"""Đánh giá các biến thể N3 trên benchmark chung N3/N4 (gói n4_shared) và chấm run file của nhóm khác trên cùng tập.

Benchmark: 898 theorem của random/test, truy vấn = statement của theorem, premise định danh bằng full_name
(docs/n4_shared_benchmark.md). Khác benchmark chính của báo cáo (truy vấn mức tactic, docs/benchmark_spec.md), nên số
liệu ở đây không so trực tiếp với bảng trong báo cáo.

- Danh sách xếp hạng của N3 (180.957 premise theo cặp full_name, path) được đổi sang full_name, giữ vị trí đầu của mỗi
  tên lặp.
- Quy tắc eligible (shared.eligible): loại chính theorem truy vấn trước khi cắt top-k, áp dụng như nhau cho mọi hệ
  thống, kể cả run file ngoài. V4 và V5 còn lọc premise truy cập được (một phần phương pháp của N3), dùng file và dòng
  bắt đầu của theorem lấy từ random/test.json; B0 và V2 chỉ dùng statement.
- 16 truy vấn không còn premise đúng nào trong corpus không có Recall xác định: được đếm và báo riêng, không đưa vào
  trung bình. Run file ngoài thiếu truy vấn nào thì truy vấn đó nhận danh sách rỗng (metric bằng 0).
- Mỗi metric báo trên hai lát: "all" (882 truy vấn chấm được) và "signature" (bỏ truy vấn mà statement còn chứa chứng
  minh, shared.has_proof_body; chứng minh thường gọi thẳng tên premise đúng).
- Trộn (--fuse "A+B"): Reciprocal Rank Fusion như scripts/fuse_rrf.py, điểm = Σ 1/(k + hạng) với k = 60, trọng số bằng
  nhau, hòa điểm thì full_name nhỏ trước, giữ top-k. Hạng lấy từ danh sách đã qua quy tắc eligible. Gói chung chỉ có tập
  test nên k và trọng số không được tinh chỉnh. Có run file ngoài mà không ghi --fuse thì mỗi run ngoài được trộn với V4
  (cấu hình N3 nên dùng chung).
- Kiểm định: hoán vị ghép cặp trên R@10 (mỗi truy vấn là một theorem nên cụm là truy vấn), hiệu chỉnh Holm trong mỗi
  lát; khoảng tin cậy 95% của chênh lệch bằng paired bootstrap. Họ so sánh: các bước liên tiếp của N3, mỗi run ngoài với
  từng biến thể N3, và mỗi hệ trộn với từng thành phần của nó.

Dùng:  python scripts/eval_shared.py                                     # B0, V2, V4, V5
       python scripts/eval_shared.py --external N4-dense=path/run_dense_full.csv   # chấm thêm run file khác, trộn với V4
       python scripts/eval_shared.py --external N4-dense=... --fuse V4+N4-dense V5+N4-dense
Run file ngoài: CSV hoặc TSV có cột query_id, premise_id (full_name); có cột rank thì xếp theo rank, không thì theo thứ
tự dòng.
Ra:    results/runs/shared/<hệ thống>.tsv, results/raw/shared/<run_id>.json và .perquery.parquet,
       results/figures/shared_benchmark.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import runlog  # noqa: E402
from eval import metrics  # noqa: E402
from eval.run_eval import PROCESSED, ensure_index, for_strategy, load_config, make_searcher  # noqa: E402
from retrieval.search import InvertedIndex  # noqa: E402
from shared import DATA_VERSION, eligible, has_proof_body, theorem_id, verify_archive, verify_directory  # noqa: E402

DEFAULT_CONFIGS = ["configs/b0_bm25_default.yaml", "configs/v2_bm25f_lean.yaml", "configs/v4_rerank_dep.yaml",
                   "configs/v5_rerank_expand.yaml"]
LEANDOJO = ROOT / "data" / "raw" / "leandojo_benchmark_4_v10" / "leandojo_benchmark_4"
METRICS = ["R@1", "R@5", "R@10", "R@50", "MRR", "nDCG@10"]
SLICES = ["all", "signature"]


def load_queries(bundle: Path) -> tuple[list[dict], dict]:
    """Kiểm tra SHA-256 và schema của gói rồi đọc queries.jsonl."""
    report = verify_archive(bundle) if bundle.is_file() else verify_directory(bundle)
    if bundle.is_file():
        with zipfile.ZipFile(bundle) as z:
            (name,) = [n for n in z.namelist() if Path(n).name == "queries.jsonl"]
            text = z.read(name).decode("utf-8")
    else:
        text = (bundle / "queries.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in io.StringIO(text) if line.strip()], report


def theorem_contexts() -> dict[str, dict]:
    """Ngữ cảnh lúc truy vấn của từng theorem random/test: tên, file, dòng bắt đầu (như query_contexts của run_eval)."""
    data = json.loads((LEANDOJO / "random" / "test.json").read_text(encoding="utf-8"))
    return {t["full_name"]: {"theorem_full_name": t["full_name"], "file_path": t["file_path"],
                             "theorem_start_line": int(t["start"][0])} for t in data}


def eligible_ranking(pairs, query_id: str, k: int) -> list[tuple[str, float | None]]:
    """Bỏ tên lặp (giữ vị trí đầu) và chính theorem truy vấn, rồi cắt top-k."""
    out, seen = [], set()
    for name, score in pairs:
        if name in seen or not eligible(name, query_id):
            continue
        seen.add(name)
        out.append((name, score))
        if len(out) == k:
            break
    return out


def run_n3(cfg: dict, queries: list[dict], full_name_of: np.ndarray, ctxs: dict, k: int):
    cfg = for_strategy(cfg, "random")
    searcher = make_searcher(cfg, InvertedIndex(ensure_index(cfg["index"])))
    rankings, lat = {}, {}
    for q in queries:
        t = time.perf_counter()
        hits = searcher.search(q["statement"], k=k + 50, ctx=ctxs.get(theorem_id(q["id"])))
        lat[q["id"]] = (time.perf_counter() - t) * 1000
        rankings[q["id"]] = eligible_ranking(((full_name_of[p], s) for p, s in hits), q["id"], k)
    return rankings, lat


def read_external(path: Path, queries: list[dict], k: int) -> tuple[dict, dict]:
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t" if "\t" in text.split("\n", 1)[0] else ","))
    if not rows or not {"query_id", "premise_id"} <= rows[0].keys():
        raise SystemExit(f"{path}: cần cột query_id và premise_id")
    by_q: dict[str, list[tuple[float, int, str]]] = {}
    for i, r in enumerate(rows):
        by_q.setdefault(r["query_id"], []).append((float(r["rank"]) if r.get("rank") else i, i, r["premise_id"]))
    known = {q["id"] for q in queries}
    rankings = {qid: eligible_ranking(((p, None) for _, _, p in sorted(v)), qid, k) for qid, v in by_q.items()
                if qid in known}
    info = {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows),
            "queries_in_file": len(by_q), "unknown_queries": len(set(by_q) - known),
            "missing_queries": len(known - set(by_q))}
    return rankings, info


def rrf(runs: list[dict], queries: list[dict], k: int, rrf_k: int = 60) -> dict:
    """Reciprocal Rank Fusion của các danh sách đã qua quy tắc eligible (cùng công thức với fuse_rrf.fuse)."""
    fused = {}
    for q in queries:
        scores: dict[str, float] = {}
        for r in runs:
            for rank, (name, _) in enumerate(r.get(q["id"], []), 1):
                scores[name] = scores.get(name, 0.0) + 1.0 / (rrf_k + rank)
        fused[q["id"]] = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
    return fused


def per_query(system: str, rankings: dict, lat: dict | None, queries: list[dict]) -> pd.DataFrame:
    rows = []
    for q in queries:
        if not q["relevant_ids"]:
            continue
        r = metrics.per_query([n for n, _ in rankings.get(q["id"], [])], set(q["relevant_ids"]))
        r.update(system=system, query_id=q["id"], signature=not has_proof_body(q["statement"]),
                 latency_ms=lat.get(q["id"], np.nan) if lat else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def slice_of(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df if name == "all" else df[df.signature]


def compare(pq_df: pd.DataFrame, pairs: list[tuple[str, str]]) -> list[dict]:
    out = []
    for sl in SLICES:
        part = slice_of(pq_df, sl)
        res = []
        for a, b in pairs:
            x = part[part.system == a].set_index("query_id")["R@10"]
            y = part[part.system == b].set_index("query_id")["R@10"].reindex(x.index)
            ci = metrics.paired_cluster_bootstrap(x.to_numpy(), y.to_numpy(), x.index.to_numpy())
            p = metrics.paired_cluster_permutation(x.to_numpy(), y.to_numpy(), x.index.to_numpy())["p"]
            res.append({"slice": sl, "a": a, "b": b, "delta_R@10": ci["delta"], "ci95_low": ci["ci95_low"],
                        "ci95_high": ci["ci95_high"], "p": p})
        for r, ph in zip(res, metrics.holm([r["p"] for r in res])):
            r["p_holm"] = ph
        out += res
    return out


def write_markdown(path: Path, run_id: str, report: dict, queries: list[dict], pq_df: pd.DataFrame,
                   systems: list[str], comps: list[dict], external: dict, fused: dict, rrf_k: int) -> None:
    n_sig = int(pq_df[pq_df.system == systems[0]].signature.sum())
    n_scored = int((pq_df.system == systems[0]).sum())
    n_body = sum(has_proof_body(q["statement"]) for q in queries)
    md = [f"# Benchmark chung N3/N4 (sinh bởi scripts/eval_shared.py, run `{run_id}`)", "",
          f"Gói `{DATA_VERSION}`, pool `{report['pool_id']}`: {report['corpus_count']} premise, "
          f"{report['query_count']} truy vấn mức theorem (random/test), SHA-256 khớp bản ghim. "
          f"{report['queries_without_direct_premises']} truy vấn không có premise đúng nào trong corpus nên không "
          f"chấm; còn {n_scored} truy vấn. {n_body} truy vấn có statement còn chứa chứng minh; lát `signature` bỏ chúng, "
          f"còn {n_sig} truy vấn.", "",
          "Quy tắc eligible: loại chính theorem truy vấn trước khi cắt top-k, như nhau cho mọi hệ thống. V4 và V5 lọc "
          "thêm premise truy cập được (dùng file và dòng của theorem); B0 và V2 chỉ dùng statement. Benchmark này khác "
          "benchmark mức tactic của báo cáo, nên không so số trực tiếp với bảng trong báo cáo.", ""]
    for sl in SLICES:
        part = slice_of(pq_df, sl)
        md += [f"## Lát `{sl}`", "", "| Hệ thống | n | " + " | ".join(METRICS) + " | P50 ms | P95 ms |",
               "|---|---|" + "---|" * (len(METRICS) + 2)]
        for s in systems:
            d = part[part.system == s]
            lat = d.latency_ms.dropna()
            p50, p95 = (f"{np.percentile(lat, 50):.1f}", f"{np.percentile(lat, 95):.1f}") if len(lat) else ("", "")
            md.append(f"| {s} | {len(d)} | " + " | ".join(f"{d[m].mean():.3f}" for m in METRICS) + f" | {p50} | {p95} |")
        md.append("")
    md += ["## So sánh R@10", "", "Hoán vị ghép cặp hai phía, 10.000 phép hoán vị; Holm trong từng lát; KTC 95% bằng "
           "paired bootstrap.", "", "| Lát | A | B | B − A | KTC 95% | p | p Holm |", "|---|---|---|---|---|---|---|"]
    md += [f"| {c['slice']} | {c['a']} | {c['b']} | {c['delta_R@10']:+.3f} | [{c['ci95_low']:.3f}; {c['ci95_high']:.3f}] "
           f"| {c['p']:.4f} | {c['p_holm']:.4f} |" for c in comps]
    if external:
        md += ["", "## Run file ngoài", ""] + [
            f"- {name}: `{i['path']}` (SHA-256 `{i['sha256']}`), {i['queries_in_file']} truy vấn trong tệp, "
            f"{i['missing_queries']} truy vấn thiếu, {i['unknown_queries']} truy vấn lạ bị bỏ" for name, i in external.items()]
    else:
        md += ["", "Chưa có run file của Nhóm 4. Khi nhận được: `python scripts/eval_shared.py --external "
               "N4-dense=<run_dense>.csv` (tự trộn RRF với V4)."]
    if fused:
        md += ["", "## Trộn RRF", "", f"Điểm = Σ 1/({rrf_k} + hạng), trọng số bằng nhau, hòa điểm thì full_name nhỏ "
               "trước; hạng lấy từ danh sách đã qua quy tắc eligible. Gói chung chỉ có tập test nên không tinh chỉnh k "
               "hay trọng số.", ""] + [f"- {f}: {' + '.join(parts)}" for f, parts in fused.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/n4_shared", help="thư mục hoặc tệp zip của gói (scripts/export_n4_shared.py)")
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--external", nargs="*", default=[], metavar="TÊN=ĐƯỜNG_DẪN")
    ap.add_argument("--fuse", nargs="*", default=None, metavar="A+B",
                    help="hệ thống cần trộn RRF; mặc định mỗi run ngoài trộn với V4")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy n truy vấn đầu (chạy thử, không ghi kết quả)")
    args = ap.parse_args()
    queries, report = load_queries(ROOT / args.bundle)
    if args.limit:
        queries = queries[:args.limit]
    names = pq.read_table(PROCESSED / "premises.parquet", columns=["premise_id", "full_name"]).to_pandas()
    full_name_of = names.sort_values("premise_id").full_name.to_numpy()
    ctxs = theorem_contexts()

    frames, systems, cfgs, summary, run_files, ranked_by = [], [], [], {}, {}, {}
    run_dir = ROOT / "results" / "runs" / "shared"

    def write_run(name: str, rankings: dict) -> None:
        if args.limit:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"{qid}\t{n}\t{r}\t{s:.6f}\t{name}" for qid, ranked in rankings.items() for r, (n, s) in enumerate(ranked, 1)]
        run_file = run_dir / f"{name}.tsv"
        run_file.write_text("query_id\tpremise_id\trank\tscore\tsystem\n" + "\n".join(lines) + "\n", encoding="utf-8")
        run_files[name] = run_file.relative_to(ROOT).as_posix()

    for path in args.configs:
        cfg = load_config(ROOT / path)
        t0 = time.perf_counter()
        rankings, lat = run_n3(cfg, queries, full_name_of, ctxs, args.k)
        frames.append(per_query(cfg["name"], rankings, lat, queries))
        systems.append(cfg["name"])
        cfgs.append(cfg)
        ranked_by[cfg["name"]] = rankings
        print(f"{cfg['name']}: R@10 {frames[-1]['R@10'].mean():.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        write_run(cfg["name"], rankings)
    external = {}
    for spec in args.external:
        name, _, path = spec.partition("=")
        rankings, external[name] = read_external(Path(path), queries, args.k)
        frames.append(per_query(name, rankings, None, queries))
        systems.append(name)
        ranked_by[name] = rankings
    fuse_specs = args.fuse if args.fuse is not None else [f"V4+{e}" for e in external if "V4" in ranked_by]
    fused = {}
    for spec in fuse_specs:
        parts = spec.split("+")
        if len(parts) < 2 or any(p not in ranked_by for p in parts):
            raise SystemExit(f"--fuse {spec}: cần ít nhất hai hệ thống trong {sorted(ranked_by)}")
        name = f"RRF({spec})"
        fused[name] = parts
        rankings = rrf([ranked_by[p] for p in parts], queries, args.k, args.rrf_k)
        frames.append(per_query(name, rankings, None, queries))
        systems.append(name)
        write_run(name, rankings)
        print(f"{name}: R@10 {frames[-1]['R@10'].mean():.4f}", flush=True)
    pq_df = pd.concat(frames, ignore_index=True)
    for s in systems:
        d = pq_df[pq_df.system == s]
        summary[s] = {sl: {"n": int(len(slice_of(d, sl))), **{m: float(slice_of(d, sl)[m].mean()) for m in METRICS}}
                      for sl in SLICES}
        lat = d.latency_ms.dropna()
        if len(lat):
            summary[s].update(latency_ms_p50=float(np.percentile(lat, 50)), latency_ms_p95=float(np.percentile(lat, 95)))
    n3_names = [c["name"] for c in cfgs]
    pairs = list(zip(n3_names, n3_names[1:])) + [(a, e) for e in external for a in n3_names] + \
        [(p, f) for f, parts in fused.items() for p in parts]
    comps = compare(pq_df, pairs)
    if args.limit:
        print(json.dumps({s: v["all"] for s, v in summary.items()}, indent=1))
        return
    run_id = runlog.new_run_id("shared")
    out = ROOT / "results" / "raw" / "shared"
    out.mkdir(parents=True, exist_ok=True)
    pq_df.to_parquet(out / f"{run_id}.perquery.parquet", index=False)
    rec = runlog.write_record(
        "shared", parameters={"configs": cfgs, "k": args.k, "bundle": {
            "data_version": DATA_VERSION, "pool_id": report["pool_id"], "sha256": report["checksums_sha256"]},
            "eligible_rule": report["eligible_rule"], "external": external,
            "fusion": {"systems": fused, "rrf_k": args.rrf_k, "weights": "equal", "depth": args.k}},
        metrics=summary, seed=42, run_id=run_id,
        extra={"comparisons": comps, "per_query_file": f"results/raw/shared/{run_id}.perquery.parquet",
               "run_files": run_files, "n_queries": len(queries),
               "n_scored": int((pq_df.system == systems[0]).sum())})
    write_markdown(ROOT / "results" / "figures" / "shared_benchmark.md", run_id, report, queries, pq_df, systems,
                   comps, external, fused, args.rrf_k)
    print(f"{rec.relative_to(ROOT).as_posix()}\nresults/figures/shared_benchmark.md")


if __name__ == "__main__":
    main()
