"""Xây inverted index bằng Spark (MapReduce), kết quả phải trùng từng byte với bản tuần tự.

Luồng (Chương 2, slide 2-10 đến 2-14; Chương 4, slide 4-14 đến 4-21):
  Map      premise → ((field, term), (doc_id, tf))   [kết hợp cục bộ trong tài liệu]
           hoặc  ((field, term, doc_id), 1) cho mỗi token rồi reduceByKey  [--no-local-combine]
  Shuffle  partition = hash((field, term)) mod R
           hot key (df ≥ ngưỡng) được tách theo khoảng doc_id: salt = doc_id * S // N  [--salt S]
  Reduce   mỗi key: sắp theo doc_id → df, cf → mã hóa (codec) → ghi shard part-xxxxx trên executor
  Merge    driver trộn dictionary các shard theo (field, term) và chép byte vào postings.bin

Không ghi qua Hadoop FileSystem (tránh phụ thuộc winutils trên Windows): executor tự ghi file shard.
Số đo lấy từ REST API của Spark UI: shuffle bytes, spill, thời gian task max/median theo stage.

Dùng:  python -m index.build_spark --out indexes/spark_x --master local[4] [--subset data/processed/subsets/premises_10k.npy]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from collections import Counter
from operator import add
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.tokenizer import get_tokenizer
from index import codec as codec_mod
from index.build_seq import iter_premises, write_index_encoded


class PeakMemory(threading.Thread):
    """Lấy mẫu tổng RSS của tiến trình hiện tại và mọi tiến trình con (JVM, Python worker)."""

    def __init__(self, interval: float = 0.2):
        super().__init__(daemon=True)
        self.interval, self.peak, self._stop = interval, 0, threading.Event()

    def run(self):
        me = psutil.Process()
        while not self._stop.is_set():
            total = 0
            for p in [me, *me.children(recursive=True)]:
                try:
                    total += p.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()


def spark_stage_metrics(ui_url: str, app_id: str) -> list[dict]:
    def get(path):
        with urllib.request.urlopen(f"{ui_url}/api/v1/applications/{app_id}/{path}", timeout=30) as r:
            return json.load(r)
    out = []
    for st in get("stages?status=complete"):
        row = {k: st.get(k) for k in ("stageId", "attemptId", "name", "numTasks", "executorRunTime", "inputRecords",
                                      "outputRecords", "shuffleReadBytes", "shuffleReadRecords", "shuffleWriteBytes",
                                      "shuffleWriteRecords", "memoryBytesSpilled", "diskBytesSpilled")}
        try:
            q = get(f"stages/{st['stageId']}/{st['attemptId']}/taskSummary?quantiles=0.5,1.0")
            run = q.get("executorRunTime", [0, 0])
            row["task_ms_median"], row["task_ms_max"] = run[0], run[1]
        except Exception:  # noqa: BLE001  (taskSummary có thể thiếu ở stage rỗng)
            row["task_ms_median"] = row["task_ms_max"] = None
        out.append(row)
    return sorted(out, key=lambda r: r["stageId"])


# ---------- các hàm chạy trên executor (phải pickle được) ----------
def _analyze(fields, tokenizer):
    tok = get_tokenizer(tokenizer)

    def f(doc):
        doc_id, rec = doc
        return doc_id, {fld: Counter(tok(rec.get(fld) or "")) for fld in fields}
    return f


def _write_shard(out_dir: str, codec: str):
    def f(part_idx, it):
        items = sorted(it, key=lambda kv: kv[0])
        path = Path(out_dir) / f"part-{part_idx:05d}.bin"
        rows, off = [], 0
        with path.open("wb") as fh:
            for (field, term), plist in items:
                doc_ids = np.fromiter((d for d, _ in plist), dtype=np.int64, count=len(plist))
                tfs = np.fromiter((t for _, t in plist), dtype=np.int64, count=len(plist))
                buf = codec_mod.encode_postings(doc_ids, tfs, codec)
                fh.write(buf)
                rows.append((field, term, len(plist), int(tfs.sum()), part_idx, off, len(buf)))
                off += len(buf)
        return iter(rows)
    return f


def build(premises: Path, out_dir: Path, *, fields: list[str], tokenizer: str = "default", codec: str = "vb",
          master: str = "local[4]", partitions: int | None = None, local_combine: bool = True, salt: int = 0,
          hot_df: int = 20000, subset: str | None = None, ui_port: int = 4045) -> dict:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    limit = np.load(subset) if subset else None
    docs_iter, n_docs, source = iter_premises(premises, fields, limit)
    pids = source.pop("premise_ids")
    source["subset"] = subset
    docs = list(docs_iter)

    mem = PeakMemory()
    mem.start()
    t0 = time.perf_counter()
    spark = (SparkSession.builder.master(master).appName(f"n3-index-{out_dir.name}")
             .config("spark.ui.port", str(ui_port)).config("spark.driver.memory", "4g")
             .config("spark.python.worker.reuse", "true").config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    t_start = time.perf_counter() - t0
    R = partitions or sc.defaultParallelism
    shard_dir = out_dir / "_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    t1 = time.perf_counter()
    analyzed = sc.parallelize(docs, R).map(_analyze(fields, tokenizer)).cache()
    lengths = dict(analyzed.map(lambda d: (d[0], [sum(d[1][f].values()) for f in fields])).collect())
    df_any = dict(analyzed.flatMap(lambda d: [(t, 1) for t in set().union(*d[1].values())]).reduceByKey(add, R).collect())

    if local_combine:
        pairs = analyzed.flatMap(lambda d: [((f, t), (d[0], c)) for f, cnt in d[1].items() for t, c in cnt.items()])
    else:
        pairs = (analyzed.flatMap(lambda d: [((f, t, d[0]), 1) for f, cnt in d[1].items() for t, c in cnt.items()
                                             for _ in range(c)])
                 .reduceByKey(add, R).map(lambda kv: ((kv[0][0], kv[0][1]), (kv[0][2], kv[1]))))

    hot: set = set()
    if salt > 1:
        hot = set(pairs.map(lambda kv: (kv[0], 1)).reduceByKey(add, R).filter(lambda kv: kv[1] >= hot_df).keys().collect())
        hot_b = sc.broadcast(hot)
        n_b = n_docs

        def salted(kv):
            key, (d, c) = kv
            s = d * salt // n_b if key in hot_b.value else 0
            return (key, s), (d, c)
        partial = (pairs.map(salted).groupByKey(R)
                   .map(lambda kv: (kv[0][0], (kv[0][1], sorted(kv[1])))))
        grouped = partial.groupByKey(R).mapValues(lambda parts: [p for _, lst in sorted(parts) for p in lst])
    else:
        grouped = pairs.groupByKey(R).mapValues(sorted)
    shard_rows = grouped.mapPartitionsWithIndex(_write_shard(str(shard_dir), codec)).collect()
    t_job = time.perf_counter() - t1

    stages = []
    try:
        stages = spark_stage_metrics(sc.uiWebUrl, sc.applicationId)
    except Exception as e:  # noqa: BLE001
        stages = [{"error": repr(e)}]
    spark.stop()

    # ---- merge trên driver ----
    t2 = time.perf_counter()
    shard_rows.sort(key=lambda r: (r[0], r[1]))
    handles = {}

    def entries():
        for field, term, df, cf, part, off, nb in shard_rows:
            fh = handles.get(part) or handles.setdefault(part, (shard_dir / f"part-{part:05d}.bin").open("rb"))
            fh.seek(off)
            yield field, term, df, cf, fh.read(nb)

    lens = {f: np.array([lengths[i][j] for i in range(n_docs)], dtype=np.int64) for j, f in enumerate(fields)}
    shard_bytes = sum(p.stat().st_size for p in shard_dir.glob("part-*.bin"))
    manifest = write_index_encoded(out_dir, entries(), lens, df_any, fields=fields, tokenizer=tokenizer, codec=codec,
                                   source={**source, "premises": str(premises).replace("\\", "/")},
                                   build_stats={"builder": "spark"})
    for fh in handles.values():
        fh.close()
    pq.write_table(pa.table({"doc_id": np.arange(n_docs, dtype=np.int32), "premise_id": pids.astype(np.int32)}),
                   out_dir / "doc_map.parquet")
    shutil.rmtree(shard_dir)
    t_merge = time.perf_counter() - t2
    mem.stop()

    reduce_stage = max((s for s in stages if s.get("shuffleReadBytes")), key=lambda s: s["shuffleReadBytes"], default={})
    df_arr = np.array([r[2] for r in shard_rows])
    manifest["build"].update({
        "master": master, "partitions": R, "local_combine": local_combine, "salt": salt, "hot_df": hot_df,
        "hot_keys": len(hot), "spark_start_s": round(t_start, 2), "job_s": round(t_job, 2), "merge_s": round(t_merge, 2),
        "total_s": round(time.perf_counter() - t0, 2), "docs_per_s": round(n_docs / t_job, 1),
        "peak_rss_mb": round(mem.peak / 2**20, 1), "shard_bytes": shard_bytes,
        "shuffle_write_bytes": sum(s.get("shuffleWriteBytes") or 0 for s in stages),
        "shuffle_write_records": sum(s.get("shuffleWriteRecords") or 0 for s in stages),
        "spill_memory_bytes": sum(s.get("memoryBytesSpilled") or 0 for s in stages),
        "spill_disk_bytes": sum(s.get("diskBytesSpilled") or 0 for s in stages),
        "reduce_task_ms_median": reduce_stage.get("task_ms_median"), "reduce_task_ms_max": reduce_stage.get("task_ms_max"),
        "max_df": int(df_arr.max()) if df_arr.size else 0,
        "stages": stages,
    })
    (out_dir / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Xây inverted index bằng Spark")
    ap.add_argument("--premises", default="data/processed/premises.parquet")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--fields", default="statement")
    ap.add_argument("--tokenizer", default="default", choices=["default", "lean"])
    ap.add_argument("--codec", default="vb", choices=list(codec_mod.CODECS))
    ap.add_argument("--master", default="local[4]")
    ap.add_argument("--partitions", type=int, default=None)
    ap.add_argument("--no-local-combine", action="store_true")
    ap.add_argument("--salt", type=int, default=0, help="số khoảng tách hot key (0 = tắt)")
    ap.add_argument("--hot-df", type=int, default=20000)
    ap.add_argument("--engine", default="arrow", choices=["arrow", "rdd"],
                    help="arrow: bản tối ưu (build_spark_arrow); rdd: bản đầu tiên, giữ để so sánh")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.engine == "arrow":
        if args.no_local_combine or args.salt:
            ap.error("--no-local-combine và --salt chỉ áp dụng cho --engine rdd")
        from index.build_spark_arrow import build_arrow
        m = build_arrow(Path(args.premises), Path(args.out), fields=args.fields.split(","), tokenizer=args.tokenizer,
                        codec=args.codec, master=args.master, partitions=args.partitions, subset=args.subset)
    else:
        m = build(Path(args.premises), Path(args.out), fields=args.fields.split(","), tokenizer=args.tokenizer,
                  codec=args.codec, master=args.master, partitions=args.partitions,
                  local_combine=not args.no_local_combine, salt=args.salt, hot_df=args.hot_df, subset=args.subset)
    b = {k: v for k, v in m["build"].items() if k != "stages"}
    print(json.dumps({"n_docs": m["n_docs"], "n_postings": m["n_postings"], **b}, ensure_ascii=False))


if __name__ == "__main__":
    main()
