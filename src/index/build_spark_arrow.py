"""Xây inverted index bằng Spark, bản tối ưu (engine "arrow"). Kết quả trùng từng byte với bản tuần tự.

Bản RDD đầu tiên (build_spark.build) chậm hơn bản tuần tự vì ba lý do: driver đọc toàn bộ dữ liệu rồi parallelize,
mỗi posting là một bản ghi Python phải pickle khi shuffle, và có thêm một lượt shuffle riêng cho df_any.
Bản này sửa cả ba theo các mẫu thiết kế của Lin và Dyer (2010):

  Đọc      executor đọc Parquet qua Spark SQL; driver chỉ đọc cột premise_id để đánh doc_id và ghi doc_map
  Map      mapInArrow: Python nhận lô Arrow, tách từ và GỘP THEO PARTITION (in-mapper combining ở mức partition):
           mỗi (field, term) chỉ phát MỘT bản ghi cho mỗi partition, chứa cả dãy doc_id và tf dưới dạng bytes
  Shuffle repartition theo (field, term) trong JVM, sắp trong partition để các mảnh cùng khóa nằm liền nhau
  Reduce   mapInArrow: nối các mảnh, sắp theo doc_id, mã hóa (codec) và ghi shard trên executor
  Merge    driver trộn dictionary theo (field, term) bằng đúng hàm ghi của bản tuần tự

df_any (số tài liệu chứa term ở bất kỳ field nào) và độ dài field đi cùng lượt shuffle dưới dạng khóa đặc biệt,
nên chỉ có một lượt shuffle dữ liệu chính. Với chỉ mục một field, df_any suy ra trực tiếp từ df.

Số bản ghi shuffle giảm từ số posting (bản RDD) xuống tối đa (số khóa) × (số partition); khi corpus lớn lên,
mức giảm tăng theo vì số khóa gần như không đổi.

Dùng:  python -m index.build_spark --engine arrow --out indexes/x --master local[4] [--fields name,namespace,statement --tokenizer lean]
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import uuid
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.tokenizer import get_tokenizer
from index import codec as codec_mod
from index.build_seq import write_index_encoded

ANY = "\x01any"      # khóa đặc biệt: số tài liệu chứa term ở bất kỳ field nào
LEN = "\x02len"      # khóa đặc biệt: độ dài field của từng tài liệu (term = tên field)
CHUNK_SCHEMA = pa.schema([("field", pa.string()), ("term", pa.string()), ("docs", pa.binary()),
                          ("tfs", pa.binary()), ("cnt", pa.int64())])
DICT_SCHEMA = pa.schema([("field", pa.string()), ("term", pa.string()), ("df", pa.int64()), ("cf", pa.int64()),
                         ("shard", pa.string()), ("off", pa.int64()), ("nbytes", pa.int64())])
BATCH_ROWS = 50_000
assert array("I").itemsize == 4


def _ddl(schema: pa.Schema) -> str:
    names = {pa.string(): "string", pa.binary(): "binary", pa.int64(): "bigint"}
    return ", ".join(f"{f.name} {names[f.type]}" for f in schema)


# ---------- chạy trên executor (phải pickle được) ----------
def _mapper(fields: list[str], tokenizer: str, emit_any: bool):
    def f(batches):
        tok = get_tokenizer(tokenizer)
        docs: dict = defaultdict(lambda: array("I"))
        tfs: dict = defaultdict(lambda: array("I"))
        any_cnt: Counter = Counter()
        len_docs = array("I")
        len_vals = {fld: array("I") for fld in fields}
        for b in batches:
            ids = b.column(0).to_pylist()
            cols = [b.column(i + 1).to_pylist() for i in range(len(fields))]
            for k, d in enumerate(ids):
                len_docs.append(d)
                seen: set = set()
                for j, fld in enumerate(fields):
                    c = Counter(tok(cols[j][k] or ""))
                    len_vals[fld].append(sum(c.values()))
                    for t, tf in c.items():
                        docs[(fld, t)].append(d)
                        tfs[(fld, t)].append(tf)
                    if emit_any:
                        seen.update(c)
                if emit_any:
                    any_cnt.update(seen)
        out = {n: [] for n in CHUNK_SCHEMA.names}

        def push(fld, term, d, t, n):
            out["field"].append(fld)
            out["term"].append(term)
            out["docs"].append(d)
            out["tfs"].append(t)
            out["cnt"].append(n)

        def flush():
            batch = pa.RecordBatch.from_pydict(out, schema=CHUNK_SCHEMA)
            for v in out.values():
                v.clear()
            return batch

        for (fld, term), d in docs.items():
            push(fld, term, d.tobytes(), tfs[(fld, term)].tobytes(), len(d))
            if len(out["field"]) >= BATCH_ROWS:
                yield flush()
        for term, n in any_cnt.items():
            push(ANY, term, b"", b"", n)
            if len(out["field"]) >= BATCH_ROWS:
                yield flush()
        if len(len_docs):
            for fld in fields:
                push(LEN, fld, len_docs.tobytes(), len_vals[fld].tobytes(), len(len_docs))
        if out["field"]:
            yield flush()
    return f


def _reducer(shard_dir: str, codec: str):
    def f(batches):
        name = f"part-{uuid.uuid4().hex}.bin"
        path = Path(shard_dir) / name
        out = {n: [] for n in DICT_SCHEMA.names}
        state = {"key": None, "docs": [], "tfs": [], "cnt": 0, "off": 0}

        def emit(fld, term, df, cf, shard, off, nb):
            for k, v in zip(DICT_SCHEMA.names, (fld, term, df, cf, shard, off, nb)):
                out[k].append(v)

        with path.open("wb") as fh:
            def close_key():
                key = state["key"]
                if key is None:
                    return
                fld, term = key
                if fld == ANY:
                    emit(fld, term, state["cnt"], 0, "", 0, 0)
                else:
                    d = np.concatenate([np.frombuffer(x, dtype=np.uint32) for x in state["docs"]]).astype(np.int64)
                    t = np.concatenate([np.frombuffer(x, dtype=np.uint32) for x in state["tfs"]]).astype(np.int64)
                    order = np.argsort(d, kind="stable")
                    d, t = d[order], t[order]
                    if fld == LEN:
                        buf = d.astype("<u4").tobytes() + t.astype("<u4").tobytes()
                        emit(fld, term, len(d), 0, name, state["off"], len(buf))
                    else:
                        buf = codec_mod.encode_postings(d, t, codec)
                        emit(fld, term, len(d), int(t.sum()), name, state["off"], len(buf))
                    fh.write(buf)
                    state["off"] += len(buf)
                state["docs"], state["tfs"], state["cnt"] = [], [], 0

            for b in batches:
                fl, te = b.column(0).to_pylist(), b.column(1).to_pylist()
                dc, tc, cn = b.column(2).to_pylist(), b.column(3).to_pylist(), b.column(4).to_pylist()
                for i in range(b.num_rows):
                    key = (fl[i], te[i])
                    if key != state["key"]:
                        close_key()
                        state["key"] = key
                    state["docs"].append(dc[i])
                    state["tfs"].append(tc[i])
                    state["cnt"] += cn[i]
                if len(out["field"]) >= BATCH_ROWS:
                    yield pa.RecordBatch.from_pydict(out, schema=DICT_SCHEMA)
                    for v in out.values():
                        v.clear()
            close_key()
        if out["field"]:
            yield pa.RecordBatch.from_pydict(out, schema=DICT_SCHEMA)
    return f


def merge_on_driver(table: pa.Table, shard_dir: Path, out_dir: Path, *, fields: list[str], tokenizer: str, codec: str,
                    n_docs: int, pids: np.ndarray, emit_any: bool, source: dict) -> dict:
    """Trộn dictionary do reducer trả về (bảng DICT_SCHEMA) và các shard thành chỉ mục, bằng đúng hàm ghi của bản tuần tự."""
    rows = table.to_pydict()
    recs = list(zip(rows["field"], rows["term"], rows["df"], rows["cf"], rows["shard"], rows["off"], rows["nbytes"]))
    shards: dict[str, bytes] = {}

    def blob(shard: str, off: int, nb: int) -> bytes:
        if shard not in shards:
            shards[shard] = (shard_dir / shard).read_bytes()
        return shards[shard][off:off + nb]

    lengths = {fld: np.zeros(n_docs, dtype=np.int64) for fld in fields}
    df_any: dict[str, int] = {}
    post = []
    for r in recs:
        if r[0] == LEN:
            buf = np.frombuffer(blob(r[4], r[5], r[6]), dtype="<u4")
            lengths[r[1]][buf[:r[2]].astype(np.int64)] = buf[r[2]:]
        elif r[0] == ANY:
            df_any[r[1]] = r[2]
        else:
            post.append(r)
    post.sort(key=lambda r: (r[0], r[1]))
    if not emit_any:
        df_any = {r[1]: r[2] for r in post}

    def entries():
        for fld, term, df, cf, shard, off, nb in post:
            yield fld, term, df, cf, blob(shard, off, nb)

    manifest = write_index_encoded(out_dir, entries(), lengths, df_any, fields=fields, tokenizer=tokenizer, codec=codec,
                                   source=source, build_stats={"builder": "spark-arrow"})
    pq.write_table(pa.table({"doc_id": np.arange(n_docs, dtype=np.int32), "premise_id": pids.astype(np.int32)}),
                   out_dir / "doc_map.parquet")
    df_arr = np.array([r[2] for r in post])
    manifest["build"]["max_df"] = int(df_arr.max()) if df_arr.size else 0
    return manifest


def build_arrow(premises: Path, out_dir: Path, *, fields: list[str], tokenizer: str = "default", codec: str = "vb",
                master: str = "local[4]", partitions: int | None = None, subset: str | None = None,
                ui_port: int = 4045) -> dict:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from index.build_spark import PeakMemory, spark_stage_metrics

    mem = PeakMemory()
    mem.start()
    t0 = time.perf_counter()
    # Driver chỉ đọc cột premise_id: đủ để đánh doc_id liên tục (giống bản tuần tự) và ghi doc_map.
    ids = pq.read_table(premises, columns=["premise_id"]).column("premise_id").to_numpy()
    if subset:
        ids = ids[np.isin(ids, np.load(subset))]
    pids = np.sort(ids)
    n_docs = len(pids)
    contiguous = n_docs > 0 and pids[0] == 0 and pids[-1] == n_docs - 1 and len(np.unique(pids)) == n_docs

    spark = (SparkSession.builder.master(master).appName(f"n3-index-arrow-{out_dir.name}")
             .config("spark.ui.port", str(ui_port)).config("spark.driver.memory", "4g")
             .config("spark.python.worker.reuse", "true").config("spark.ui.showConsoleProgress", "false")
             .config("spark.sql.adaptive.enabled", "false")          # giữ đúng R partition để đo song song
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    t_start = time.perf_counter() - t0
    R = partitions or sc.defaultParallelism
    spark.conf.set("spark.sql.shuffle.partitions", str(R))
    shard_dir = out_dir / "_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    t1 = time.perf_counter()
    sdf = spark.read.parquet(Path(premises).resolve().as_uri()).select("premise_id", *fields)
    if contiguous and not subset:
        sdf = sdf.withColumnRenamed("premise_id", "doc_id")
    else:  # index con: doc_id = thứ hạng của premise_id trong tập con, nối qua broadcast join
        import pandas as pd
        mapping = spark.createDataFrame(pd.DataFrame({"premise_id": pids.astype(np.int64),
                                                      "doc_id": np.arange(n_docs, dtype=np.int64)}))
        sdf = sdf.join(F.broadcast(mapping), "premise_id").select("doc_id", *fields)
    # Spark chỉ chia việc đọc Parquet theo row group. Tệp có ít row group hơn R thì dù báo đủ R partition, dữ liệu vẫn
    # dồn vào vài partition, nên phải chia lại (round robin trong JVM) để mọi worker đều có việc.
    if pq.ParquetFile(premises).metadata.num_row_groups < R:
        sdf = sdf.repartition(R)
    emit_any = len(fields) > 1
    chunks = sdf.mapInArrow(_mapper(fields, tokenizer, emit_any), _ddl(CHUNK_SCHEMA))
    reduced = (chunks.repartition(R, "field", "term").sortWithinPartitions("field", "term")
               .mapInArrow(_reducer(str(shard_dir), codec), _ddl(DICT_SCHEMA)))
    table = reduced.toArrow() if hasattr(reduced, "toArrow") else pa.Table.from_pandas(reduced.toPandas())
    t_job = time.perf_counter() - t1

    stages = []
    try:
        stages = spark_stage_metrics(sc.uiWebUrl, sc.applicationId)
    except Exception as e:  # noqa: BLE001
        stages = [{"error": repr(e)}]
    spark.stop()

    t2 = time.perf_counter()
    shard_bytes = sum(p.stat().st_size for p in shard_dir.glob("part-*.bin"))
    manifest = merge_on_driver(table, shard_dir, out_dir, fields=fields, tokenizer=tokenizer, codec=codec, n_docs=n_docs,
                               pids=pids, emit_any=emit_any,
                               source={"premises": str(premises).replace("\\", "/"), "subset": subset})
    shutil.rmtree(shard_dir)
    t_merge = time.perf_counter() - t2
    mem.stop()

    reduce_stage = max((s for s in stages if s.get("shuffleReadBytes")), key=lambda s: s["shuffleReadBytes"], default={})
    manifest["build"].update({
        "engine": "arrow", "master": master, "partitions": R, "local_combine": "partition",
        "spark_start_s": round(t_start, 2), "job_s": round(t_job, 2), "merge_s": round(t_merge, 2),
        "total_s": round(time.perf_counter() - t0, 2), "docs_per_s": round(n_docs / t_job, 1),
        "peak_rss_mb": round(mem.peak / 2**20, 1), "shard_bytes": shard_bytes,
        "shuffle_write_bytes": sum(s.get("shuffleWriteBytes") or 0 for s in stages),
        "shuffle_write_records": sum(s.get("shuffleWriteRecords") or 0 for s in stages),
        "spill_memory_bytes": sum(s.get("memoryBytesSpilled") or 0 for s in stages),
        "spill_disk_bytes": sum(s.get("diskBytesSpilled") or 0 for s in stages),
        "reduce_task_ms_median": reduce_stage.get("task_ms_median"), "reduce_task_ms_max": reduce_stage.get("task_ms_max"),
        "stages": stages,
    })
    import json
    (out_dir / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
