"""Bộ xây inverted index tuần tự (Python thuần): bản chuẩn để đối chiếu với bản Spark.

Đơn vị dữ liệu: một premise (doc_id nguyên, liên tục, ổn định) với nhiều field văn bản.
Key trung gian: (field, term). Posting: (doc_id, tf), sắp tăng theo doc_id.

Bố cục index trên đĩa (thư mục):
- postings.bin          các posting list đã mã hóa, nối liên tiếp theo thứ tự (field, term)
- dictionary.parquet    field, term, df, cf, offset, nbytes
- terms_any.parquet     term, df_any (số tài liệu chứa term ở bất kỳ field nào, dùng cho idf)
- doc_lengths.parquet   doc_id, len_<field>
- index_manifest.json   tham số, thống kê, checksum
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.tokenizer import TOKENIZER_VERSION, get_tokenizer
from index import codec as codec_mod

FORMAT_VERSION = 1


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_index(out_dir: Path, postings: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
                doc_lengths: dict[str, np.ndarray], df_any: dict[str, int], *, fields: list[str],
                tokenizer: str, codec: str, source: dict, build_stats: dict) -> dict:
    """Ghi index từ posting chưa mã hóa (bản tuần tự)."""
    def entries():
        for field, term in sorted(postings):
            doc_ids, tfs = postings[(field, term)]
            yield field, term, len(doc_ids), int(np.sum(tfs)), codec_mod.encode_postings(doc_ids, tfs, codec)
    return write_index_encoded(out_dir, entries(), doc_lengths, df_any, fields=fields, tokenizer=tokenizer,
                               codec=codec, source=source, build_stats=build_stats)


def write_index_encoded(out_dir: Path, entries, doc_lengths: dict[str, np.ndarray], df_any: dict[str, int], *,
                        fields: list[str], tokenizer: str, codec: str, source: dict, build_stats: dict) -> dict:
    """Ghi index từ các posting đã mã hóa, duyệt theo thứ tự (field, term) tăng dần.
    Dùng chung cho bản tuần tự và bản Spark để hai bản so được từng byte."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {"field": [], "term": [], "df": [], "cf": [], "offset": [], "nbytes": []}
    offset = 0
    prev = None
    with (out_dir / "postings.bin").open("wb") as f:
        for field, term, df, cf, buf in entries:
            if prev is not None and (field, term) <= prev:
                raise ValueError("entries phải tăng ngặt theo (field, term)")
            prev = (field, term)
            f.write(buf)
            rows["field"].append(field)
            rows["term"].append(term)
            rows["df"].append(df)
            rows["cf"].append(cf)
            rows["offset"].append(offset)
            rows["nbytes"].append(len(buf))
            offset += len(buf)
    keys = list(zip(rows["field"], rows["term"]))
    pq.write_table(pa.table(rows), out_dir / "dictionary.parquet")
    any_terms = sorted(df_any)
    pq.write_table(pa.table({"term": any_terms, "df_any": [df_any[t] for t in any_terms]}), out_dir / "terms_any.parquet")
    n_docs = len(next(iter(doc_lengths.values())))
    pq.write_table(pa.table({"doc_id": np.arange(n_docs, dtype=np.int32),
                             **{f"len_{fld}": doc_lengths[fld].astype(np.int32) for fld in fields}}),
                   out_dir / "doc_lengths.parquet")
    df_arr = np.asarray(rows["df"], dtype=np.int64)
    manifest = {
        "format_version": FORMAT_VERSION,
        "tokenizer": tokenizer,
        "tokenizer_version": TOKENIZER_VERSION[tokenizer],
        "fields": fields,
        "codec": codec,
        "n_docs": int(n_docs),
        "avgdl": {fld: float(doc_lengths[fld].mean()) if n_docs else 0.0 for fld in fields},
        "n_terms": len(any_terms),
        "n_term_field_keys": len(keys),
        "n_postings": int(df_arr.sum()),
        "max_df": int(df_arr.max()) if df_arr.size else 0,
        "postings_bytes": offset,
        "source": source,
        "build": build_stats,
    }
    manifest["checksums"] = {p.name: _sha1(p) for p in sorted(out_dir.glob("*")) if p.suffix in (".bin", ".parquet")}
    (out_dir / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build(docs: Iterable[tuple[int, dict[str, str]]], n_docs: int, out_dir: Path, *, fields: list[str],
          tokenizer: str = "default", codec: str = "vb", source: dict | None = None) -> dict:
    """docs phải duyệt theo doc_id tăng dần 0..n_docs-1 để posting tự sắp sẵn."""
    tok = get_tokenizer(tokenizer)
    t0 = time.perf_counter()
    doc_acc: dict[tuple[str, str], array] = defaultdict(lambda: array("I"))
    tf_acc: dict[tuple[str, str], array] = defaultdict(lambda: array("I"))
    df_any: Counter = Counter()
    lengths = {f: np.zeros(n_docs, dtype=np.int64) for f in fields}
    expected = 0
    for doc_id, record in docs:
        if doc_id != expected:
            raise ValueError(f"doc_id phải liên tục và tăng dần: gặp {doc_id}, chờ {expected}")
        expected += 1
        seen: set[str] = set()
        for field in fields:
            counts = Counter(tok(record.get(field) or ""))  # kết hợp cục bộ trong tài liệu (slide 4-14)
            lengths[field][doc_id] = sum(counts.values())
            for term, tf in counts.items():
                doc_acc[(field, term)].append(doc_id)
                tf_acc[(field, term)].append(tf)
            seen.update(counts)
        df_any.update(seen)
    if expected != n_docs:
        raise ValueError(f"số tài liệu {expected} khác n_docs={n_docs}")
    t_map = time.perf_counter() - t0
    postings = {k: (np.frombuffer(doc_acc[k], dtype=np.uint32).astype(np.int64),
                    np.frombuffer(tf_acc[k], dtype=np.uint32).astype(np.int64)) for k in doc_acc}
    t1 = time.perf_counter()
    manifest = write_index(out_dir, postings, lengths, dict(df_any), fields=fields, tokenizer=tokenizer, codec=codec,
                           source=source or {}, build_stats={"builder": "sequential"})
    manifest["build"].update({"analyze_s": round(t_map, 3), "write_s": round(time.perf_counter() - t1, 3),
                              "total_s": round(time.perf_counter() - t0, 3)})
    (out_dir / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def iter_premises(path: Path, fields: list[str], limit_ids: np.ndarray | None = None) -> tuple[Iterator, int, dict]:
    """Đọc premises.parquet (hoặc một subset) và đánh lại doc_id liên tục 0..n-1.
    Trả (iterator, n_docs, source). Bảng ánh xạ doc_id → premise_id ghi vào index (doc_map.parquet)."""
    table = pq.read_table(path, columns=["premise_id", *fields])
    if limit_ids is not None:
        mask = np.isin(table.column("premise_id").to_numpy(), limit_ids)
        table = table.filter(pa.array(mask))
    table = table.sort_by("premise_id")
    cols = {f: table.column(f).to_pylist() for f in fields}
    pids = table.column("premise_id").to_numpy()

    def gen():
        for i in range(len(pids)):
            yield i, {f: cols[f][i] for f in fields}

    return gen(), len(pids), {"premises": str(path).replace("\\", "/"), "premise_ids": pids}


def main() -> None:
    ap = argparse.ArgumentParser(description="Xây inverted index tuần tự")
    ap.add_argument("--premises", default="data/processed/premises.parquet")
    ap.add_argument("--subset", default=None, help="file .npy chứa premise_id của index con (bỏ trống = toàn bộ)")
    ap.add_argument("--fields", default="statement")
    ap.add_argument("--tokenizer", default="default", choices=["default", "lean"])
    ap.add_argument("--codec", default="vb", choices=list(codec_mod.CODECS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fields = args.fields.split(",")
    limit = np.load(args.subset) if args.subset else None
    docs, n, source = iter_premises(Path(args.premises), fields, limit)
    pids = source.pop("premise_ids")
    source["subset"] = args.subset
    out = Path(args.out)
    manifest = build(docs, n, out, fields=fields, tokenizer=args.tokenizer, codec=args.codec, source=source)
    pq.write_table(pa.table({"doc_id": np.arange(n, dtype=np.int32), "premise_id": pids.astype(np.int32)}),
                   out / "doc_map.parquet")
    print(json.dumps({k: manifest[k] for k in ("n_docs", "n_terms", "n_postings", "max_df", "postings_bytes", "build")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
