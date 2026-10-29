"""Logic của engine Spark Arrow, kiểm tra không cần Spark: mapper gộp posting theo partition, shuffle được mô phỏng bằng
cách chia theo hash khóa rồi sắp trong partition, reducer nối và mã hóa, driver trộn. Chỉ mục phải trùng từng byte với
bản tuần tự, kể cả khi tài liệu trong partition không theo thứ tự và một khóa vắt qua nhiều lô Arrow."""
import numpy as np
import pyarrow as pa
import pytest

from index import build_seq
from index.build_spark_arrow import CHUNK_SCHEMA, DICT_SCHEMA, _mapper, _reducer, merge_on_driver

DOCS = [
    {"name": "Nat.add_comm", "namespace": "Nat", "statement": "theorem add_comm (n m : ℕ) : n + m = m + n"},
    {"name": "Nat.add_zero", "namespace": "Nat", "statement": "theorem add_zero (n : ℕ) : n + 0 = n"},
    {"name": "mul_assoc", "namespace": "", "statement": "theorem mul_assoc : ∀ a b c : G, a * b * c = a * (b * c)"},
    {"name": "Int.natAbs_mul", "namespace": "Int", "statement": "theorem natAbs_mul (a b : ℤ) : (a * b).natAbs = a.natAbs * b.natAbs"},
    {"name": "Finset.sum_comm", "namespace": "Finset", "statement": "theorem sum_comm {s : Finset γ} : ∑ x ∈ s, f x = ∑ x ∈ s, f x"},
    {"name": "le_refl", "namespace": "", "statement": "theorem le_refl (a : α) : a ≤ a"},
    {"name": "Nat.succ_le_of_lt", "namespace": "Nat", "statement": "theorem succ_le_of_lt {n m : ℕ} (h : n < m) : n + 1 ≤ m"},
    {"name": "Real.exp_pos", "namespace": "Real", "statement": "theorem exp_pos (x : ℝ) : 0 < Real.exp x"},
]


def run_arrow(tmp_path, fields, tokenizer, n_parts=3, n_reducers=2, batch_rows=3):
    docs = DOCS * 3                      # lặp để term phổ biến có posting dài, trải trên mọi partition
    ids = np.arange(len(docs))
    chunks = []
    for i in range(n_parts):
        part = ids[i::n_parts][::-1]     # partition không liên tục và không theo thứ tự doc_id
        batch = pa.RecordBatch.from_pydict({"doc_id": part.tolist(),
                                            **{f: [docs[d][f] for d in part] for f in fields}})
        for b in _mapper(fields, tokenizer, len(fields) > 1)(iter([batch])):
            chunks.extend(b.to_pylist())
    buckets = [[] for _ in range(n_reducers)]
    for r in chunks:
        buckets[hash((r["field"], r["term"])) % n_reducers].append(r)
    shard_dir = tmp_path / "_shards"
    shard_dir.mkdir()
    out_rows = []
    for bucket in buckets:
        bucket.sort(key=lambda r: (r["field"], r["term"]))
        batches = [pa.RecordBatch.from_pylist(bucket[k:k + batch_rows], schema=CHUNK_SCHEMA)
                   for k in range(0, len(bucket), batch_rows)]
        for b in _reducer(str(shard_dir), "vb")(iter(batches)):
            out_rows.extend(b.to_pylist())
    table = pa.Table.from_pylist(out_rows, schema=DICT_SCHEMA)
    m = merge_on_driver(table, shard_dir, tmp_path / "arrow", fields=fields, tokenizer=tokenizer, codec="vb",
                        n_docs=len(docs), pids=ids, emit_any=len(fields) > 1, source={})
    seq = build_seq.build(((i, docs[i]) for i in range(len(docs))), len(docs), tmp_path / "seq", fields=fields,
                          tokenizer=tokenizer, codec="vb")
    return m, seq


@pytest.mark.parametrize("fields,tokenizer", [(["statement"], "default"),
                                              (["name", "namespace", "statement"], "lean")])
def test_arrow_logic_matches_sequential_byte_for_byte(tmp_path, fields, tokenizer):
    m, seq = run_arrow(tmp_path, fields, tokenizer)
    assert m["checksums"] == seq["checksums"]
    assert m["n_postings"] == seq["n_postings"]


def test_partition_combining_emits_one_record_per_key_and_partition(tmp_path):
    fields = ["statement"]
    ids = np.arange(len(DOCS))
    batch = pa.RecordBatch.from_pydict({"doc_id": ids.tolist(), "statement": [d["statement"] for d in DOCS]})
    rows = [r for b in _mapper(fields, "default", False)(iter([batch])) for r in b.to_pylist()]
    keys = [(r["field"], r["term"]) for r in rows if r["field"] == "statement"]
    assert len(keys) == len(set(keys))                       # mỗi khóa đúng một bản ghi trong partition
    theorem = next(r for r in rows if r["term"] == "theorem")
    assert theorem["cnt"] == len(DOCS)                       # bản ghi mang cả dãy doc_id của partition
