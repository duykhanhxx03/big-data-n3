"""Nguồn ứng viên và đặc trưng sử dụng của V5 không được dùng thông tin "tương lai": chỉ đếm theorem train-A cùng file
khai báo trước theorem đang chứng minh, hoặc theorem ở các file cùng thư mục mà file hiện tại import."""
import numpy as np
import pyarrow.parquet as pq
import pytest

from eval.run_eval import PROCESSED
from retrieval.rerank import get_meta

pytestmark = pytest.mark.skipif(not (PROCESSED / "dep_usage_random.parquet").exists(),
                                reason="cần data/processed/dep_usage_random.parquet (scripts/build_dependency_stats.py)")


def _busiest_file():
    u = pq.read_table(PROCESSED / "dep_usage_random.parquet").to_pandas()
    path = u.groupby("file_path").theorem.nunique().idxmax()
    return path, u[u.file_path == path]


def test_file_usage_only_counts_earlier_theorems():
    meta = get_meta("random")
    path, rows = _busiest_file()
    first, last = rows.line.min(), rows.line.max()
    assert meta.usage_of("file", path, first)[0].size == 0           # chưa có theorem nào khai báo trước
    ids, cnt = meta.usage_of("file", path, last + 1)                  # sau theorem cuối: thấy mọi theorem của file
    expected = rows.groupby("premise_id").theorem.nunique()
    assert dict(zip(ids.tolist(), cnt.tolist())) == expected.to_dict()
    mid = int(np.median(rows.line))
    ids_mid, cnt_mid = meta.usage_of("file", path, mid)
    seen = rows[rows.line < mid].groupby("premise_id").theorem.nunique()
    assert dict(zip(ids_mid.tolist(), cnt_mid.tolist())) == seen.to_dict()


def test_dir_usage_only_uses_imported_files():
    meta = get_meta("random")
    path, _ = _busiest_file()
    u = pq.read_table(PROCESSED / "dep_usage_random.parquet").to_pandas()
    fi = meta.file_index[path]
    ok = meta._closure(fi)
    same_dir = u[u.file_path.map(meta.dir_of) == meta.dir_of(path)]
    allowed = same_dir[(same_dir.file_path == path) |
                       same_dir.file_path.map(lambda f: f != path and f in meta.file_index and ok[meta.file_index[f]])]
    allowed = allowed[(allowed.file_path != path) | (allowed.line < 10**9)]
    ids, cnt = meta.usage_of("dir", path, 10**9)
    assert dict(zip(ids.tolist(), cnt.tolist())) == allowed.groupby("premise_id").size().to_dict()


def test_extra_candidates_respect_mask_and_exclusion():
    meta = get_meta("random")
    path, rows = _busiest_file()
    line = int(rows.line.max()) + 1
    access = meta.accessible_mask(path, line)
    exclude = set(meta.usage_of("file", path, line)[0][:5].tolist())
    out = meta.extra_candidates(path, line, access, exclude, n_file=20, n_dir=20, n_pop=20)
    assert len(out) == len(set(out)) and not (set(out) & exclude)
    assert all(access[p] for p in out)
