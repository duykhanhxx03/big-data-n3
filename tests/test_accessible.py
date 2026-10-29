"""Bộ lọc premise truy cập được không được cho phép chính theorem đang chứng minh (lỗi phát hiện ở error analysis)."""
import pyarrow.parquet as pq
import pytest

from eval.run_eval import PROCESSED, load_eval, query_contexts
from retrieval.rerank import get_meta

pytestmark = pytest.mark.skipif(not (PROCESSED / "premises.parquet").exists(), reason="cần data/processed")


def test_theorem_itself_is_never_accessible_and_gold_always_is():
    meta = get_meta("random")
    prem = pq.read_table(PROCESSED / "premises.parquet", columns=["full_name", "path"]).to_pandas()
    key = {(f, p): i for i, (f, p) in enumerate(zip(prem.full_name, prem.path))}
    q, gold = load_eval("random_val_all")
    q = q.head(400)
    for qid, c in zip(q.query_id, query_contexts(q)):
        m = meta.accessible_mask(c["file_path"], c["theorem_start_line"])
        assert not m[key[(c["theorem_full_name"], c["file_path"])]]
        assert all(m[g] for g in gold[qid])
