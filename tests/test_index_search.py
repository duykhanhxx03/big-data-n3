"""Index tự xây phải cho cùng điểm với công thức BM25/BM25F tính trực tiếp (oracle), với mọi codec."""
import math
from collections import Counter

import pyarrow.parquet as pq
import pytest

from analysis.tokenizer import get_tokenizer
from index import build_seq
from retrieval.search import BM25FParams, BM25FSearcher, InvertedIndex

DOCS = [  # ví dụ slide 4-12, thêm field name
    {"statement": "data science data", "name": "Data.science"},
    {"statement": "science search", "name": "Search.basic"},
    {"statement": "data search search", "name": "Data.search"},
    {"statement": "search index", "name": "Index.build"},
]


def oracle(query, docs, fields, weights, k1=1.2, b=0.75, tokenizer="default"):
    tok = get_tokenizer(tokenizer)
    tfs = [{f: Counter(tok(d[f])) for f in fields} for d in docs]
    n = len(docs)
    avg = {f: sum(sum(t[f].values()) for t in tfs) / n for f in fields}
    scores = []
    for t in tfs:
        s = 0.0
        for term, qtf in Counter(tok(query)).items():
            df = sum(1 for x in tfs if any(term in x[f] for f in fields))
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tt = sum(weights.get(f, 1.0) * t[f][term] / (1 - b + b * sum(t[f].values()) / avg[f]) for f in fields)
            if tt:
                s += qtf * idf * tt * (k1 + 1) / (tt + k1)
        scores.append(s)
    return scores


def build_tmp(tmp_path, fields, codec, tokenizer="default"):
    out = tmp_path / f"idx_{codec}_{'_'.join(fields)}"
    build_seq.build(((i, d) for i, d in enumerate(DOCS)), len(DOCS), out, fields=fields, tokenizer=tokenizer, codec=codec)
    return InvertedIndex(out)


def test_index_matches_slide_4_13(tmp_path):
    idx = build_tmp(tmp_path, ["statement"], "vb")
    d, tf = idx.postings("statement", "search")
    assert d.tolist() == [1, 2, 3] and tf.tolist() == [1, 2, 1]  # (2,1),(3,2),(4,1) với docID từ 0
    dic = pq.read_table(idx.path / "dictionary.parquet").to_pandas().set_index("term")
    assert dic.loc["search", "df"] == 3 and dic.loc["search", "cf"] == 4
    assert dic.loc["data", "df"] == 2 and dic.loc["data", "cf"] == 3
    assert dic["cf"].sum() == 10 and dic["df"].sum() == 8  # kiểm tra tổng ở slide 4-13


@pytest.mark.parametrize("codec", ["raw32", "vb", "gamma"])
@pytest.mark.parametrize("fields,weights", [(["statement"], {}), (["statement", "name"], {"name": 2.0})])
def test_scores_match_oracle(tmp_path, codec, fields, weights):
    idx = build_tmp(tmp_path, fields, codec)
    s = BM25FSearcher(idx, BM25FParams(weights=weights))
    for query in ["data search", "science", "search search index", "nothing here"]:
        expected = oracle(query, DOCS, fields, weights)
        got = dict(s.search(query, k=10))
        for doc_id, e in enumerate(expected):
            assert got.get(doc_id, 0.0) == pytest.approx(e, abs=1e-9)


def test_ranking_is_deterministic_on_ties(tmp_path):
    idx = build_tmp(tmp_path, ["statement"], "vb")
    a = BM25FSearcher(idx).search("science", k=10)
    assert [d for d, _ in a] == [1, 0]  # doc 1 ngắn hơn nên điểm cao hơn


def test_build_rejects_non_contiguous_ids(tmp_path):
    with pytest.raises(ValueError):
        build_seq.build(iter([(0, DOCS[0]), (2, DOCS[1])]), 2, tmp_path / "bad", fields=["statement"])
