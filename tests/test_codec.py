"""Kiểm tra codec bằng đúng các ví dụ tính tay trong slide Chương 4 (4-25 đến 4-29)."""
import numpy as np
import pytest

from index import codec


def test_dgap_slide_4_25():
    assert codec.to_gaps([3, 10, 11, 130]).tolist() == [3, 7, 1, 119]
    assert codec.from_gaps([3, 7, 1, 119]).tolist() == [3, 10, 11, 130]


def test_dgap_rejects_unsorted():
    with pytest.raises(ValueError):
        codec.to_gaps([5, 3])


@pytest.mark.parametrize("n,bits", [(1, "10000001"), (127, "11111111"), (128, "00000001 10000000"), (130, "00000001 10000010")])
def test_vb_slide_4_26(n, bits):
    assert " ".join(f"{b:08b}" for b in codec.vb_encode_number(n)) == bits


def test_vb_slide_4_27_hex():
    assert codec.vb_encode([3, 7, 1, 119]).hex(" ").upper() == "83 87 81 F7"


def test_vb_numpy_matches_reference():
    rng = np.random.default_rng(0)
    nums = np.concatenate([rng.integers(0, 2**31, 500), [0, 1, 127, 128, 16383, 16384]])
    buf = codec.vb_encode(nums)
    assert codec.vb_decode(buf).tolist() == codec.vb_decode_slow(buf) == nums.tolist()


@pytest.mark.parametrize("n,bits", [(1, "0"), (3, "101"), (7, "11011"), (119, "1111110110111")])
def test_gamma_slide_4_28(n, bits):
    assert codec.gamma_bits(n) == bits


def test_gamma_slide_4_29_total_bits():
    _, nbits = codec.gamma_encode([3, 7, 1, 119])
    assert nbits == 22


@pytest.mark.parametrize("name", codec.CODECS)
def test_postings_roundtrip(name):
    rng = np.random.default_rng(1)
    doc_ids = np.unique(rng.integers(0, 200_000, 3000))
    tfs = rng.integers(1, 20, doc_ids.size)
    buf = codec.encode_postings(doc_ids, tfs, name)
    d, t = codec.decode_postings(buf, doc_ids.size, name)
    assert d.tolist() == doc_ids.tolist() and t.tolist() == tfs.tolist()


def test_postings_with_doc_zero():
    for name in codec.CODECS:
        d, t = codec.decode_postings(codec.encode_postings([0, 5], [2, 1], name), 2, name)
        assert d.tolist() == [0, 5] and t.tolist() == [2, 1]


def test_compression_is_smaller_than_raw():
    doc_ids = np.arange(0, 100_000, 7)
    tfs = np.ones_like(doc_ids)
    raw = len(codec.encode_postings(doc_ids, tfs, "raw32"))
    vb = len(codec.encode_postings(doc_ids, tfs, "vb"))
    gamma = len(codec.encode_postings(doc_ids, tfs, "gamma"))
    assert gamma < vb < raw
