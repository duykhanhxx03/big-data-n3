"""Mã hóa posting list theo Chương 4: d-gap, Variable-byte, gamma.

Quy ước theo slide 4-26 đến 4-28:
- VB: mỗi byte mang 7 bit dữ liệu; bit đầu bằng 1 đánh dấu byte cuối của một số.
- Gamma: tiền tố gồm (L-1) bit 1 rồi bit 0, sau đó là L-1 bit thấp của n (bỏ bit đầu). Chỉ cho n ≥ 1.

Mọi codec phải thỏa decode(encode(x)) == x (kiểm tra trong tests/test_codec.py).
"""
from __future__ import annotations

import numpy as np

CODECS = ("raw32", "vb", "gamma")


# ---------- d-gap ----------
def to_gaps(doc_ids: np.ndarray) -> np.ndarray:
    doc_ids = np.asarray(doc_ids, dtype=np.int64)
    if doc_ids.size and (np.any(np.diff(doc_ids) <= 0) or doc_ids[0] < 0):
        raise ValueError("doc_id phải tăng ngặt và không âm")
    return np.diff(doc_ids, prepend=0)


def from_gaps(gaps: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(gaps, dtype=np.int64))


# ---------- Variable-byte ----------
def vb_encode_number(n: int) -> bytes:
    if n < 0:
        raise ValueError("VB chỉ mã số không âm")
    parts = []
    while True:
        parts.insert(0, n % 128)
        if n < 128:
            break
        n //= 128
    parts[-1] += 128
    return bytes(parts)


def vb_encode(numbers) -> bytes:
    return b"".join(vb_encode_number(int(x)) for x in numbers)


def vb_decode(buf: bytes) -> np.ndarray:
    """Giải mã VB bằng numpy (không vòng lặp Python theo từng byte)."""
    if not buf:
        return np.zeros(0, dtype=np.int64)
    arr = np.frombuffer(buf, dtype=np.uint8)
    last = (arr & 0x80) != 0
    if not last[-1]:
        raise ValueError("chuỗi VB bị cắt: byte cuối không có bit kết thúc")
    payload = (arr & 0x7F).astype(np.int64)
    group = np.concatenate(([0], np.cumsum(last)[:-1]))
    ends = np.flatnonzero(last)
    shift = (ends[group] - np.arange(arr.size)) * 7
    return np.bincount(group, weights=(payload << shift).astype(np.float64)).astype(np.int64)


def vb_decode_slow(buf: bytes) -> list[int]:
    """Bản tham chiếu đúng như giả mã slide 4-27, dùng để kiểm tra bản numpy."""
    out, n = [], 0
    for byte in buf:
        n = 128 * n + (byte & 127)
        if byte & 128:
            out.append(n)
            n = 0
    return out


# ---------- Gamma ----------
def gamma_bits(n: int) -> str:
    if n < 1:
        raise ValueError("gamma chỉ áp dụng cho n ≥ 1")
    b = bin(n)[3:]  # bỏ '0b' và bit đầu
    return "1" * len(b) + "0" + b


def gamma_encode(numbers) -> tuple[bytes, int]:
    """Trả (bytes, số bit thực). Byte cuối đệm bit 0."""
    bits = "".join(gamma_bits(int(x)) for x in numbers)
    nbits = len(bits)
    bits += "0" * (-nbits % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b"", nbits


def gamma_decode(buf: bytes, count: int) -> np.ndarray:
    bits = "".join(f"{b:08b}" for b in buf)
    out = np.empty(count, dtype=np.int64)
    i = 0
    for k in range(count):
        length = 0
        while bits[i] == "1":
            length += 1
            i += 1
        i += 1  # bit 0 phân cách
        out[k] = int("1" + bits[i:i + length], 2)
        i += length
    return out


# ---------- Posting list = (doc_ids, tfs) ----------
def encode_postings(doc_ids, tfs, codec: str) -> bytes:
    """Bố cục: [mã của doc_id hoặc d-gap][mã của tf]. Số phần tử (df) lưu trong dictionary."""
    doc_ids = np.asarray(doc_ids, dtype=np.int64)
    tfs = np.asarray(tfs, dtype=np.int64)
    if codec == "raw32":
        return doc_ids.astype("<u4").tobytes() + tfs.astype("<u4").tobytes()
    gaps = to_gaps(doc_ids)
    if codec == "vb":
        a, b = vb_encode(gaps), vb_encode(tfs)
        return len(a).to_bytes(4, "little") + a + b
    if codec == "gamma":
        # gap đầu tiên có thể bằng 0 (doc_id 0) nên cộng 1 trước khi mã gamma
        a, _ = gamma_encode(gaps + 1)
        b, _ = gamma_encode(tfs)
        return len(a).to_bytes(4, "little") + a + b
    raise ValueError(f"codec không hỗ trợ: {codec}")


def decode_postings(buf: bytes, df: int, codec: str) -> tuple[np.ndarray, np.ndarray]:
    if codec == "raw32":
        arr = np.frombuffer(buf, dtype="<u4").astype(np.int64)
        return arr[:df], arr[df:2 * df]
    split = int.from_bytes(buf[:4], "little")
    a, b = buf[4:4 + split], buf[4 + split:]
    if codec == "vb":
        return from_gaps(vb_decode(a)), vb_decode(b)
    if codec == "gamma":
        return from_gaps(gamma_decode(a, df) - 1), gamma_decode(b, df)
    raise ValueError(f"codec không hỗ trợ: {codec}")
