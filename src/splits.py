"""Chia đôi cố định theorem của train: nửa A tính thống kê phụ thuộc, nửa B huấn luyện reranker."""
import hashlib


def train_half(theorem_full_name: str) -> str:
    return "A" if int(hashlib.md5(theorem_full_name.encode()).hexdigest(), 16) % 2 == 0 else "B"
