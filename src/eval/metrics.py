"""Metric truy hồi theo từng truy vấn và trung bình.

- Recall@k = |gold ∩ top-k| / |gold|  (định nghĩa dùng trong LeanDojo/ReProver)
- MRR      = 1 / hạng của premise đúng đầu tiên (0 nếu không có trong danh sách trả về)
- nDCG@k   = nhãn nhị phân, log2
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

DEFAULT_KS = (1, 5, 10, 50)


def recall_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    if not gold:
        raise ValueError("truy vấn không có premise đúng")
    return len(gold.intersection(ranked[:k])) / len(gold)


def reciprocal_rank(ranked: Sequence[int], gold: set[int]) -> float:
    for i, pid in enumerate(ranked, start=1):
        if pid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, pid in enumerate(ranked[:k], start=1) if pid in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def per_query(ranked: Sequence[int], gold: set[int], ks: Iterable[int] = DEFAULT_KS) -> dict[str, float]:
    row = {f"R@{k}": recall_at_k(ranked, gold, k) for k in ks}
    row["MRR"] = reciprocal_rank(ranked, gold)
    row["nDCG@10"] = ndcg_at_k(ranked, gold, 10)
    return row


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys() if rows else []
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def _cluster_sums(a: Sequence[float], b: Sequence[float], clusters: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chênh lệch từng truy vấn d = b - a, tổng d và số truy vấn của từng cụm."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or len(clusters) != a.size:
        raise ValueError("a, b và clusters phải cùng độ dài")
    d = b - a
    _, inv = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    return d, np.bincount(inv, weights=d), np.bincount(inv).astype(float)


def paired_cluster_bootstrap(a: Sequence[float], b: Sequence[float], clusters: Sequence, n_boot: int = 10000,
                             seed: int = 42, estimand: str = "query") -> dict[str, float]:
    """Khoảng tin cậy bằng paired bootstrap theo cụm (ví dụ theorem): lấy mẫu cụm có hoàn lại, mang theo mọi truy vấn
    của cụm.

    estimand="query":   chênh lệch trung bình theo truy vấn (query-macro), giống metric chính của báo cáo
    estimand="cluster": chênh lệch trung bình của trung bình từng cụm (theorem-macro, mỗi theorem nặng như nhau)
    Trả delta (b - a) trên toàn bộ dữ liệu và khoảng tin cậy 95% theo phân vị. Hàm không trả giá trị p: kiểm định dùng
    paired_cluster_permutation, có giả thuyết không rõ ràng."""
    d, sums, sizes = _cluster_sums(a, b, clusters)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, sums.size, size=(n_boot, sums.size))
    if estimand == "query":
        point = d.mean()
        boots = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    elif estimand == "cluster":
        means = sums / sizes
        point = means.mean()
        boots = means[idx].mean(axis=1)
    else:
        raise ValueError("estimand phải là 'query' hoặc 'cluster'")
    return {"delta": float(point), "ci95_low": float(np.percentile(boots, 2.5)),
            "ci95_high": float(np.percentile(boots, 97.5)), "n_clusters": int(sums.size), "n_boot": n_boot}


def paired_cluster_permutation(a: Sequence[float], b: Sequence[float], clusters: Sequence, n_perm: int = 10000,
                               seed: int = 42, estimand: str = "query", alternative: str = "two-sided",
                               exact_max: int = 16) -> dict[str, float]:
    """Kiểm định hoán vị ghép cặp theo cụm (đổi dấu theo cụm).

    H0: trong mỗi cụm, kết quả của hai hệ thống hoán đổi được cho nhau (nhãn a/b là tùy ý). Mỗi phép hoán vị đổi nhãn
    hai hệ thống cho toàn bộ truy vấn của một cụm, tức đổi dấu tổng chênh lệch của cụm đó, rồi tính lại đúng thống kê
    quan sát: trung bình theo truy vấn (estimand="query") hoặc theo cụm (estimand="cluster").

    alternative="two-sided": p = P(|T*| >= |T|); "greater": p = P(T* >= T) với giả thuyết đối b > a.
    Nếu số cụm có chênh lệch khác 0 không vượt exact_max thì duyệt hết 2^m phép đổi dấu (p chính xác, gồm cả phép đồng
    nhất). Nếu không, lấy n_perm phép đổi dấu ngẫu nhiên và dùng p = (1 + số lần cực trị) / (1 + n_perm), nên p không bao
    giờ bằng 0 và không nhỏ hơn 1 / (1 + n_perm). Hai giá trị bằng nhau trong sai số dấu phẩy động được tính là cực trị."""
    d, sums, sizes = _cluster_sums(a, b, clusters)
    if estimand == "query":
        w = sums / d.size
    elif estimand == "cluster":
        w = (sums / sizes) / sums.size
    else:
        raise ValueError("estimand phải là 'query' hoặc 'cluster'")
    if alternative not in ("two-sided", "greater"):
        raise ValueError("alternative phải là 'two-sided' hoặc 'greater'")
    t_obs = float(w.sum())
    w = w[w != 0]                      # cụm không đổi không ảnh hưởng thống kê
    tol = 1e-12 * max(1.0, abs(t_obs))

    def extreme(t: np.ndarray) -> np.ndarray:
        return np.abs(t) >= abs(t_obs) - tol if alternative == "two-sided" else t >= t_obs - tol

    if w.size <= exact_max:
        signs = 1 - 2 * ((np.arange(2 ** w.size)[:, None] >> np.arange(w.size)) & 1)
        hits = extreme(signs @ w)
        p, n_used, exact = float(hits.mean()), int(hits.size), True
    else:
        rng = np.random.default_rng(seed)
        count = 0
        for start in range(0, n_perm, 2000):       # theo lô để giới hạn bộ nhớ
            m = min(2000, n_perm - start)
            signs = rng.integers(0, 2, size=(m, w.size), dtype=np.int8) * 2 - 1
            count += int(extreme(signs @ w).sum())
        p, n_used, exact = (1 + count) / (1 + n_perm), n_perm, False
    return {"delta": t_obs, "p": float(min(1.0, p)), "alternative": alternative, "exact": exact, "n_perm": n_used,
            "n_clusters": int(sums.size)}


def holm(pvalues: Sequence[float]) -> list[float]:
    """Hiệu chỉnh Holm cho một họ giả thuyết; trả p đã hiệu chỉnh theo đúng thứ tự đầu vào."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 42) -> dict[str, float]:
    """So sánh hai hệ thống trên cùng tập truy vấn, mỗi truy vấn là một đơn vị độc lập: trả chênh lệch trung bình
    (b - a), khoảng tin cậy 95% theo bootstrap và p hai phía của kiểm định hoán vị ghép cặp theo từng truy vấn."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("hai hệ thống phải có cùng tập truy vấn")
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    perm = paired_cluster_permutation(a, b, np.arange(d.size), n_perm=n_boot, seed=seed)
    return {
        "delta": float(d.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "p_value": perm["p"],
    }
