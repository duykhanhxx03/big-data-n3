"""Latency profile theo giai đoạn (tuần 5): thời gian mỗi bước của pipeline cho B0, V2 và V4.

Chạy lại đúng các bước của BM25FSearcher.search và StructuralReranker.search nhưng bấm giờ từng bước,
rồi kiểm tra kết quả trùng với đường chạy thật. Một luồng, index toàn bộ, mẫu 500 truy vấn random/test.
Mỗi cấu hình chạy hai lượt: lượt 1 cache posting rỗng (lạnh), lượt 2 dùng cache đã đầy (nóng).

Dùng:  python scripts/profile_latency.py
Ra:    results/raw/profile/latency_profile.parquet, results/figures/latency_profile.md
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import runlog  # noqa: E402
from eval import run_eval  # noqa: E402
from retrieval.features import build_features  # noqa: E402
from retrieval.rerank import StructuralReranker  # noqa: E402
from retrieval.search import InvertedIndex  # noqa: E402

CONFIGS = ["b0_bm25_default", "v2_bm25f_lean", "v4_rerank_dep"]
EVAL = "scale_random_test_500"
STAGES = ["tokenize", "bm25_score", "accessible_mask", "filter_topk", "features", "rerank_sort"]
K = 100


def topk(searcher, ids, scores, k, mask):
    """Giống phần cuối BM25FSearcher.search."""
    if mask is not None and ids.size:
        keep = mask[searcher.index.doc_to_premise[ids]]
        ids, scores = ids[keep], scores[keep]
    if ids.size == 0:
        return []
    if ids.size > k:
        top = np.argpartition(-scores, k - 1)[:k]
        ids, scores = ids[top], scores[top]
    order = np.lexsort((ids, -scores))
    pids = searcher.index.doc_to_premise[ids[order]]
    return [(int(p), float(s)) for p, s in zip(pids, scores[order])]


def profile_one(searcher, text: str, ctx: dict) -> tuple[dict, list]:
    t = {s: 0.0 for s in STAGES}
    rr = searcher if isinstance(searcher, StructuralReranker) else None
    s1 = rr.stage1 if rr else searcher
    c0 = time.perf_counter()
    q = Counter(s1.tokenize(text))
    c1 = time.perf_counter()
    ids, scores = s1.score_terms(q)
    c2 = time.perf_counter()
    t["tokenize"], t["bm25_score"] = c1 - c0, c2 - c1
    access = None
    if rr and (rr.filter or (rr.model and "accessible" in rr.model.features)):
        access = rr.meta.accessible_mask(ctx.get("file_path", ""), ctx.get("theorem_start_line", 10**9))
    c3 = time.perf_counter()
    t["accessible_mask"] = c3 - c2
    hits = topk(s1, ids, scores, rr.k_cand if rr else K, access if rr and rr.filter else None)
    c4 = time.perf_counter()
    t["filter_topk"] = c4 - c3
    if rr and rr.model is not None and hits:
        cand = np.array([p for p, _ in hits], dtype=np.int64)
        sc = np.array([s for _, s in hits], dtype=np.float64)
        X = build_features(rr.meta, cand, sc, text, ctx, rr.model.features, access)
        c5 = time.perf_counter()
        z = rr.model.score(X)
        order = np.lexsort((np.arange(cand.size), -z))[:K]
        hits = [(int(cand[i]), float(z[i])) for i in order]
        c6 = time.perf_counter()
        t["features"], t["rerank_sort"] = c5 - c4, c6 - c5
    return {s: v * 1000 for s, v in t.items()}, hits


def main() -> None:
    queries, _ = run_eval.load_eval(EVAL)
    ctxs = run_eval.query_contexts(queries)
    rows, checks = [], []
    for name in CONFIGS:
        cfg = run_eval.load_config(ROOT / "configs" / f"{name}.yaml")
        path = run_eval.ensure_index(cfg["index"], "full")
        searcher = run_eval.make_searcher(cfg, InvertedIndex(path))
        # Kiểm tra: kết quả của đường bấm giờ phải trùng đường chạy thật (đường thật chạy trên index riêng
        # để không làm nóng cache của lượt lạnh).
        ref = run_eval.make_searcher(cfg, InvertedIndex(path))
        same = all([p for p, _ in profile_one(ref, txt, c)[1]] == [p for p, _ in ref.search(txt, k=K, ctx=c)]
                   for txt, c in list(zip(queries.state_before, ctxs))[:50])
        checks.append((cfg["name"], same))
        for rep, label in ((1, "lạnh"), (2, "nóng")):
            for qid, txt, c in zip(queries.query_id, queries.state_before, ctxs):
                st, _ = profile_one(searcher, txt, c)
                rows.append({"config": cfg["name"], "pass": label, "query_id": qid, **st,
                             "total": sum(st.values())})
        print(cfg["name"], "xong; kết quả trùng đường chạy thật:", same, flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "raw" / "profile"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "latency_profile.parquet", index=False)

    lines = ["# Latency profile theo giai đoạn (sinh bởi scripts/profile_latency.py)", "",
             f"Một luồng, index toàn bộ (180.957 premise), {queries.shape[0]} truy vấn `{EVAL}`. Đơn vị: ms. "
             "Lượt *lạnh*: cache posting rỗng lúc bắt đầu; lượt *nóng*: chạy lại cùng truy vấn với cache đã đầy. "
             "Tổng của trung vị từng bước không bằng trung vị của tổng. Cache LRU giữ 4.096 posting đã giải mã nên đầy sau vài "
             "truy vấn đầu; vì vậy hai lượt gần như bằng nhau và phần còn lại là chi phí cộng điểm trên posting dài.", ""]
    for label in ("lạnh", "nóng"):
        lines += [f"## Lượt {label}: trung vị (P95) theo bước", "",
                  "| Biến thể | " + " | ".join(STAGES) + " | Tổng | Tỉ lệ BM25 trong tổng (trung bình) |",
                  "|---" * (len(STAGES) + 3) + "|"]
        for name in df.config.unique():
            d = df[(df.config == name) & (df["pass"] == label)]
            cells = []
            for s in STAGES + ["total"]:
                if d[s].max() == 0:
                    cells.append("—")
                else:
                    cells.append(f"{d[s].median():.2f} ({d[s].quantile(0.95):.2f})")
            share = d.bm25_score.sum() / d.total.sum()
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {share:.0%} |")
        lines.append("")
    lines += ["## Ý nghĩa các bước", "",
              "- `tokenize`: tách từ proof state (tokenizer mặc định hoặc Lean).",
              "- `bm25_score`: giải nén posting của mọi term trong truy vấn và cộng điểm BM25/BM25F.",
              "- `accessible_mask`: tính tập premise truy cập được qua bao đóng import (có cache theo file).",
              "- `filter_topk`: áp mặt nạ và chọn top-K bằng argpartition.",
              "- `features`: tính 13 đặc trưng cấu trúc cho 100 ứng viên.",
              "- `rerank_sort`: chuẩn hóa, nhân trọng số hồi quy logistic và sắp xếp lại.", "",
              "Kiểm tra đúng đắn (50 truy vấn đầu, so với `searcher.search`): "
              + ", ".join(f"{n}: {'trùng' if ok else 'KHÁC'}" for n, ok in checks) + "."]
    (ROOT / "results" / "figures" / "latency_profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {f"{n}_{lab}_p50_total": float(df[(df.config == n) & (df["pass"] == lab)].total.median())
               for n in df.config.unique() for lab in ("lạnh", "nóng")}
    runlog.write_record("profile", parameters={"configs": CONFIGS, "eval_set": EVAL, "k": K}, metrics=summary,
                        seed=42, extra={"output": "results/raw/profile/latency_profile.parquet",
                                        "checks": dict(checks)})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
