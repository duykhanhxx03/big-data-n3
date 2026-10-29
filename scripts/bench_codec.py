"""Đo chi phí giải mã posting theo codec khi KHÔNG có cache (bổ sung cho ablation codec).

1. Giải mã toàn bộ posting của index B0 (đầy đủ) → số posting/giây
2. Truy vấn B0 trên mẫu 500 truy vấn với cache tắt → P50/P95 một luồng
Lặp 3 lần, ghi run record kind=codec và bảng results/figures/codec.md
"""
from __future__ import annotations

import time

import numpy as np

import runlog
from eval.run_eval import ROOT, ensure_index, load_eval
from index import codec as codec_mod
from retrieval.search import BM25FParams, BM25FSearcher, InvertedIndex


def main() -> None:
    q, _ = load_eval("scale_random_test_500")
    md = ["# Codec posting khi tắt cache (sinh bởi scripts/bench_codec.py)\n",
          "| Codec | Postings (MB) | Giải mã toàn bộ (s) | Triệu posting/s | P50 truy vấn (ms) | P95 truy vấn (ms) |",
          "|---|---|---|---|---|---|"]
    for name in ("raw32", "vb", "gamma"):
        path = ensure_index({"fields": ["statement"], "tokenizer": "default", "codec": name})
        decode_s, p50s, p95s = [], [], []
        for rep in range(3):
            idx = InvertedIndex(path, cache_terms=0)
            raw = np.fromfile(path / "postings.bin", dtype=np.uint8)
            t = time.perf_counter()
            for (_, _), (df, off, nb) in idx.dictionary.items():
                codec_mod.decode_postings(raw[off:off + nb].tobytes(), df, name)
            decode_s.append(time.perf_counter() - t)
            s = BM25FSearcher(idx, BM25FParams(k1=1.2, b={"statement": 0.75}))
            lat = []
            for text in q.state_before:
                t = time.perf_counter()
                s.search(text, k=100)
                lat.append((time.perf_counter() - t) * 1000)
            p50s.append(float(np.percentile(lat, 50)))
            p95s.append(float(np.percentile(lat, 95)))
        n_post = idx.manifest["n_postings"]
        metrics = {"postings_bytes": idx.manifest["postings_bytes"], "decode_all_s": decode_s,
                   "postings_per_s": [n_post / d for d in decode_s], "query_p50_ms": p50s, "query_p95_ms": p95s}
        runlog.write_record("codec", parameters={"codec": name, "index": path.name, "cache_terms": 0,
                                                 "eval_set": "scale_random_test_500"}, metrics=metrics)
        md.append(f"| {name} | {idx.manifest['postings_bytes'] / 2**20:.1f} | {np.median(decode_s):.2f} | "
                  f"{n_post / np.median(decode_s) / 1e6:.2f} | {np.median(p50s):.1f} | {np.median(p95s):.1f} |")
        print(md[-1], flush=True)
    md.append("\nMedian của 3 lần. Cache posting đã giải mã bị tắt (cache_terms = 0) để thấy chi phí giải mã thật.")
    (ROOT / "results" / "figures" / "codec.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
