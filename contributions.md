# Contributions — Nhóm 3

Cập nhật mỗi tuần. Mỗi dòng ghi việc đã làm kèm bằng chứng (commit, PR, issue, file kết quả).
Phân vai lấy từ "Nhóm 3 - Big Data - 2026.docx". Việc cần chạy lại và giải thích
của từng vai trò: `docs/role_guides.md`.

| Vai trò | Người chính | Dự phòng | Phần mã / tài liệu phụ trách | Phần bảo vệ |
|---|---|---|---|---|
| Data/extraction | Nguyễn Đức Hoàng (B25CHHT096) | Nguyễn Ngọc Minh | `scripts/download_data.py`, `src/extract/`, `scripts/make_subsets.py`, `scripts/make_sample.py`, gói chung N3/N4 (`src/shared.py`, `scripts/export_n4_shared.py`, `scripts/verify_n4_shared.py`, `configs/n4_shared_manifest.json`, `docs/n4_shared_benchmark.md`), `docs/data_card.md`, `docs/license_note.md` | Nguồn, giấy phép, data contract và gói chung với Nhóm 4, dead-letter, vì sao 115K premise đúng bị loại |
| Indexing | Trần Duy Khánh (B25CHHT100) | Vũ Trung Kiên | `src/index/` (build_seq, build_spark, codec), `scripts/run_scale.py --part build` | MapReduce build, partition/skew/salting, nén d-gap/VB/gamma, overhead Spark trên Windows |
| BM25 baseline | Nguyễn Mạnh Cường (B25CHHT082) | Trần Đình Dũng | `src/analysis/tokenizer.py`, `src/retrieval/search.py`, `scripts/tune_bm25.py`, configs B0/B0t/V1/V2 | Công thức BM25/BM25F, tokenizer Lean, tinh tham số chỉ trên val |
| Structural features | Trần Đình Dũng (B25CHHT089) | Nguyễn Mạnh Cường | `src/retrieval/features.py`, `rerank.py`, `scripts/build_dependency_stats.py`, `scripts/train_reranker.py` | Đặc trưng cấu trúc, PageRank, bộ lọc truy cập được, chống rò rỉ train A/B |
| Serving/performance | Vũ Trung Kiên (B25CHHT101) | Trần Duy Khánh | `src/serving/`, `src/eval/loadtest.py`, `scripts/run_scale.py --part serve`, môi trường `.venv`/`.venv-spark` | API, warm-up, GIL và nhiều worker, P95 dưới tải |
| Evaluation/error analysis | Nguyễn Ngọc Minh (B25CHHT106) | Nguyễn Đức Hoàng | `src/eval/metrics.py`, `run_eval.py`, `scripts/summarize_quality.py`, `scripts/ablation.py`, `scripts/error_analysis.py`, `scripts/make_figures.py`, `scripts/eval_shared.py`, `docs/benchmark_spec.md` | Metric, bootstrap, ablation, 30 failure case, so sánh với Nhóm 4 |

## Nhật ký

| Tuần | Ngày | Ai | Việc | Bằng chứng |
|---|---|---|---|---|
| 1–9 | 2026-10-17 | (AI hỗ trợ, xem docs/ai_use_statement.md) | Dựng repo, pipeline B0–V4, Spark build, API, thí nghiệm quy mô (build 78 lần chạy, tải 36 lần), ablation, error analysis, bản nháp báo cáo | toàn bộ repo; results/raw; results/figures; report/ |
| 8 | 2026-10-17 | (AI hỗ trợ) | Phát hiện và sửa lỗi bộ lọc để lọt chính theorem (42% truy vấn test); thêm test hồi quy; chạy lại V2a/V4 | tests/test_accessible.py; docs/benchmark_spec.md |
| 1, 2, 5 | 2026-10-18 | (AI hỗ trợ) | Bổ sung sản phẩm còn thiếu: 20 truy vấn kiểm tra, data dictionary, ablation plan, latency profile theo bước; KTC bootstrap cho ablation; sửa lỗi làm tròn hai lần trong ablation.py (sai lệch ≤ 0,001); hướng dẫn theo vai trò | docs/check_queries.md; docs/data_dictionary.md; docs/ablation_plan.md; results/figures/latency_profile.md; results/figures/ablation_significance.md; docs/role_guides.md |
| 7 | 2026-10-18 | (AI hỗ trợ) | Viết lại job Spark (engine Arrow: executor đọc Parquet, gộp posting theo partition, shuffle trong JVM); thí nghiệm build2 trên Windows và Linux (Docker), 194 lần chạy, chỉ mục trùng từng byte; Spark nhanh hơn tuần tự tới 3,84 lần ở 3,6 triệu bản ghi | src/index/build_spark_arrow.py; tests/test_spark_arrow_logic.py; docker/; results/figures/scale_tables.md |
| 5, 6, 8 | 2026-10-18 | (AI hỗ trợ) | V5: ứng viên bổ sung theo lịch sử sử dụng (chỉ theorem có trước), chọn trên validation; test random R@10 0,243 → 0,281. Baseline Lucene: cùng kết quả với B0, nhanh hơn khoảng 5 lần. Thí nghiệm 4/8/12 worker và kiểm tra CPU: dịch vụ bị giới hạn bởi CPU. Báo cáo rút về 20 trang phần thân | configs/v5_rerank_expand.yaml; scripts/tune_v5.py; scripts/lucene_baseline.py; scripts/serve_cpu_check.py; tests/test_usage_causal.py |
| 1 đến 8 | 2026-10-01 đến 10-15 | TV6 (Evaluation), theo khai báo cũ trong `docs/ai_use_statement.md` | Pipeline mức theorem trước khi gộp repo: data contract v1, mock end-to-end, corpus 180.907 premise từ Zenodo, 4 hệ BM25/hybrid, load test, ablation, failure case, bản nháp báo cáo. Đã thay bằng `src/` và gỡ khỏi cây mã | lịch sử Git đến b1fbb35 |
| 9 | 2026-10-20 | Nguyễn Đức Hoàng | Ghim và kiểm tra gói benchmark chung N3/N4 (`n4_shared`: 180.907 premise, 898 truy vấn, SHA-256, schema, quy tắc eligible) | 661a392, 20d0cc5, b1fbb35; `tests/test_shared_data.py` |
| 9 | 2026-10-20 | (AI hỗ trợ) | Gộp repo big-data-n3 vào lean-premise-retrieval, giữ lịch sử của cả hai; gỡ pipeline cũ trùng với `src/`; dựng lại gói chung từ dữ liệu gốc đúng từng byte; đánh giá B0, V2, V4, V5 trên gói chung; phát hiện 158 truy vấn còn chứa chứng minh; sửa kiểm tra SHA-256 đầu vào báo cáo để không phụ thuộc ký tự xuống dòng | merge 4fbc9ca và các commit sau; `scripts/eval_shared.py`; `results/figures/shared_benchmark.md` |
| | | | | |
