# Ablation plan — Nhóm 3 (sản phẩm tuần 5)

Mục đích: mỗi thành phần của hệ thống phải chứng minh được đóng góp bằng một thí nghiệm bỏ đúng thành phần đó, trả lời hai
câu hỏi nghiên cứu của đề:

- **RQ1 (câu 15 của hướng dẫn).** BM25 có đủ mạnh cho dữ liệu Lean, hay cần thêm trường ký hiệu, namespace và dependency?
- **RQ2 (câu 16).** Kết hợp điểm sparse với tín hiệu cấu trúc có cải thiện Recall@k mà vẫn giữ độ trễ thấp không?

Các biến thể được thiết kế ở tuần 5 và cài thành file trong `configs/` cùng `scripts/ablation.py`. Tài liệu này được viết
lại thành văn bản ngày 2026-10-18, sau khi đã chạy; cột *Kết quả* điền từ các file trong `results/figures/`.

## 1. Quy tắc chung

| Quy tắc | Nội dung |
|---|---|
| Một thay đổi mỗi lần | Mỗi biến thể chỉ khác tham chiếu ở đúng một thành phần. |
| Tham chiếu | Chuỗi tích lũy: so với bước liền trước. Leave-one-out: so với V4 đầy đủ. |
| Tinh chỉnh | Mọi tham số (k1, b, trọng số field, K, C của hồi quy) chọn trên `random_val_all`. Test chỉ chạy cấu hình đã chốt. |
| Huấn luyện lại | Mỗi biến thể của reranker huấn luyện lại trên cùng 6.000 truy vấn train nửa B (seed 42), stage 1 và bộ lọc như lúc chạy. |
| Chống rò rỉ | Thống kê phụ thuộc lấy từ train nửa A; `novel_premises` dùng train riêng của nó (`docs/benchmark_spec.md` §4). |
| Metric chính | Recall@10 (chất lượng) và P95 ở 10 client (độ trễ). Metric phụ: R@1/5/50, MRR, nDCG@10, QPS, kích thước index, thời gian build. |
| Kiểm định | Khoảng tin cậy: paired bootstrap theo cụm theorem, 10.000 mẫu, seed 42 (`eval.metrics.paired_cluster_bootstrap`). Giá trị p: kiểm định hoán vị ghép cặp theo cụm theorem (`paired_cluster_permutation`), đổi nhãn hai hệ thống cho mọi truy vấn của một theorem, 10.000 phép hoán vị, p hai phía = (1 + số lần cực trị) / 10.001. Trước ngày 2026-10-19, p được lấy từ tỉ lệ đuôi của bootstrap; cách đó sai khi chênh lệch bằng 0 nên đã bị thay (bản đánh giá lần 2, mục 3). |
| Quy tắc kết luận | Leave-one-out (mục B, C): p hai phía hiệu chỉnh Holm trên 9 phép so sánh của từng metric; p Holm < 0,05 → thành phần có tác dụng, ngược lại là "không phân biệt được", không kết luận là vô dụng. Chuỗi tích lũy (mục A): p một phía, Holm trên 7 bước trong từng split. Các so sánh còn lại báo KTC 95% và p chưa hiệu chỉnh. |
| Thời gian chạy | Mọi phép đo thời gian lặp ít nhất 3 lần; báo trung vị và khoảng min–max. |

## 2. Các trục ablation

### A. Chuỗi tích lũy chất lượng (RQ1)

Đánh giá trên `random_test_all` (2.641 truy vấn) và `novel_premises_test_all` (4.141). Nguồn:
`results/figures/robustness/significance_cluster.csv` (KTC bootstrap theo cụm, p hoán vị theo cụm, Holm trên 7 bước).

| Bước | Thay đổi duy nhất | Giả thuyết | Kết quả random: ΔR@10 [KTC 95%], p Holm |
|---|---|---|---|
| B0 → B0t | tinh k1, b, query_tf trên val | Tham số mặc định chưa hợp với proof state dài | +0,023 [+0,013; +0,033], 0,0007 ✓ |
| B0t → V1 | tokenizer Lean (tên có dấu chấm, camelCase, ký hiệu) | Tokenizer `\w+` làm mất ký hiệu và tên ghép | +0,010 [+0,0001; +0,022], 0,053: không phân biệt được (MRR có ý nghĩa) |
| V1 → V2 | BM25F ba trường name / namespace / statement | Tên và namespace là tín hiệu riêng, không nên trộn vào statement | +0,013 [+0,006; +0,020], 0,0007 ✓ |
| V2 → V3 | reranker 11 đặc trưng cấu trúc | Vị trí, namespace, ký hiệu, loại khai báo sắp lại top-100 tốt hơn BM25 | +0,021 [+0,010; +0,033], 0,0007 ✓ |
| V2 → V2a | bộ lọc premise truy cập được | Premise không import được không thể là đáp án | +0,036 [+0,029; +0,044], 0,0007 ✓ |
| V2a → V4 | reranker 13 đặc trưng gồm popularity, PageRank | Premise được dùng nhiều trong train có xác suất cao hơn | +0,060 [+0,046; +0,074], 0,0007 ✓ |

Trên `novel_premises`, B0t → V1 và V4 → V5 có p Holm 0,094 (không phân biệt được); các bước còn lại có ý nghĩa.

### B. Leave-one-group-out trên reranker V4 (RQ2)

Nguồn: `results/figures/ablation.md`, `ablation_significance.md`. Bỏ một nhóm đặc trưng rồi huấn luyện lại.

| Biến thể | Nhóm bị bỏ | Giả thuyết | Kết quả ΔR@10 / ΔMRR |
|---|---|---|---|
| `abl_no_bm25` | bm25_rel, bm25_rr | Reranker vẫn cần điểm stage 1 | −0,036 / −0,046, có ý nghĩa |
| `abl_no_namespace` | ns_in_query, ns_thm | Premise cùng namespace với theorem dễ đúng hơn | ≈ 0 / ≈ 0, không phân biệt được |
| `abl_no_location` | same_file, dir_overlap | Premise ở gần theorem dễ đúng hơn | ≈ 0 / −0,010, chỉ có ý nghĩa ở MRR |
| `abl_no_symbols` | sym_jaccard | Goal và premise chung ký hiệu toán | −0,009 / −0,008, có ý nghĩa |
| `abl_no_kind` | kind_thm/def/inst/type | Theorem/lemma được dùng làm premise nhiều hơn def | −0,001 / −0,008, chỉ có ý nghĩa ở MRR |
| `abl_no_dependency` | popularity, pagerank | Tín hiệu phụ thuộc từ train | −0,018 / −0,015, có ý nghĩa |

### C. Stage 1 và bộ lọc

| Biến thể | Thay đổi | Giả thuyết | Kết quả ΔR@10 |
|---|---|---|---|
| `abl_k50` | K = 50 ứng viên | Ít ứng viên hơn làm mất premise đúng ở hạng 51–100 | −0,009, có ý nghĩa |
| `abl_k200` | K = 200 ứng viên | Recall bị giới hạn bởi stage 1 | +0,010, có ý nghĩa (R@50 +0,038) |
| `abl_no_filter` | tắt bộ lọc (V4 không có đặc trưng `accessible`, nên mất hẳn tín hiệu này) | Premise không import được chiếm chỗ của premise đúng trong top-100 | −0,051, có ý nghĩa |

### D. Quy mô và hệ thống (yêu cầu mục 8 của hướng dẫn)

| Trục | Mức | Đo | Nguồn |
|---|---|---|---|
| Kích thước index | 10K ⊂ 100K ⊂ toàn bộ (181K); ×5 (905K) cho build | R@10, P95, kích thước, thời gian build | `scale_tables.md`, `fig_quality_vs_size.png` |
| BM25 mặc định vs field-aware | B0 vs V2 | R@10, P95 | `quality.md`, `scale_tables.md` |
| Có / không ký hiệu Lean | B0t vs V1 | R@10 | mục A |
| Có / không structural reranking | V2 vs V3, V2a vs V4 | R@10, P95 | mục A, `latency_profile.md` |
| Số truy vấn đồng thời | 1 / 10 / 50 client, 3 lần | P50/P95/P99, QPS, lỗi | `scale_tables.md`, `fig_latency_vs_clients.png` |
| Builder, engine RDD | tuần tự; Spark local[2/4/8]; tới 905K; Windows | thời gian, RAM đỉnh, shuffle | `scale_tables.md`, `fig_build_time.png` |
| Builder, engine Arrow | tuần tự; Spark local[2/4/8]; chỉ mục B0 và 3 trường; tới 3,6 triệu; Windows và Linux | thời gian, tăng tốc, RAM, bản ghi shuffle, trùng byte | `scale_tables.md`, `fig_build2_*.png` |
| Engine | RDD so với Arrow (local[4], B0, 905K và 3,6 triệu) | thời gian, RAM đỉnh | `scale_tables.md` |
| Kết hợp cục bộ | bật / tắt (Spark local[4]) | shuffle ghi, thời gian | `scale_tables.md` |
| Salting hot key | không / 4 khoảng với df ≥ 20.000 | task reduce max/median, thời gian | `scale_tables.md` |
| Codec posting | raw32 / VB / gamma | kích thước, P50 có và không có cache | `ablation.md`, `codec.md` |

### E. Bổ sung sau tuần 8 (V5, Lucene, dịch vụ)

| Trục | Các mức | Nguồn |
|---|---|---|
| Ứng viên bổ sung của V5 | 8 tổ hợp (tệp, thư mục, phổ biến), chọn trên validation; đo cả độ phủ ứng viên | `results/raw/tuning/V5__extra__random_val_all.csv` |
| BM25 của Lucene | bộ phân tích tiếng Anh mặc định; token của B0 với cùng k1, b | `results/figures/lucene.md` |
| Số worker của API | 4, 8, 12; 1 đến 100 client; kèm đo CPU | `scale_tables.md`, `scripts/serve_cpu_check.py` |

## 3. Giới hạn của kế hoạch

- **Nhiều phép so sánh.** Mục B và C có 9 phép kiểm định. Với bootstrap theo cụm và hiệu chỉnh Holm
  (`results/figures/ablation_significance.csv`), hai kết luận yếu nhất về Recall@10 không còn đứng: `abl_no_symbols` và
  `abl_k50` (p Holm = 0,132), dù vẫn có ý nghĩa với MRR; `abl_k200` còn p Holm = 0,019.
- **Đối chứng bổ sung sau phản biện.** V4-K240 (cùng ngân sách ứng viên với V5), V2-nons (BM25F không có trường namespace),
  V4-poponly và V4-pronly (tách hai đặc trưng phụ thuộc); cấu hình trong `configs/review/`, chạy bằng
  `scripts/review_experiments.py`, kết quả trong `results/figures/robustness.md`. Tập holdout mới
  (`scripts/make_holdout.py`) kiểm tra V5 vì V5 được thiết kế sau khi xem lỗi trên tập kiểm tra.
- Leave-one-out chỉ chạy trên split `random`; chưa lặp lại trên `novel_premises`.
- Các thành phần bên trong tokenizer Lean (tách camelCase, giữ tên đầy đủ, bỏ biến một chữ cái) chưa được tách riêng.
- Reranker là hồi quy logistic; chưa so với mô hình phi tuyến (ví dụ gradient boosting) nên kết luận "nhóm X không có tác
  dụng" chỉ đúng với mô hình tuyến tính.
- Chỉ có một máy đo (6 nhân). Thí nghiệm xây chỉ mục có thêm Linux trong container trên cùng phần cứng, nhưng chưa có
  cụm nhiều máy nên chưa đo chi phí mạng của shuffle.

## 4. Tái tạo

```bash
python scripts/ablation.py               # huấn luyện lại và đánh giá 10 biến thể + 3 codec
python scripts/ablation_significance.py  # KTC và p từ run record
python scripts/profile_latency.py        # thời gian theo giai đoạn
python scripts/run_scale.py --part build|quality|serve   # mục D
```
