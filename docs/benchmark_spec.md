# Benchmark specification v0.1 — Nhóm 3 (chờ Nhóm 4 xác nhận)

Trạng thái: **bản nháp do Nhóm 3 đề xuất**. Mọi mục dưới đây đã được cài đặt và chạy được trong repo; khi
Nhóm 4 xác nhận hoặc đề nghị sửa, nâng phiên bản (v0.2) và chạy lại. Những mục cần Nhóm 4 trả lời được đánh dấu **[cần chốt]**.

Cập nhật 2026-10-20: benchmark để so sánh với Nhóm 4 đã được ghim riêng ở mức theorem (gói `n4_shared`, xem
`docs/n4_shared_benchmark.md`, kết quả của Nhóm 3 ở `results/figures/shared_benchmark.md`). Spec này vẫn định nghĩa
benchmark mức tactic dùng cho mọi bảng trong báo cáo; các mục **[cần chốt]** chỉ còn áp dụng nếu hai nhóm muốn so sánh
thêm ở mức tactic.

## 1. Dữ liệu nguồn

| Mục | Giá trị |
|---|---|
| Bộ dữ liệu | LeanDojo Benchmark 4, phiên bản **v10** (công bố 2024-07-14, tác giả Kaiyu Yang) |
| Nơi tải | Zenodo record 12740403, tệp `leandojo_benchmark_4.tar.gz`, md5 `25e1ee60cd8925b9d2e8673ddcc34b4c` |
| Giấy phép | CC BY 2.0 (dataset); mã Lean gốc của mathlib4/Lean 4/Batteries: Apache-2.0 |
| mathlib4 commit | `29dcec074de168ac2bf835a77ef68bbe069194c5` (LeanDojo 2.0.0, tạo 2024-07-02) |
| Tải lại | `python scripts/download_data.py` (kiểm tra md5, ghi `data/raw/MANIFEST.json`) |

## 2. Corpus premise (tập ứng viên)

- Nguồn: `corpus.jsonl`, 5.674 file, **180.957 premise** sau khi bỏ 16 bản ghi trùng hoàn toàn (tên + file).
- Định danh: cặp `(full_name, path)`, vì có 57 tên trùng giữa Lean core và Mathlib (ví dụ `not_le`).
- `premise_id` = thứ tự của `(full_name, path)` sau khi sắp xếp tăng dần → ổn định với cùng snapshot.
- **Tập ứng viên mặc định: toàn bộ 180.957 premise.** Hệ thống được phép tự lọc (ví dụ chỉ giữ premise truy cập
  được qua import) như một phần phương pháp, nhưng phải báo rõ. **[cần chốt]**

## 3. Truy vấn và premise đúng

- Một truy vấn = `state_before` của một tactic trong `traced_tactics`.
- Premise đúng = các premise trong `annotated_tactic[1]`, ánh xạ sang `premise_id` qua `(full_name, def_path)`.
- Premise đúng **không có trong corpus** bị bỏ (115.736 lượt trên toàn bộ các split). Đây là đặc điểm của benchmark:
  lemma do `@[to_additive]` sinh (`add_comm`, `zero_add`), constructor (`Or.inl`), trường structure/class
  (`Iff.mp`, `Category.assoc`) không phải khai báo tường minh trong corpus. Truy vấn không còn premise đúng nào bị bỏ.
- Tactic không có premise nào được chú thích không tạo truy vấn.
- **Không** đưa text của tactic vào truy vấn (tactic chứa sẵn tên premise cần tìm).
- Ngữ cảnh được phép dùng lúc truy vấn: tên theorem đang chứng minh, file và dòng bắt đầu của nó.
- Chính theorem đang chứng minh có trong corpus (nó là một premise của thư viện). Hệ thống có bộ lọc truy cập được của Nhóm 3
  luôn loại nó (premise cùng file phải kết thúc trước dòng bắt đầu của theorem); hệ thống không lọc có thể trả về nó.
  Đề xuất mọi hệ thống loại nó khỏi kết quả. **[cần chốt]**
- Kiểm tra định nghĩa: 20 truy vấn `random/val` đối chiếu với tactic gốc trong `docs/check_queries.md`
  (`scripts/make_check_queries.py`); 20/20 đạt bốn kiểm tra tự động. Mô tả cột: `docs/data_dictionary.md`.

## 4. Split và tập đánh giá

Dùng nguyên split của LeanDojo. Split chính: `random`; split khó: `novel_premises`.

| Split | Theorem | Truy vấn (sau lọc) |
|---|---|---|
| random/train | 118.517 | 152.330 |
| random/val | 2.000 | 2.536 |
| random/test | 2.000 | 2.641 |
| novel_premises/train | 118.517 | 149.459 |
| novel_premises/val | 2.000 | 3.907 |
| novel_premises/test | 2.000 | 4.141 |

- Kết quả chất lượng chính: **toàn bộ** truy vấn test (`random_test_all`, `novel_premises_test_all`).
- Mọi tinh chỉnh tham số chỉ trên val. Test chỉ chạy cấu hình cuối.
- Train chỉ dùng để học/thống kê. Nhóm 3 chia đôi train theo md5 tên theorem: nửa A tính thống kê phụ thuộc,
  nửa B huấn luyện reranker (`src/splits.py`).
- Thống kê/mô hình dùng cho `novel_premises` phải lấy từ `novel_premises/train` (train của `random` chứa theorem
  thuộc test của `novel_premises`).

## 5. Index con cho thí nghiệm quy mô

- Mẫu 500 truy vấn `random/test`, seed 42 (`eval_sets/scale_random_test_500.npy`).
- Index con lồng nhau 10K ⊂ 100K ⊂ toàn bộ (181K): luôn chứa 870 premise đúng của mẫu này, phần còn lại lấy ngẫu nhiên
  với seed 42 (`scripts/make_subsets.py`). Tỉ lệ premise đúng: 8,7% ở 10K; 0,48% ở toàn bộ.
- Ba mức 10K, 100K và toàn bộ (181K) khớp thang Nhóm 4 dùng và Data/extraction đang chuẩn bị cho gói chung. Mức 50K đã
  chạy trước đây vẫn còn trong `results/raw/` nhưng không đưa vào báo cáo.

## 6. Metric và định dạng kết quả

- Recall@k = |gold ∩ top-k| / |gold|, k ∈ {1, 5, 10, 50}; MRR; nDCG@10 (nhãn nhị phân). Trung bình theo truy vấn.
- **Primary metrics: Recall@10 và P95 latency ở 10 client đồng thời.** **[cần chốt]**
- Kiểm định: paired bootstrap 10.000 mẫu trên metric từng truy vấn (`eval.metrics.paired_bootstrap`).
- Run file TSV: `query_id  premise_id  rank  score  system`, top-100 mỗi truy vấn (để trộn RRF ở tuần 9).
- Script đánh giá dùng chung: `src/eval/metrics.py` (có unit test bằng ví dụ tính tay). **[cần chốt: Nhóm 4 review]**

## 7. Đo độ trễ

- Load test 1 / 10 / 50 client, mỗi mức 400 request, warm-up 50 request, lặp 3 lần; báo P50/P95/P99 và QPS.
- Ghi cấu hình máy trong run record. Nếu hai nhóm khác máy (ví dụ Nhóm 4 dùng GPU), bảng so sánh phải ghi rõ. **[cần chốt]**

## 8. Quy trình thay đổi

Sau khi đóng băng, mọi thay đổi phải báo nhóm kia, nâng phiên bản spec và cả hai nhóm chạy lại.
