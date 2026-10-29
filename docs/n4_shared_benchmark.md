# Benchmark chung N3/N4 (gói `n4_shared`)

Phụ trách: Data/extraction (Nguyễn Đức Hoàng). Đây là giao diện dữ liệu đã đóng băng để Nhóm 3 và Nhóm 4 so sánh trên
cùng một split. Nguồn chuẩn là gói `n4_shared.zip` do Data/extraction ghim cho Nhóm 4; mọi so sánh N3/N4 dùng đúng các
byte này, không dựng một split khác. Benchmark mức tactic dùng trong báo cáo của Nhóm 3 được mô tả riêng ở
`docs/benchmark_spec.md`; khác biệt giữa hai benchmark ở mục 8.

## 1. Nguồn và phiên bản

| Mục | Giá trị |
|---|---|
| Dữ liệu gốc | LeanDojo Benchmark 4 v10, Zenodo 12740403, CC BY 2.0 |
| mathlib4 commit | `29dcec074de168ac2bf835a77ef68bbe069194c5` |
| `data_version` | `leandojo-benchmark-4@29dcec07+split-v1.0` |
| `pool_id` | `shared-frozen-pool` |
| `corpus.jsonl` | 180.907 premise, SHA-256 `c9157bd1381ca4147aa931f14b9985ac9bfe5cbfe892a982a24a133ff4221efd` |
| `queries.jsonl` | 898 truy vấn, SHA-256 `c2b764fafe981755ae7ca2614912498f0559317ea5427bcb7a7150cb45166004` |
| Giá trị ghim | `configs/n4_shared_manifest.json` và `src/shared.py` |

## 2. Schema

`corpus.jsonl`, mỗi dòng một premise:

| Trường | Nội dung |
|---|---|
| `id` | `full_name` của khai báo |
| `name` | bằng `id` (LeanDojo không tách tên ngắn khỏi namespace) |
| `statement` | trường `code` của khai báo trong `corpus.jsonl` gốc, giữ nguyên văn |
| `namespace` | phần trước dấu chấm cuối của `full_name`, rỗng nếu khai báo ở cấp gốc |
| `source_commit` | mathlib4 commit ở trên |

`queries.jsonl`, mỗi dòng một theorem của `random/test`:

| Trường | Nội dung |
|---|---|
| `id` | `Q` + `full_name` của theorem |
| `statement` | `code` của theorem trong corpus |
| `relevant_ids` | premise được chú thích trong mọi tactic của chứng minh và có trong corpus, sắp tăng dần; 4.165 nhãn, trung bình 4,72 trên mỗi truy vấn có nhãn |
| `split` | `test` |
| `pool_id` | `shared-frozen-pool` |

Cách dựng (`shared.build_records`): corpus lấy mọi premise của `corpus.jsonl`, bỏ tên trùng và giữ lần xuất hiện đầu,
sắp theo `id`. Vì vậy 50 bản ghi của 45 tên có ở cả Lean core lẫn Mathlib bị gộp: corpus chung có 180.907 premise, còn
corpus mức tactic có 180.957 premise định danh theo cặp `(full_name, path)`. Truy vấn là mọi theorem của `random/test` có
ít nhất một premise được chú thích. Tệp dùng xuống dòng CRLF vì bản gốc được ghi trên Windows; `write_bundle` ghi cố định
CRLF nên SHA-256 giống nhau trên mọi hệ điều hành.

## 3. Quy tắc eligible và cách chấm

- Chính theorem truy vấn không được xuất hiện trong kết quả của nó: loại trước khi cắt top-k (`shared.eligible`).
- Theorem test khác vẫn là premise hợp lệ. 56 nhãn trong `relevant_ids` là theorem của `random/test`, 17 trong số đó cũng
  là truy vấn của gói.
- 16 truy vấn có `relevant_ids` rỗng vì mọi premise được chú thích của chúng không có trong corpus. Không sửa tệp và không
  âm thầm bỏ: chúng được đếm và báo riêng, còn trung bình metric tính trên 882 truy vấn có nhãn (Recall không xác định khi
  không có nhãn).
- Run file nhận vào: CSV hoặc TSV có cột `query_id`, `premise_id` (là `full_name`); có cột `rank` thì xếp theo `rank`,
  không thì theo thứ tự dòng. Nhóm 3 ghi TSV `query_id, premise_id, rank, score, system`, top-100 mỗi truy vấn. Mọi run
  file, kể cả của Nhóm 4, được chấm cùng một quy tắc: bỏ tên lặp, loại chính theorem, rồi cắt top-k.

## 4. Lưu ý: nhãn lộ qua phần chứng minh

Trường `code` của LeanDojo giữ cả chứng minh của một số theorem, nên 158 trên 898 truy vấn có `statement` còn chứa phần
sau `:=` (`shared.has_proof_body`). Trong 155 truy vấn như vậy có nhãn, 145 truy vấn có chứng minh gọi thẳng tên một
premise đúng. Hệ thống nào khớp tên premise với văn bản truy vấn (BM25, cả dense) đều được lợi trên các truy vấn này.

Các byte đã đóng băng không được sửa. Mọi metric vì vậy được báo trên hai lát: `all` (882 truy vấn) và `signature` (727
truy vấn không chứa chứng minh). Đề nghị cả hai nhóm báo lát `signature` bên cạnh kết quả chính. Nếu hai nhóm thống nhất
cắt statement tại `:=` thì đó là phiên bản gói mới: nâng `data_version`, ghim lại SHA-256 và cả hai nhóm chạy lại.

## 5. Chạy

```bash
python scripts/export_n4_shared.py --zip       # dựng lại gói từ data/raw → data/n4_shared/ (+ .zip); dừng nếu SHA-256 lệch
python scripts/verify_n4_shared.py data/n4_shared.zip   # kiểm tra một gói nhận được (zip hoặc thư mục)
python scripts/eval_shared.py                  # B0, V2, V4, V5 trên benchmark chung → results/figures/shared_benchmark.md
python scripts/eval_shared.py --external N4-dense=<run_dense>.csv   # chấm run file của Nhóm 4 và trộn RRF với V4
python scripts/eval_shared.py --external N4-dense=<run_dense>.csv --fuse V4+N4-dense V5+N4-dense
python -m pytest -q tests/test_shared_data.py
```

`verify_n4_shared.py` kiểm tra SHA-256 toàn tệp, schema, kiểu dữ liệu, tính duy nhất của `id`, khóa ngoại của
`relevant_ids` và quy tắc self, rồi ghi `results/data_quality/shared_validation.json`. Theo quy định của học phần, gói
(66 MB JSONL) không đưa vào Git; Git chỉ giữ manifest, mã dựng và mã kiểm tra.

## 6. Kết quả hiện có của Nhóm 3

Run `shared-20261020-112058-e591` (`results/figures/shared_benchmark.md`), R@10:

| Hệ thống | `all` (882) | `signature` (727) | P95 tuần tự (ms) |
|---|---|---|---|
| B0 | 0,215 | 0,190 | 19,2 |
| V2 | 0,313 | 0,240 | 33,5 |
| V4 | 0,418 | 0,347 | 40,9 |
| V5 | 0,432 | 0,369 | 35,8 |

V4 và V5 lọc premise truy cập được (dùng file và dòng bắt đầu của theorem); B0 và V2 chỉ dùng statement. Reranker và
thống kê phụ thuộc lấy từ train của `random`, không chứa theorem test nào.

## 7. Bàn giao Nhóm 4

Nhóm 4 nhận nguyên gói (hoặc tự dựng lại bằng `export_n4_shared.py` và đối chiếu SHA-256), rồi trả run file theo mục 3.
Nhóm 3 chỉ chấm run file đó bằng `eval_shared.py`, không sửa kết quả của nhóm khác. Trộn RRF dùng điểm 1/(60 + hạng), trọng số bằng
nhau, giống `scripts/fuse_rrf.py`; gói chỉ có tập test nên không tinh chỉnh hằng số hay trọng số.

## 8. Khác biệt với benchmark mức tactic của báo cáo

| | Benchmark chung (gói này) | Benchmark mức tactic (`docs/benchmark_spec.md`) |
|---|---|---|
| Đơn vị truy vấn | theorem | tactic (`state_before`) |
| Văn bản truy vấn | `code` của theorem | proof state trước tactic |
| Định danh premise | `full_name` (180.907) | `(full_name, path)` (180.957) |
| Tập đánh giá | `random/test`, 882 truy vấn có nhãn | `random/test` 2.641 và `novel_premises/test` 4.141 truy vấn |
| Dùng cho | so sánh với Nhóm 4 | mọi bảng chất lượng trong báo cáo |

Số liệu của hai benchmark không so trực tiếp với nhau.
