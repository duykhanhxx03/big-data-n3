# Data dictionary — Nhóm 3 (sinh bởi scripts/make_data_dictionary.py)

Mọi bảng tạo bằng `python -m extract.leandojo` và các script trong `scripts/` từ LeanDojo Benchmark 4 v10 (xem `docs/data_card.md`). Định nghĩa truy vấn và premise đúng: `docs/benchmark_spec.md`. Bản mẫu nhỏ có thể đưa vào Git nằm trong `data_sample/` với cùng tên cột. Số liệu dưới đây tính lại mỗi lần chạy script.

Quan hệ: `qrels.query_id → queries.query_id`, `qrels.premise_id → premises.premise_id`, `premises.path → files.path`, `queries.file_path → files.path`, `dep_stats_*.premise_id → premises.premise_id`, `doc_map.premise_id → premises.premise_id`.

## 1. Bảng sau trích xuất (`data/processed/`)

### `data/processed/premises.parquet`

Một dòng = một khai báo trong corpus.jsonl (tập ứng viên truy hồi). **180,957 dòng.**

| Cột | Kiểu | Rỗng | Số giá trị khác nhau | Ví dụ / miền giá trị | Ý nghĩa |
|---|---|---|---|---|---|
| `premise_id` | int64 | 0 | 180,957 | 0 … 180,956 | Khóa chính. Thứ tự của (full_name, path) sau khi sắp xếp; ổn định với cùng snapshot. |
| `full_name` | string | 0 | 180,907 | `ADEInequality.A` | Tên đầy đủ của khai báo, gồm namespace (ví dụ `Nat.succ_le_of_lt`). Không duy nhất: 57 tên trùng giữa Lean core và Mathlib, nên định danh là cặp (full_name, path). |
| `name` | string | 0 | 180,907 | `ADEInequality.A` | Trường `name` của LeanDojo; trong snapshot này luôn trùng full_name. |
| `path` | string | 0 | 5,392 | `Mathlib/NumberTheory/ADEInequality.lean` | File chứa khai báo, tương đối so với gốc repo mathlib4 (thư viện phụ nằm dưới `.lake/packages/`). |
| `namespace` | string | 20,895 | 6,507 | `ADEInequality` | Phần trước thành phần cuối của full_name; rỗng nếu khai báo ở gốc. |
| `short_name` | string | 0 | 127,750 | `A` | Thành phần cuối của full_name. |
| `decl_kind` | string | 0 | 11 | `theorem` 109,823, `def` 36,826, `lemma` 16,969, `instance` 10,071, `abbrev` 2,487, `structure` 2,065, `class` 1,518, `inductive` 647, `opaque` 410, `irreducible_def` 125, `axiom` 16 | Loại khai báo: từ khóa đầu tiên của mã nguồn sau khi bỏ docstring/attribute. |
| `statement` | string | 0 | 178,957 | `def A (r : ℕ+) : Multiset ℕ+` | Phát biểu: mã nguồn cắt tại `:=` cấp ngoài cùng, `where` hoặc nhánh `|` đầu dòng. Field chính của index. |
| `code` | string | 0 | 179,765 | `def A (r : ℕ+) : Multiset ℕ+ := A' 1 r` | Mã nguồn đầy đủ của khai báo (gồm docstring, attribute, chứng minh). Không dùng để index. |
| `library` | string | 0 | 7 | `Mathlib` 156,947, `lean4` 19,619, `batteries` 2,883, `aesop` 1,122, `proofwidgets` 252, `Qq` 119, `importGraph` 15 | Thư viện gốc suy từ path: Mathlib, lean4 (core), batteries, aesop, ... |
| `start_line` | int64 | 0 | 3,249 | 4 … 4,606 | Dòng bắt đầu của khai báo trong file (tính cả docstring/attribute). |
| `end_line` | int64 | 0 | 3,260 | 6 … 4,612 | Dòng kết thúc. Dùng cho bộ lọc truy cập được: premise cùng file phải có end_line < dòng bắt đầu theorem. |

### `data/processed/queries.parquet`

Một dòng = một tactic có ít nhất một premise đúng trong corpus. **315,014 dòng.**

| Cột | Kiểu | Rỗng | Số giá trị khác nhau | Ví dụ / miền giá trị | Ý nghĩa |
|---|---|---|---|---|---|
| `query_id` | string | 0 | 315,014 | `random:train:0:1` | Khóa chính, dạng `{split_strategy}:{split}:{theorem_idx}:{tactic_idx}`. |
| `split_strategy` | string | 0 | 2 | `random` 157,507, `novel_premises` 157,507 | Cách chia của LeanDojo: `random` (chính) hoặc `novel_premises` (khó). |
| `split` | string | 0 | 3 | `train` 301,789, `test` 6,782, `val` 6,443 | train / val / test. |
| `theorem_full_name` | string | 0 | 54,423 | `Algebra.intTrace_eq_trace` | Theorem đang được chứng minh. Được phép dùng lúc truy vấn (ngữ cảnh). |
| `file_path` | string | 0 | 3,510 | `Mathlib/RingTheory/IntegralRestrict.lean` | File chứa theorem. Được phép dùng lúc truy vấn. |
| `theorem_idx` | int64 | 0 | 82,086 | 0 … 118,507 | Vị trí theorem trong file JSON của split (để truy ngược dữ liệu gốc). |
| `theorem_start_line` | int64 | 0 | 2,590 | 11 … 3,668 | Dòng bắt đầu của theorem. Được phép dùng lúc truy vấn (bộ lọc truy cập được). |
| `tactic_idx` | int64 | 0 | 126 | 0 … 126 | Vị trí tactic trong `traced_tactics` của theorem. |
| `state_before` | string | 0 | 156,258 | `case h A : Type u_1 K : Type u_2 L : Type u_3 B : Type u_4 inst✝³⁰ : C …` | Proof state trước tactic = **văn bản truy vấn**. Text của tactic không được lưu (chứa tên premise). |
| `n_gold` | int64 | 0 | 39 | 1 … 48 | Số premise đúng của truy vấn sau khi bỏ premise không có trong corpus. |

### `data/processed/qrels.parquet`

Một dòng = một cặp (truy vấn, premise đúng). Nhãn nhị phân. **674,628 dòng.**

| Cột | Kiểu | Rỗng | Số giá trị khác nhau | Ví dụ / miền giá trị | Ý nghĩa |
|---|---|---|---|---|---|
| `query_id` | string | 0 | 315,014 | `random:train:0:1` | Khóa ngoại tới queries. |
| `premise_id` | int64 | 0 | 61,547 | 1 … 180,947 | Khóa ngoại tới premises. |

### `data/processed/files.parquet`

Một dòng = một file Lean trong corpus. **5,674 dòng.**

| Cột | Kiểu | Rỗng | Số giá trị khác nhau | Ví dụ / miền giá trị | Ý nghĩa |
|---|---|---|---|---|---|
| `path` | string | 0 | 5,674 | `.lake/packages/lean4/src/lean/Init/Prelude.lean` | Đường dẫn file, cùng quy ước với premises.path. |
| `imports` | list<element: string> | 1 | — | `['.lake/packages/lean4/src/lean/Init/Prelude.lean']` | Danh sách file được import trực tiếp. Bao đóng bắc cầu = tập file truy cập được. |

### `data/processed/dead_letter.parquet`

Bản ghi bị loại khi trích xuất, kèm lý do. Không có bản ghi nào mất lặng lẽ. **136,296 dòng.**

| Cột | Kiểu | Rỗng | Số giá trị khác nhau | Ví dụ / miền giá trị | Ý nghĩa |
|---|---|---|---|---|---|
| `entity` | string | 0 | 3 | `gold` 115,736, `query` 20,544, `premise` 16 | Loại bản ghi: `premise` (trùng), `gold` (premise đúng không có trong corpus), `query` (truy vấn không còn premise đúng). |
| `key` | string | 0 | 127,388 | `Lean.Parser.$n@.lake/packages/lean4/src/lean/Init/Simproc.lean` | Khóa của bản ghi; với `gold` là `{query_id}:{full_name}`. |
| `reason` | string | 0 | 3 | `premise_not_in_corpus` 115,736, `no_gold_in_corpus` 20,544, `duplicate_full_name_and_path` 16 | Lý do loại. |

### `data/processed/dep_stats_random.parquet`

Thống kê phụ thuộc tính từ train nửa A của split random (`scripts/build_dependency_stats.py`); bản `_novel_premises` cùng cấu trúc, tính từ train của novel_premises. **180,957 dòng.**

| Cột | Kiểu | Rỗng | Số giá trị khác nhau | Ví dụ / miền giá trị | Ý nghĩa |
|---|---|---|---|---|---|
| `premise_id` | int32 | 0 | 180,957 | 0 … 180,956 | Khóa ngoại tới premises (đủ 180.957 dòng). |
| `popularity` | int32 | 0 | 152 | 0 … 1,253 | Số theorem train-A dùng premise ít nhất một lần. |
| `pagerank` | double | 0 | 15,204 | 4.84e-06 … 0.00117 | PageRank trên đồ thị theorem→premise của train-A (damping 0,85, tổng = 1). |

### Tập đánh giá và index con

| File | Nội dung |
|---|---|
| `data/processed/eval_sets/novel_premises_test_all.npy` | 4,141 `query_id` (mảng chuỗi numpy), ví dụ `novel_premises:test:0:0` |
| `data/processed/eval_sets/novel_premises_val_all.npy` | 3,907 `query_id` (mảng chuỗi numpy), ví dụ `novel_premises:val:0:0` |
| `data/processed/eval_sets/random_test_all.npy` | 2,641 `query_id` (mảng chuỗi numpy), ví dụ `random:test:1003:0` |
| `data/processed/eval_sets/random_val_all.npy` | 2,536 `query_id` (mảng chuỗi numpy), ví dụ `random:val:0:1` |
| `data/processed/eval_sets/scale_random_test_500.npy` | 500 `query_id` (mảng chuỗi numpy), ví dụ `random:test:1012:10` |
| `data/processed/subsets/premises_10k.npy` | 10,000 `premise_id` của index con (lồng nhau, chứa mọi premise đúng của `scale_random_test_500`) |
| `data/processed/subsets/premises_50k.npy` | 50,000 `premise_id` của index con (lồng nhau, chứa mọi premise đúng của `scale_random_test_500`) |
| `data/processed/subsets/premises_100k.npy` | 100,000 `premise_id` của index con (lồng nhau, chứa mọi premise đúng của `scale_random_test_500`) |
| `data/processed/subsets/subsets_info.json` | seed (42), kích thước tập đánh giá và, với mỗi index con, số premise đúng bắt buộc (870) và tỉ lệ của chúng |
| `data/processed/synthetic/premises_x5.parquet` | Corpus nhân 5 lần cho thí nghiệm build: 4 cột `premise_id` (đánh lại 0..5N−1), `name`, `namespace`, `statement`; nội dung lặp nguyên văn |
| `data/processed/extract_stats.json` | Số liệu trích xuất; ví dụ decl_kind: `{"theorem": 109823, "def": 36826, "lemma": 16969, "instance": 10071, "abbrev": 2487, "structure": 2065, "class": 1518, " …` |

## 2. Index (`indexes/<tokenizer>__<fields>__<codec>__<subset>/`)

Ví dụ dưới đây là index của V2–V4 (`lean__name+namespace+statement__vb__full`). Index tạo lại được bằng `index.build_seq` hoặc `index.build_spark` (kết quả trùng từng byte).

### `indexes/lean__name+namespace+statement__vb__full/dictionary.parquet`

Từ điển của index: một dòng = một cặp (field, term). **481,002 dòng.**

| Cột | Kiểu | Rỗng | Số giá trị khác nhau | Ví dụ / miền giá trị | Ý nghĩa |
|---|---|---|---|---|---|
| `field` | string | 0 | 3 | `name` 289,711, `statement` 180,782, `namespace` 10,509 | Field của tài liệu: name / namespace / statement. |
| `term` | string | 0 | 324,435 | `''` | Term sau tokenizer. |
| `df` | int64 | 0 | 1,302 | 1 … 109,960 | Số tài liệu chứa term trong field này. |
| `cf` | int64 | 0 | 1,460 | 1 … 110,028 | Tổng số lần xuất hiện (collection frequency). |
| `offset` | int64 | 0 | 481,002 | 0 … 11,922,800 | Vị trí byte bắt đầu của posting list trong postings.bin. |
| `nbytes` | int64 | 0 | 1,852 | 6 … 219,963 | Số byte của posting list sau nén. |

### `indexes/lean__name+namespace+statement__vb__full/terms_any.parquet`

df gộp mọi field, dùng tính idf của BM25F. **324,435 dòng.**

| Cột | Kiểu | Rỗng | Số giá trị khác nhau | Ví dụ / miền giá trị | Ý nghĩa |
|---|---|---|---|---|---|
| `term` | string | 0 | 324,435 | `!` | Term. |
| `df_any` | int64 | 0 | 1,000 | 1 … 109,980 | Số tài liệu chứa term ở ít nhất một field. |

### `indexes/lean__name+namespace+statement__vb__full/doc_lengths.parquet`

Độ dài từng field của từng tài liệu (chuẩn hóa độ dài BM25). **180,957 dòng.**

| Cột | Kiểu | Rỗng | Số giá trị khác nhau | Ví dụ / miền giá trị | Ý nghĩa |
|---|---|---|---|---|---|
| `doc_id` | int32 | 0 | 180,957 | 0 … 180,956 | Mã tài liệu nội bộ của index (0..n_docs−1). |
| `len_name` | int32 | 0 | 26 | 0 … 25 | Số token field name (mỗi field index có một cột `len_<field>`). |
| `len_namespace` | int32 | 0 | 17 | 0 … 17 | Số token field namespace. |
| `len_statement` | int32 | 0 | 177 | 1 … 488 | Số token field statement. |

### `indexes/lean__name+namespace+statement__vb__full/doc_map.parquet`

Ánh xạ doc_id của index → premise_id (index con chỉ chứa một phần corpus). **180,957 dòng.**

| Cột | Kiểu | Rỗng | Số giá trị khác nhau | Ví dụ / miền giá trị | Ý nghĩa |
|---|---|---|---|---|---|
| `doc_id` | int32 | 0 | 180,957 | 0 … 180,956 | Mã tài liệu nội bộ. |
| `premise_id` | int32 | 0 | 180,957 | 0 … 180,956 | Khóa ngoại tới premises. |

### `postings.bin` và `index_manifest.json`

- `postings.bin`: các posting list nối liền, mỗi list = dãy doc_id (lưu d-gap) và dãy tf, nén bằng codec ghi trong manifest (`raw32`, `vb` hoặc `gamma`; xem `src/index/codec.py`). Index này: 4,323,308 posting, 11,923,305 byte.
- `index_manifest.json`: `format_version`, `tokenizer`, `tokenizer_version`, `fields`, `codec`, `n_docs`, `avgdl`, `n_terms`, `n_term_field_keys`, `n_postings`, `max_df`, `postings_bytes`, `source`, `build`, `checksums`. `checksums` là sha1 của từng file; `build` ghi builder và thời gian.

## 3. Kết quả (`results/`)

| File | Cột / khóa | Ý nghĩa |
|---|---|---|
| `results/runs/<run_id>.tsv` | `query_id`, `premise_id`, `rank`, `score`, `system` | Top-100 mỗi truy vấn; định dạng chung với Nhóm 4 để tính metric và trộn RRF |
| `results/raw/eval/<run_id>.perquery.parquet` | `query_id`, `R@1`, `R@5`, `R@10`, `R@50`, `MRR`, `nDCG@10`, `latency_ms` | Metric từng truy vấn, đầu vào của paired bootstrap |
| `results/raw/**/<run_id>.json` | `run_id`, `kind`, `timestamp`, `git_commit`, `code_hash`, `data_version`, `machine`, `software`, `parameters`, `seed`, `timeout_s`, `metrics`, `status`, `log` | Run record theo Phụ lục B của hướng dẫn |
| `results/models/*.json` | `type`, `features`, `mean`, `std`, `coef`, `intercept`, `C`, `train` | Trọng số reranker hồi quy logistic |
