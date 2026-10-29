# Data card — Lean premise retrieval (Nhóm 3) v0.1

## Nguồn và phiên bản
- LeanDojo Benchmark 4 **v10**, Zenodo record 12740403 (DOI qua trang Zenodo), md5 lưu trong `data/raw/MANIFEST.json`.
- Trích từ mathlib4 commit `29dcec074de168ac2bf835a77ef68bbe069194c5`, LeanDojo 2.0.0.
- Giấy phép: CC BY 2.0 cho dataset; mã nguồn Lean gốc: Apache-2.0 (xem `docs/license_note.md`).

## Bảng sau xử lý (`data/processed/`, tạo bằng `python -m extract.leandojo`)

| Bảng | Số dòng | Cột chính |
|---|---|---|
| premises | 180.957 | premise_id, full_name, path, namespace, short_name, decl_kind, statement, code, library, start_line, end_line |
| queries | 315.014 | query_id, split_strategy, split, theorem_full_name, file_path, theorem_start_line, tactic_idx, state_before, n_gold |
| qrels | 674.628 | query_id, premise_id |
| files | 5.674 | path, imports |
| dead_letter | 136.296 | entity, key, reason |

`statement` = phần phát biểu/chữ ký đã bỏ attribute `@[...]`, docstring và thân định nghĩa (cắt ở `:=`, `where`
hoặc nhánh `|` ở độ sâu ngoặc 0). Với theorem, `code` gốc của LeanDojo vốn không chứa chứng minh.

## Thành phần corpus
- Theo thư viện: Mathlib 156.947; Lean 4 core 19.619; Batteries 2.883; Aesop 1.122; ProofWidgets 252; Qq 119; ImportGraph 15.
- Theo loại khai báo: theorem 109.823; def 36.826; lemma 16.969; instance 10.071; abbrev 2.487; structure 2.065;
  class 1.518; inductive 647; opaque 410; irreducible_def 125; axiom 16.

## Kiểm tra chất lượng và bản ghi bị loại (`dead_letter.parquet`)
- 16 premise trùng hoàn toàn (tên + file): giữ bản đầu.
- 115.736 lượt premise đúng không có trong corpus (lemma `@[to_additive]`, constructor, trường structure/class):
  bỏ khỏi qrels. 20.544 truy vấn không còn premise đúng nào: bỏ.
- Không có trường hợp tên tồn tại trong corpus nhưng lệch file (đã kiểm tra trên val/test).
- 100% premise đúng của val nằm trong tập premise truy cập được qua import (kiểm tra cho bộ lọc V2a/V4).

## Hạn chế
- Recall chỉ tính trên premise có trong corpus; premise sinh tự động không bao giờ tìm được bằng truy hồi.
- Corpus là snapshot mathlib giữa năm 2024; kết luận có thể khác với mathlib hiện tại.
- Proof state chứa tên giả thiết do người viết đặt (`h`, `hx`), là nhiễu với truy hồi từ khóa.
- `novel_premises` có phân phối khác `random`: premise của test chưa từng được dùng trong train.
- Corpus tổng hợp ×5 (`data/processed/synthetic/`) chỉ để đo build/skew, không phản ánh dữ liệu thật.

## Tái tạo
```
python scripts/download_data.py
python -m extract.leandojo
python scripts/make_subsets.py
python scripts/build_dependency_stats.py
```
