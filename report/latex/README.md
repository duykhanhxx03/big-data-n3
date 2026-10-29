# Báo cáo LaTeX

`main.tex` là báo cáo chính thức của Nhóm 3; bản PDF đã biên dịch được chép ra `report/report.pdf`.
Báo cáo dùng lớp `report` của LaTeX: phần đầu đánh số La Mã (tóm tắt, mục lục, danh sách hình vẽ và danh sách bảng),
thân bài gồm 8 chương (20 trang), sau đó là tài liệu tham khảo và ba phụ lục A (khai báo AI, phân công), B (tái lập) và
C (bảng bổ sung). Hình, bảng và công thức đánh số theo chương.

## Nguồn số liệu

- Bảng số liệu trong `tables/*.tex` sinh bằng `python scripts/make_report_tables.py` từ kết quả thô; không sửa tay.
  `tables/SOURCES.md` ghi nguồn của từng bảng, các `run_id` đã ghim và SHA-256 của mọi tệp đầu vào.
- Hình trong `figures/` sinh bằng `python scripts/make_figures.py --report`; `figures/SOURCES.md` ghi SHA-256 của các run
  record dùng cho hình. PDF của hình không ghi ngày tạo, nên cùng dữ liệu luôn cho cùng tệp.
- `tables/example.tex` (ví dụ xuyên suốt) sinh bằng `python scripts/running_example.py`.
- Các lần chạy dùng cho bản báo cáo được ghim trong `results/figures/report_manifest.json` (lần chạy đánh giá) và
  `results/figures/report_records.json` (run record của tải, xây chỉ mục, Lucene, chất lượng theo quy mô và ablation), đều
  do `python scripts/pin_runs.py` ghi. Chỉ chạy lại script này khi chủ ý cập nhật báo cáo.
- Phân tích độ bền (`scripts/robustness.py`), kiểm định ablation (`scripts/ablation_significance.py`) và phân rã candidate
  (`scripts/candidate_analysis.py`) phải chạy trước khi sinh bảng nếu có lần chạy mới.

## Biên dịch

Cần XeLaTeX (hoặc Tectonic, dựa trên XeTeX) vì báo cáo dùng `fontspec` với font TeX Gyre Termes và DejaVu Sans Mono.

```bash
bash build.sh                  # kiểm tra đầu vào khớp bản đã phát hành, sinh lại bảng, ví dụ, hình rồi biên dịch
bash build.sh --compile-only   # chỉ biên dịch từ bảng và hình đã có trong kho
bash build.sh --update         # chủ ý cập nhật báo cáo bằng đầu vào hiện tại (bỏ bước kiểm tra, ghi lại SOURCES.md)
```

Ở chế độ mặc định, `scripts/verify_report_inputs.py` so SHA-256 của mọi tệp đầu vào với giá trị trong `tables/SOURCES.md`
và `figures/SOURCES.md`; nếu có tệp thiếu hoặc lệch, build dừng thay vì âm thầm tạo bảng khác bản đã phát hành. Biến
`PYTHON` chọn trình thông dịch (mặc định `.venv` của kho mã), biến `TECTONIC` chọn đường dẫn Tectonic (mặc định tìm
trong PATH rồi `tools/tectonic/` của kho). SHA-256 của tệp văn bản được so ở dạng xuống dòng LF hoặc CRLF, vì Git đổi
xuống dòng khi checkout.

Biên dịch thủ công: `tectonic -X compile main.tex` hoặc `xelatex main.tex` hai lần. Trên Overleaf: tải cả thư mục này lên
(gồm `tables/` và `figures/`) và chọn trình biên dịch XeLaTeX.

## Kiểm tra từ bản checkout sạch

Bản checkout không có `data/` và `results/runs/` (không đưa vào Git), nhưng có đủ kết quả đã ghim để dựng lại đúng báo cáo:
`PYTHON=<python của môi trường> bash build.sh`. Khi đó `running_example.py` giữ nguyên `tables/example.tex`, còn bảng và
hình được sinh lại. Lần kiểm tra ngày 19/10/2026 trên bản sao chỉ gồm các tệp sẽ đưa vào Git: 54 test qua, 4 test bỏ qua vì
cần `data/processed`, 795 tệp đầu vào khớp SHA-256, bảng và hình sinh lại trùng từng byte với bản trong kho; khi sửa một
tệp đầu vào, build dừng và báo tệp lệch.

## Các chỗ nhóm phải điền trước khi nộp

Báo cáo không còn chữ đỏ (macro `\canchot`), nhưng các việc sau vẫn ghi là "chưa được thực hiện" trong bài:

- Đối chiếu thủ công và ký xác nhận 20 truy vấn kiểm tra (`docs/check_queries.md`). Mục 3.4 "Kiểm tra chất lượng dữ
  liệu" đang được comment trong `main.tex`; bỏ dấu `%` để hiện lại khi làm xong.
- Bảng so sánh với Nhóm 4 (đoạn "Bối cảnh và so sánh với nhóm khác" cuối Mục 6.1).
- Gán nhãn thủ công 30 trường hợp thất bại bởi hai thành viên và tỉ lệ đồng thuận (Mục 7.3).

Khi làm xong, sửa câu mô tả trạng thái tương ứng trong báo cáo. Nhóm cũng cần mở từng tài liệu tham khảo để kiểm tra
trích dẫn.

Trang bìa theo mẫu của Học viện: khung viền và logo nằm trong `assets/`, đặt tuyệt đối bằng TikZ nên cần biên dịch hai
lượt (Tectonic tự chạy lại).

## Quy ước thuật ngữ

Theo bản đánh giá lần 3, báo cáo giữ các thuật ngữ quen thuộc của lĩnh vực và giải thích ở lần đầu: premise retrieval,
sparse/dense/hybrid retrieval, proof state, tokenizer, reranker và rerank, candidate và candidate pool, candidate recall,
candidate loss, ranking gap, gold premise, seen/unseen premise, accessible premise, train/validation/test/holdout,
query-macro và theorem-macro, P95 latency. Tên trong mã (tên đặc trưng, tên split, query ID) giữ nguyên.
