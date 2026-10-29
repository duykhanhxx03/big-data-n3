# AI-use statement

Theo yêu cầu của học phần INT4418, nhóm khai báo việc dùng công cụ AI trong dự án.

| Công cụ | Dùng cho | Phần việc |
|---|---|---|
| Claude (Anthropic) qua Claude Code | Lập kế hoạch 10 tuần, viết khung mã nguồn, test, script thí nghiệm, bản nháp tài liệu | Toàn bộ repo phiên bản đầu (2026-10-17) |
| Claude (Anthropic) qua Claude Code | Bổ sung sản phẩm tuần 1, 2, 5 (truy vấn kiểm tra, data dictionary, ablation plan, latency profile), hướng dẫn vai trò; soạn báo cáo LaTeX `report/latex/main.tex`; viết engine Spark Arrow (`src/index/build_spark_arrow.py`), môi trường Docker Linux và thí nghiệm build2; V5 (ứng viên bổ sung), baseline Lucene, thí nghiệm số worker và kiểm tra CPU | 2026-10-18 |
| Claude (Anthropic) qua VCode agent (khai báo của TV6) | Khung pipeline đánh giá mức theorem, data contract v1, mock end-to-end, corpus từ Zenodo, 4 hệ BM25/hybrid, load test, ablation, bản nháp báo cáo | 2026-10-01 và 2026-10-15; pipeline này đã được gỡ khi gộp repo, còn trong lịch sử Git đến b1fbb35 |
| Claude (Anthropic) qua Claude Code | Gộp hai repo của nhóm; gỡ pipeline trùng lặp; dựng lại gói benchmark chung N3/N4 từ dữ liệu gốc; `scripts/eval_shared.py` và kết quả trên gói chung; sửa kiểm tra SHA-256 đầu vào báo cáo | 2026-10-20 |

## Những gì nhóm phải tự kiểm chứng trước khi nộp
- Mỗi thành viên đọc và giải thích được phần mã thuộc vai trò của mình (học phần yêu cầu: không nộp đoạn mã mình chưa hiểu).
- Chạy lại toàn bộ pipeline theo README trên máy của nhóm và đối chiếu số liệu với `results/`.
- Kiểm tra lại các con số trong báo cáo bằng script, không chép tay từ đầu ra của AI.
- Đọc lại 30 failure case trong `results/figures/failure_cases.md`: phần phân loại tự động chỉ là gợi ý, nhóm phải gán nhãn lại thủ công.
- Kiểm tra trích dẫn tài liệu tham khảo trong báo cáo bằng cách mở nguồn gốc.

## Những gì AI không quyết định thay nhóm
- Phân vai, người dự phòng, nội dung thống nhất với Nhóm 4.
- Kết luận cuối cùng của báo cáo và câu trả lời khi bảo vệ.

Cập nhật bảng trên mỗi khi dùng thêm AI cho việc khác (ví dụ viết báo cáo, làm slide).
