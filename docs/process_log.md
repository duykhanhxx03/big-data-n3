# Nhật ký quy trình: tập nào được dùng khi nào

Mục đích: ghi lại thứ tự thời gian giữa việc xem tập kiểm tra, chốt cấu hình và tạo, dùng tập holdout, để người đọc kiểm
tra được phạm vi của từng kết quả (báo cáo, Mục 5.3). Mọi mốc dưới đây lấy từ trường `timestamp` của run record trong
`results/raw/eval/*.json` hoặc từ thời điểm ghi tệp; giờ địa phương UTC+7. Bản thân các tệp này không chứng minh được một
tập "chưa từng được xem"; chúng chỉ cho thấy thứ tự các lần chạy đã được ghi lại.

| Thời điểm | Sự kiện | Bằng chứng |
|---|---|---|
| 17/10 18:49 | Chốt cấu hình V4 trên validation | `configs/v4_rerank_dep.yaml` |
| 17/10 19:00 | Đánh giá V4 trên tập kiểm tra `random` lần đầu | `eval-V4-20261017-185844-8434` (tag `quality-parallel`) |
| 17/10 19:48 | Sửa lỗi bộ lọc truy cập được (phát hiện khi xem lỗi trên tập kiểm tra), huấn luyện lại V4 | `results/models/v4.json` |
| 17/10 19:50 | Đánh giá lại V4 sau khi sửa lỗi | `eval-V4-20261017-194917-88cd` (tag `quality-parallel-fixed`) |
| 17/10 20:22 | Phân tích lỗi của V4 trên tập kiểm tra `random`; đây là nguồn ý tưởng của V5 | `results/raw/error_analysis.parquet` |
| 18/10 22:03 | Chọn số ứng viên của V5 trên validation (8 tổ hợp), chốt cấu hình | `results/raw/tuning/V5__extra__random_val_all.csv`, `configs/v5_rerank_expand.yaml` |
| 18/10 22:05 | Huấn luyện mô hình V5 cho hai split | `results/models/v5.json`, `v5_novel_premises.json` |
| 18/10 22:07 | Đánh giá V5 trên validation và hai tập kiểm tra (các lần chạy đã ghim) | `eval-V5-20261018-220552-dadd`, `-220652-4f11`, `-220757-26a7` |
| 18/10 22:41 | Nhận bản đánh giá phản biện lần 1 | tệp nhận xét không còn trong kho |
| 18/10 22:45 | Tạo tập holdout từ train B, sau khi V5 đã chốt | `scripts/make_holdout.py`, `data/processed/eval_sets/random_holdoutB.json` |
| 18/10 22:44 đến 22:53 | Thêm đối chứng V4-K240, huấn luyện và đánh giá trên validation, hai tập kiểm tra và holdout | `configs/review/v4k240.yaml`, `eval-V4-K240-*` |
| 18/10 23:01 đến 23:04 | Đánh giá B0, V2, V4, V5 trên holdout, mỗi hệ thống đúng một lần | `eval-B0-20261018-230121-0d0f`, `eval-V2-20261018-230154-aeae`, `eval-V4-20261018-230243-b493`, `eval-V5-20261018-230340-4bf3` |
| 19/10 | Nhận bản đánh giá lần 2; thay giá trị p bootstrap bằng kiểm định hoán vị theo cụm, tính lại từ metric từng truy vấn đã lưu, không chạy lại truy hồi | `src/eval/metrics.py`, `scripts/robustness.py`, `scripts/ablation_significance.py` |
| 19/10 | Nhận bản đánh giá lần 3; ghim cả nguồn của hình, thêm bước kiểm tra SHA-256 trước khi dựng báo cáo, biên tập layout và thuật ngữ. Không có lần chạy thí nghiệm mới | `scripts/make_figures.py`, `scripts/verify_report_inputs.py`, `report/latex/build.sh` |

Hệ quả cho báo cáo:

- Kết quả của B0 đến V4 trên tập kiểm tra là kết quả của cấu hình chốt trên validation; riêng lần sửa lỗi bộ lọc là sửa
  tính đúng sau khi xem tập kiểm tra.
- Kết quả của V5 trên tập kiểm tra là kết quả sau quan sát, vì ý tưởng V5 đến từ phân tích lỗi trên tập kiểm tra.
- Holdout được tạo sau khi V5 chốt, và mỗi hệ thống được đánh giá trên holdout một lần. Không có cấu hình nào được chọn
  theo kết quả holdout. Từ nay holdout đã được dùng, nên không còn là tập xác nhận chưa xem cho cải tiến tiếp theo.
