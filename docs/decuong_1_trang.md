# ĐỀ CƯƠNG 1 TRANG — NỘP TUẦN 1 (theo mẫu Phụ lục A)

**Tên nhóm và đề tài:** Nhóm 3 — Lean Premise Retrieval: Truy hồi thưa và truy hồi lai (Sparse/Hybrid) trên mathlib4

**Câu hỏi nghiên cứu:**
- RQ1: BM25 có đủ mạnh cho dữ liệu Lean hay cần thêm trường ký hiệu, namespace và dependency?
- RQ2: Kết hợp sparse score với tín hiệu cấu trúc có cải thiện Recall@k mà vẫn giữ độ trễ thấp không?

**Nút thắt Big Data:** Kho định lý mathlib4 quy mô hàng trăm nghìn declarations làm truy vấn tuyến tính tốn kém; hệ thống phải xây chỉ mục, phục vụ nhiều truy vấn đồng thời, và đo chất lượng–độ trễ–kích thước chỉ mục khi kho tăng (10K → 100K → toàn bộ 181K).

**Dữ liệu và giấy phép:** LeanDojo Benchmark 4 (theorem, proof, tactic, premise từ mathlib4) + mathlib4 source/declarations. Kiểm tra license tại thời điểm tải; lưu sample + manifest + checksum; không đưa dữ liệu lớn vào Git.

**Baseline:** TF-IDF/BM25 mặc định trên statement đã tách từ — không embedding, không dependency graph. Cùng dữ liệu/split/phần cứng/timeout với phương pháp chính.

**Ba mức quy mô/tải:** Scale: 10K → 100K → toàn bộ (181K) declarations. Tải: 1, 10, 50 truy vấn đồng thời.

**Primary metrics:** Recall@10 (chất lượng) + P95 latency (hiệu năng). Các chỉ số phụ: Recall@1/5/50, MRR, nDCG@10, QPS, index build time, index size.

**Kiến trúc bản nháp:** Extractor → document fields → Tokenizer (giữ từ + ký hiệu Lean) → Inverted index → BM25 baseline → Structural features / field weights → Hybrid reranker → REST API hoặc CLI.

**Phân vai (6 người, theo `contributions.md`):**
| Vai | Người | Dự phòng |
|-----|-------|----------|
| Data/Extraction | Nguyễn Đức Hoàng | Nguyễn Ngọc Minh |
| Indexing | Trần Duy Khánh | người phụ trách Serving |
| BM25 Baseline | Nguyễn Mạnh Cường | người phụ trách Structural |
| Structural Features | chưa chốt (Vũ Trung Kiên hoặc Trần Đình Dũng) | Nguyễn Mạnh Cường |
| Serving/Performance | chưa chốt (Vũ Trung Kiên hoặc Trần Đình Dũng) | Trần Duy Khánh |
| Evaluation/Error Analysis | Nguyễn Ngọc Minh | Nguyễn Đức Hoàng |

**Rủi ro và phương án dự phòng:**
1. Data contract với Nhóm 4 trễ → họp chung tuần 1, contract 1 trang, hai nhóm đồng ký; có snapshot/mock riêng để không bị chặn.
2. LeanDojo/mathlib4 tải chậm hoặc đổi format → tải sớm, ghim commit cố định, script tải tái lập được.
3. Chỉ báo cáo quality mà thiếu latency → TV5 đo latency ngay từ tuần 3; mọi bảng kết quả bắt buộc có cột P50/P95.

---
*Người nộp: ______________ | Ngày: ____/____/2026 | Trưởng nhóm ký: ______________*
