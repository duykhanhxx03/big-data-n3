# Kịch bản demo (tuần 10, khoảng 10 phút)

Chuẩn bị trước: đã chạy `bash scripts/reproduce.sh`; mở sẵn `results/figures/`; khởi động API:
`uvicorn serving.api:app --port 8000 --workers 4` (đợi `/health` trả `ok`, khoảng 30 giây).

| Phút | Người trình bày | Nội dung | Lệnh / tệp |
|---|---|---|---|
| 0–1 | Eval (Minh) | Bài toán, RQ1/RQ2, primary metrics | slide 1–2 |
| 1–2 | Data (Hoàng) | Dữ liệu v10, 181K premise, vì sao 115K premise đúng bị loại | `docs/data_card.md` |
| 2–4 | Indexing (Khánh) | Build tuần tự và Spark trùng từng byte; skew và overhead Spark | `python -m index.build_seq --out indexes/demo_seq --subset data/processed/subsets/premises_10k.npy` rồi so `index_manifest.json`; `fig_build_time.png` |
| 4–6 | BM25 (Cường) + Structural (Dũng) | Cùng một truy vấn: B0 trượt, V4 trúng; giải thích đặc trưng | `python -m serving.cli --query-id "random:val:0:1" --mode B0 -k 5` rồi `--mode V4` |
| 6–7 | Serving (Kiên) | Gọi API trực tiếp, xem `server_ms`; load test ngắn | `curl -s -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query":"n m : ℕ\n⊢ n + m = m + n","k":5,"mode":"V4"}'`; `python -m eval.loadtest --mode V4 --clients 1,10 --requests 200` |
| 7–9 | Eval (Minh) | Bảng chất lượng + bootstrap, đánh đổi R@10–P95, error analysis | `results/figures/quality.md`, `fig_tradeoff.png`, `error_slices.md` |
| 9–10 | Cả nhóm | Hạn chế, việc tiếp theo (hybrid với Nhóm 4) | slide cuối |

Phương án dự phòng nếu máy demo chậm: chạy CLI trên index 10K (`N3_SUBSET=10k` cho API), hoặc mở sẵn đầu ra đã lưu.
