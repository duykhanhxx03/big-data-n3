# Nhóm 3 — Lean Premise Retrieval: truy hồi thưa và truy hồi lai

Bài tập lớn INT4418 Dữ liệu lớn (PTIT, AI4Math). Hệ thống tìm các định lý/bổ đề (premise) trong mathlib4 phù hợp
với một trạng thái chứng minh Lean, bằng chỉ mục đảo BM25/BM25F và tín hiệu cấu trúc, trước khi gọi bộ chứng minh.

Repo gộp hai phần việc của nhóm: pipeline truy hồi `src/` (trước ở repo `big-data-n3`) và gói benchmark chung N3/N4 của
Data/extraction. Lịch sử Git của cả hai được giữ. Pipeline mức theorem cũ trùng chức năng với `src/` đã được gỡ, vẫn xem
được trong lịch sử đến commit `b1fbb35`.

**Câu hỏi nghiên cứu**
- RQ1. BM25 có đủ mạnh cho dữ liệu Lean, hay cần thêm trường ký hiệu, namespace và dependency?
- RQ2. Kết hợp sparse score với tín hiệu cấu trúc có tăng Recall@k mà vẫn giữ độ trễ thấp không?

**Primary metrics:** Recall@10 và P95 latency ở 10 client đồng thời.

## Kết quả chính

Benchmark mức tactic của báo cáo (`docs/benchmark_spec.md`), toàn bộ tập kiểm tra, 180.957 premise ứng viên, trung bình
theo truy vấn. Số liệu lấy từ các lần chạy đã ghim trong
`results/figures/report_manifest.json`. Khoảng tin cậy từ bootstrap theo cụm theorem; giá trị p hai phía từ kiểm định
hoán vị ghép cặp theo cụm theorem, hiệu chỉnh Holm (`results/figures/robustness.md`).

| Biến thể | random R@10 | random MRR | novel_premises R@10 | Holdout R@10 |
|---|---|---|---|---|
| B0 (baseline bắt buộc) | 0,100 | 0,065 | 0,127 | 0,113 |
| V2 (tokenizer Lean + BM25F) | 0,147 | 0,093 | 0,183 | 0,190 |
| V4 (+ lọc import + reranker cấu trúc) | 0,243 | 0,196 | 0,269 | 0,310 |
| V4-K240 (V4 lấy top 240 của BM25F, đối chứng của V5) | 0,254 | 0,202 | 0,276 | 0,321 |
| V5 (+ ứng viên từ lịch sử sử dụng) | 0,281 | 0,217 | 0,280 | 0,332 |

- Trên `random`, 6 trên 7 bước của chuỗi cải tiến có ý nghĩa sau hiệu chỉnh Holm; bước bộ tách từ Lean (+0,010) có p = 0,053.
  Riêng trường namespace của BM25F chưa có tác dụng xác định (đối chứng V2-nons); độ phổ biến và PageRank chỉ kết luận được
  theo cặp.
- V5 được thiết kế sau khi xem lỗi trên tập kiểm tra, nên được kiểm tra lại trên holdout 2.500 truy vấn (phần train nửa B
  giữ riêng, xem `docs/process_log.md`): V5 hơn V4 0,022 (KTC 95% [0,009; 0,035]), nhưng phần V5 vượt đối chứng V4-K240
  (0,011, KTC chứa 0) chưa được xác lập. V4-K240 chỉ xấp xỉ số ứng viên của V5 và không có hai đặc trưng sử dụng, nên chênh
  lệch này là tác dụng gộp. Trên premise chưa thấy của `novel_premises`, V5 giảm Recall@10 từ 0,444 xuống 0,359. V4 là cấu
  hình nên dùng chung; V5 là đánh đổi có lợi cho premise quen thuộc.
- Độ trễ (API 4 worker, index đầy đủ, trung vị 3 lần): P95 ở 10 client của B0, V2, V4 là 60,2; 93,1; 95,5 ms trong đợt đo
  chính. Trong một phiên đo riêng, V4 và V5 có P95 103,2 và 116,7 ms ở 10 client; ở 50 client QPS giảm từ 132 xuống 116.
- Index Spark trùng từng byte với bản tuần tự. Engine Spark tối ưu (lô Arrow, gộp posting theo partition) nhanh hơn bản
  tuần tự 1,53 lần trên corpus thật 181K (Linux, chỉ mục 3 trường) và 3,84 lần trên corpus nhân bản 3,6 triệu bản ghi; mọi
  phép đo trên một máy, Spark chế độ local.
- Với V4, phần recall mất do tạo ứng viên (0,60) lớn gấp khoảng bốn lần phần mất do xếp hạng (0,15); với V5 hai phần gần
  bằng nhau (0,35 và 0,37) (`results/figures/robustness/candidates.csv`).
- BM25 tự cài đặt cho kết quả rất gần Lucene (Anserini 2.3.0) khi dùng cùng token và tham số (top 10 trùng 98,9%), nhưng
  chậm hơn khoảng 5,5 lần trong một luồng (`results/figures/lucene.md`). Dịch vụ bị giới hạn bởi CPU: thêm worker từ 4 lên
  12 không tăng QPS.

Bảng đầy đủ: `report/latex/tables/` (sinh bởi `scripts/make_report_tables.py`, nguồn ghi trong `SOURCES.md`),
`results/figures/robustness.md`, `results/figures/scale_tables.md`, `results/figures/ablation_significance.md`,
`results/figures/latency_profile.md`, `results/figures/error_slices.md`, `results/figures/failure_cases.md`.

## Benchmark chung với Nhóm 4

So sánh với Nhóm 4 dùng gói đóng băng `n4_shared` (`leandojo-benchmark-4@29dcec07+split-v1.0`): 180.907 premise định danh
bằng `full_name`, 898 theorem của `random/test`, truy vấn là statement của theorem (`docs/n4_shared_benchmark.md`).
`scripts/export_n4_shared.py` dựng lại gói từ dữ liệu gốc đúng từng byte và dừng nếu SHA-256 khác giá trị ghim trong
`configs/n4_shared_manifest.json`, nên không cần chuyền tay tệp zip. R@10 của Nhóm 3 trên gói này
(`results/figures/shared_benchmark.md`):

| Hệ thống | R@10, 882 truy vấn có nhãn | R@10, lát `signature` (727) |
|---|---|---|
| B0 | 0,215 | 0,190 |
| V2 | 0,313 | 0,240 |
| V4 | 0,418 | 0,347 |
| V5 | 0,432 | 0,369 |

158 truy vấn của gói còn chứa phần chứng minh, và phần này thường gọi thẳng tên premise đúng; lát `signature` bỏ các truy
vấn đó. Hai benchmark khác đơn vị truy vấn nên số liệu ở đây không so trực tiếp với bảng phía trên. Khi có run file của
Nhóm 4: `python scripts/eval_shared.py --external N4-dense=<run_dense>.csv` (chấm cùng quy tắc, rồi trộn RRF với V4).
Kết quả trên gói này nằm ở Bảng C.5 của báo cáo.

## Kiến trúc

```
Đường build (batch)                                   Đường serve (online)
LeanDojo v10 ─► extract ─► premises.parquet           proof state + ngữ cảnh (theorem, file, dòng)
                              │                                   │
                  tokenizer (default | lean)          cùng tokenizer, cùng phiên bản
                              │                                   │
     Map: ((field, term), (doc_id, tf))               Stage 1: BM25 / BM25F trên inverted index
     Shuffle: hash(term) mod R (+ salting hot key)    (+ lọc premise truy cập được qua import)
     Reduce: posting sắp theo doc_id, mã d-gap + VB              │
                              │                       Stage 2: reranker tuyến tính (namespace, ký hiệu,
     index: dictionary + postings.bin + manifest      file/thư mục, loại khai báo, popularity, PageRank)
     (bản tuần tự và bản Spark trùng từng byte)                  │
                                                      top-k premise ─► FastAPI /search, CLI
```

| Mã | Biến thể |
|---|---|
| B0 | Baseline bắt buộc: BM25 mặc định (k1 = 1,2; b = 0,75) trên statement, tokenizer tách từ thường |
| B0t | B0 với k1, b, query_tf tinh trên val (baseline mạnh, cùng ngân sách tinh như V1) |
| V1 | + tokenizer Lean (tách tên, giữ ký hiệu toán) |
| V2 | + BM25F theo trường name / namespace / statement |
| V2a | V2 + lọc premise truy cập được |
| V3 | V2 + reranker cấu trúc (không dùng thống kê phụ thuộc, không lọc) |
| V4 | V3 + popularity, PageRank (train nửa A) + lọc truy cập được |

## Cài đặt

Yêu cầu: Python 3.13 cho `.venv`, Python 3.11 cho `.venv-spark`, Java 21 (Spark trên Windows và Lucene), khoảng 3 GB đĩa.
Phiên bản chính xác của mọi gói nằm trong `requirements/main.lock.txt` và `requirements/spark.lock.txt`.

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements/main.lock.txt && .venv/Scripts/python -m pip install -e .
# Spark: trên Windows PySpark không chạy với Python 3.13 (worker crash WinError 10038) nên dùng môi trường riêng Python 3.11
.venv/Scripts/python -m pip install uv && .venv/Scripts/uv python install 3.11
.venv/Scripts/uv venv .venv-spark --python 3.11 && .venv-spark/Scripts/python -m ensurepip
.venv-spark/Scripts/python -m pip install -r requirements/spark.lock.txt && .venv-spark/Scripts/python -m pip install -e .
# Môi trường Linux cho thí nghiệm xây chỉ mục: docker/spark-linux.Dockerfile và docker/run_build2_linux.sh
# Lucene (scripts/lucene_baseline.py): Anserini 2.3.0, SHA-1 824831091154cf9eba36aa1066f08ff70b7835d4
mkdir -p tools/anserini && curl -L -o tools/anserini/anserini-2.3.0-fatjar.jar https://repo1.maven.org/maven2/io/anserini/anserini/2.3.0/anserini-2.3.0-fatjar.jar
# Báo cáo LaTeX: Tectonic 0.15.0 (đặt tectonic.exe vào tools/tectonic/ hoặc PATH) hoặc XeLaTeX (report/latex/README.md)
```

Phiên bản đã dùng cũng được ghi trong mỗi run record (`results/raw/**.json`, trường `software`). Dữ liệu lớn (`data/`,
`indexes/`, `results/runs/`) và công cụ ngoài (`tools/`) không có trong Git: tạo lại bằng `bash scripts/reproduce.sh`
(bước 1 tải dữ liệu từ Zenodo), hoặc chép từ máy đã chạy.

## Chạy nhanh

```bash
python -m pytest -q                                   # 68 test, gồm các ví dụ tính tay của slide Chương 4
python -m serving.cli --query-id "random:val:0:1" --mode V4 -k 5
uvicorn serving.api:app --port 8000 --workers 4    # rồi POST /search {"query": "...", "k": 10, "mode": "V4"}
python -m eval.run_eval --config configs/b0_bm25_default.yaml --eval random_test_all
python scripts/export_n4_shared.py --zip              # gói chung N3/N4 → data/n4_shared/ và data/n4_shared.zip
python scripts/verify_n4_shared.py data/n4_shared.zip # kiểm tra một gói nhận được: SHA-256, schema, quy tắc self
python scripts/eval_shared.py                         # B0, V2, V4, V5 trên gói chung
```

Không có dữ liệu đầy đủ: `data_sample/` có 1.000 premise và 200 truy vấn (JSONL) để đọc thử định dạng.

## Tái tạo toàn bộ kết quả

```bash
bash scripts/reproduce.sh                 # dữ liệu, gói chung → huấn luyện → đánh giá (cả gói chung) → đối chứng, holdout → ablation → phân tích
bash scripts/reproduce.sh --with-tuning   # chạy lại cả quá trình chọn tham số (lưới BM25 cơ bản và mở rộng, BM25F, V5)
bash scripts/reproduce.sh --with-scale    # thí nghiệm quy mô, tải và Lucene (~4 giờ)
bash scripts/reproduce.sh --report        # sinh bảng, hình và biên dịch report/report.pdf
bash report/latex/build.sh                # chỉ dựng lại đúng bản báo cáo đã phát hành (kiểm tra SHA-256 đầu vào trước)
```

Mỗi lần chạy ghi một run record theo Phụ lục B (run_id, timestamp, git_commit, code_hash, data_version, machine,
software, parameters, seed, metrics, status, log). Các lần chạy dùng cho báo cáo được ghim trong
`results/figures/report_manifest.json` và `report_records.json` (`scripts/pin_runs.py`); bảng và hình của báo cáo sinh từ
các lần chạy đã ghim, không phụ thuộc lần chạy mới nhất, và `report/latex/build.sh` dừng nếu một tệp đầu vào khác SHA-256
đã ghi (xem `report/latex/README.md`). SHA-256 của tệp văn bản được tính ở dạng xuống dòng LF (`runlog.file_sha256`),
nên kiểm tra này cho cùng kết quả trên Windows và Linux, với mọi cấu hình `core.autocrlf`. Mọi bảng và biểu đồ trong
`results/figures/` sinh bằng script từ `results/raw/`.

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `src/` | extract, analysis (tokenizer), index (tuần tự, Spark, codec), retrieval (BM25F, đặc trưng, reranker), serving, eval, shared (gói benchmark chung N3/N4) |
| `configs/` | mỗi biến thể một file YAML; lưới tinh tham số; `configs/ablation/` sinh tự động; `n4_shared_manifest.json` ghim gói chung |
| `scripts/` | tải dữ liệu, tập con, thống kê phụ thuộc, huấn luyện, thí nghiệm quy mô, ablation, error analysis, biểu đồ, gói chung N3/N4 |
| `tests/` | unit test (codec, tokenizer, metric, extract, index so với oracle, bộ lọc truy cập được, gói chung) |
| `data_sample/` | mẫu hợp pháp để chạy thử |
| `results/raw/` | run record JSON, metric từng truy vấn, kết quả tinh tham số |
| `results/figures/` | bảng và hình sinh bằng script |
| `results/models/` | trọng số reranker (JSON, vài KB) |
| `docs/` | benchmark spec, gói chung N3/N4, đề cương, 20 truy vấn kiểm tra, data card, data dictionary, ablation plan, hướng dẫn vai trò, system card, AI-use statement, license note |
| `data/`, `indexes/`, `tools/` | dữ liệu, chỉ mục và công cụ ngoài, chỉ có trên máy (không đưa vào Git) |

## Tài liệu
- `report/report.pdf`: báo cáo chính thức (nguồn LaTeX trong `report/latex/`, biên dịch bằng `bash report/latex/build.sh`); `report/report.md` là bản nháp cũ
- `docs/benchmark_spec.md`: benchmark mức tactic dùng trong báo cáo
- `docs/n4_shared_benchmark.md`: gói benchmark chung với Nhóm 4 (schema, SHA-256, quy tắc eligible, cách chấm, lưu ý về truy vấn còn chứa chứng minh)
- `docs/decuong_1_trang.md`: đề cương 1 trang nộp tuần 1 (Phụ lục A)
- `docs/check_queries.md`: 20 truy vấn kiểm tra định nghĩa benchmark (tuần 1)
- `docs/data_dictionary.md`: mô tả từng cột của mọi bảng (tuần 2)
- `docs/ablation_plan.md`: giả thuyết, biến thể và quy tắc kết luận của ablation (tuần 5)
- `docs/role_guides.md`: mỗi vai trò chạy lại gì, phải giải thích gì, câu hỏi bảo vệ hay gặp
- `docs/data_card.md`, `docs/system_card.md`, `docs/license_note.md`, `docs/ai_use_statement.md`
- `contributions.md`: vai trò, người dự phòng, nhật ký đóng góp
