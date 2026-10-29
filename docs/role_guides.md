# Hướng dẫn theo vai trò — Nhóm 3

Đề yêu cầu: *sáu vai trò tương ứng sáu người chính; mỗi vai trò cần một người khác có thể chạy lại và giải thích khi bảo
vệ.* Mỗi mục dưới đây ghi phần việc, lệnh chạy lại, những gì phải giải thích được và câu hỏi bảo vệ hay gặp. Người chính và
người dự phòng của vai trò nên cùng chạy lại các lệnh và tự trả lời các câu hỏi **bằng lời của mình** trước buổi bảo vệ.

Mã nguồn do AI hỗ trợ viết (`docs/ai_use_statement.md`). Giảng viên có thể hỏi bất kỳ dòng nào, nên mỗi người phải đọc
hết các file của vai trò mình.

Quy ước: lệnh chạy trong thư mục repo, dùng `.venv/Scripts/python` (viết tắt `python`) và đặt `PYTHONIOENCODING=utf-8`
trên Windows. Lệnh Spark dùng `.venv-spark/Scripts/python`.

## Bảng tổng hợp

| # | Vai trò | Người chính | Dự phòng | Sản phẩm tuần (theo hướng dẫn) | Trạng thái |
|---|---|---|---|---|---|
| 1 | Data/extraction | Nguyễn Đức Hoàng | Nguyễn Ngọc Minh | T1 benchmark spec, 20 truy vấn kiểm tra; T2 Parquet sample, data dictionary, query set | Xong; gói chung N3/N4 đã ghim (`docs/n4_shared_benchmark.md`); còn ký xác nhận 20 truy vấn |
| 2 | Indexing | Trần Duy Khánh | Vũ Trung Kiên | T2 index sample; T6 index manifest; T7 index size, build time | Xong |
| 3 | BM25 baseline | Nguyễn Mạnh Cường | Trần Đình Dũng | T3 baseline end-to-end, raw ranking; T4 biến thể retrieval, unit test | Xong |
| 4 | Structural features | Trần Đình Dũng | Nguyễn Mạnh Cường | T5 ablation plan, model v0.1 | Xong |
| 5 | Serving/performance | Vũ Trung Kiên | Trần Duy Khánh | T5 latency profile; T6 API/CLI, load test; T7 P95, QPS | Xong |
| 6 | Evaluation/error analysis | Nguyễn Ngọc Minh | Nguyễn Đức Hoàng | T3 Recall@k, MRR; T8 ablation, 30 failure case; T9 so sánh Nhóm 4, báo cáo | Còn gán nhãn tay 30 case và so sánh Nhóm 4 |

---

## 1. Data/extraction

**Phụ trách:** `scripts/download_data.py`, `src/extract/leandojo.py`, `scripts/make_subsets.py`, `scripts/make_sample.py`,
`scripts/make_check_queries.py`, `scripts/make_data_dictionary.py`; benchmark chung với Nhóm 4: `src/shared.py`,
`scripts/export_n4_shared.py`, `scripts/verify_n4_shared.py`, `configs/n4_shared_manifest.json`; tài liệu
`docs/benchmark_spec.md` (cùng vai trò 6), `docs/n4_shared_benchmark.md`, `docs/data_card.md`, `docs/data_dictionary.md`,
`docs/check_queries.md`, `docs/license_note.md`.

**Chạy lại**

```bash
python scripts/download_data.py          # tải Zenodo 12740403, kiểm tra md5 25e1ee60…, ghi data/raw/MANIFEST.json
python -m extract.leandojo            # → data/processed/*.parquet, extract_stats.json
python scripts/make_subsets.py           # tập đánh giá + index con 10K/100K
python scripts/make_check_queries.py     # 20 truy vấn kiểm tra → docs/check_queries.md
python scripts/make_data_dictionary.py   # → docs/data_dictionary.md
python scripts/export_n4_shared.py --zip # gói chung N3/N4 → data/n4_shared/, dừng nếu SHA-256 khác bản ghim
python -m pytest -q tests/test_extract.py tests/test_shared_data.py
```

Kết quả đúng: 180.957 premise; 315.014 truy vấn; 674.628 qrels; dead-letter gồm 115.736 premise đúng không có trong corpus,
20.544 truy vấn bị loại, 16 premise trùng; 20/20 truy vấn kiểm tra đạt C1–C4. Gói chung: 180.907 premise, 898 truy vấn,
SHA-256 khớp `configs/n4_shared_manifest.json`.

**Phải giải thích được**
- Data contract: truy vấn = `state_before`; premise đúng = `annotated_tactic[1]`; không lưu text của tactic vì nó chứa tên premise.
- Định danh premise là `(full_name, path)` vì có 57 tên trùng giữa Lean core và Mathlib; `premise_id` = thứ tự sau khi sắp xếp.
- `statement` cắt tại `:=` cấp ngoài cùng, `where` hoặc nhánh `|` đầu dòng, để không index phần chứng minh.
- Dead-letter: không bản ghi nào mất lặng lẽ; mỗi bản ghi bị loại có lý do.
- Giấy phép: dataset CC BY 2.0, mã Lean Apache-2.0; không đưa dữ liệu lớn vào Git, chỉ đưa `data_sample/`, manifest, checksum.

**Câu hỏi hay gặp**
- *Vì sao bỏ 115.736 premise đúng?* Chúng không có trong `corpus.jsonl`: lemma sinh bởi `@[to_additive]` (`add_comm`),
  constructor (`Or.inl`), trường structure/class (`Iff.mp`). Recall báo cáo là recall trên phần có trong corpus.
- *Làm sao biết trích xuất đúng?* Unit test với ví dụ tính tay; 20 truy vấn kiểm tra đối chiếu với JSON gốc; số dòng khớp
  với `extract_stats.json`.
- *Vì sao index con phải lồng nhau và chứa premise đúng?* Nếu index 10K không chứa đáp án thì recall gần 0 do thiếu đáp án,
  không do hệ thống. Lồng nhau thì khác biệt giữa các mức chỉ do thêm premise gây nhiễu.
- *LeanDojo ra bản mới thì sao?* md5 và phiên bản ghi trong manifest; `premise_id` sẽ đổi nên phải nâng phiên bản spec và cả hai nhóm chạy lại.
- *Vì sao gói chung có 180.907 premise chứ không phải 180.957?* Gói định danh premise bằng `full_name` và giữ lần xuất
  hiện đầu, nên 50 bản ghi của 45 tên có ở cả Lean core lẫn Mathlib bị gộp (`docs/n4_shared_benchmark.md`, mục 2).
- *Vì sao báo thêm lát `signature`?* 158 truy vấn của gói còn chứa chứng minh, và chứng minh thường gọi thẳng tên premise
  đúng; lát này bỏ các truy vấn đó (mục 4).

**Còn phải làm:** ký xác nhận 20 truy vấn trong `docs/check_queries.md`; thống nhất với Nhóm 4 cách xử lý 158 truy vấn
còn chứa chứng minh (giữ gói v1.0 và báo thêm lát `signature`, hay ra phiên bản gói mới).

---

## 2. Indexing

**Phụ trách:** `src/index/build_seq.py`, `build_spark.py`, `codec.py`; `scripts/make_synthetic.py`, `scripts/bench_codec.py`,
`scripts/run_scale.py --part build`; test `tests/test_codec.py`, `tests/test_index_search.py`.

**Chạy lại**

```bash
python -m index.build_seq --fields statement --tokenizer default --codec vb --out indexes/_tmp_seq
.venv-spark/Scripts/python -m index.build_spark --engine arrow --master "local[4]" --fields statement --tokenizer default --codec vb --out indexes/_tmp_spark
# hai index phải trùng từng byte: so phần checksums của hai manifest
python -c "import json;a,b=[json.load(open(f'indexes/_tmp_{x}/index_manifest.json'))['checksums'] for x in ('seq','spark')];print(a==b)"
python -m pytest -q tests/test_codec.py tests/test_index_search.py
python -m pytest -q tests/test_spark_arrow_logic.py   # logic gộp theo partition, không cần Spark
python scripts/run_scale.py --part build --reps 3      # thí nghiệm engine RDD (lâu)
python scripts/run_scale.py --part build2 --reps 3 --sizes 10k,full,x5,x20   # engine Arrow so với tuần tự (khoảng 1,5 giờ)
docker build -t n3-spark-linux -f docker/spark-linux.Dockerfile docker       # cùng thí nghiệm trên Linux
docker run --rm -v "<repo>:/repo" n3-spark-linux bash /repo/docker/run_build2_linux.sh --reps 3 --sizes 10k,full,x5,x20
```

**Phải giải thích được**
- Map phát `((field, term), (doc_id, tf))`; shuffle chia theo `hash(term) mod R`; reduce sắp posting theo `doc_id` rồi nén.
- Kết hợp cục bộ (combiner): đếm tf trong tài liệu trước khi phát, giảm dữ liệu shuffle.
- Skew: term phổ biến (df tới 109.960) dồn vào một reducer; salting chia theo khoảng `doc_id * S // N` rồi gộp lại.
- Nén: d-gap rồi VB hoặc gamma; ví dụ slide 4-27: d-gap [3, 7, 1, 119] → VB `83 87 81 F7`; gamma cùng dãy tốn 22 bit.
- Vì sao bản Spark trùng từng byte bản tuần tự: driver gộp các part file theo thứ tự `(field, term)` cố định.
- Hai engine: RDD (mỗi posting một cặp Python khi shuffle, driver đọc hết dữ liệu) và Arrow (executor tự đọc Parquet,
  `mapInArrow`, gộp posting theo partition nên mỗi khóa chỉ một bản ghi cho mỗi partition, shuffle trong JVM).

**Câu hỏi hay gặp**
- *Spark có nhanh hơn tuần tự không?* Engine RDD thì không, ở mọi kích thước. Engine Arrow thì có khi dữ liệu đủ lớn:
  chỉ mục 3 trường ở 3,6 triệu bản ghi, local[8] 95,6 s so với 366,8 s trên Linux (3,84 lần) và 96,3 s so với 284,7 s
  trên Windows (2,96 lần). Ở 10K, Spark vẫn chậm hơn vì chi phí khởi động phiên và tạo worker Python.
- *Vì sao Windows chậm hơn Linux ở kích thước nhỏ?* PySpark trên Windows không dùng được daemon fork nên mỗi task phải
  khởi động một tiến trình Python mới: ở 10K, local[8] mất 22,6 s trên Windows và 10,9 s trên Linux.
- *Engine Arrow nhanh hơn engine RDD vì đâu?* Không phải do ít byte shuffle hơn (183 so với 204 MB, Spark nén cả hai) mà
  do số cặp Python phải tuần tự hóa: khoảng 67 triệu (RDD) so với 0,5 triệu (Arrow) ở 3,6 triệu bản ghi; driver của
  engine RDD còn giữ toàn bộ dữ liệu nên tốn 11,6 đến 12,5 GB RAM, engine Arrow 2 đến 4 GB.
- *Kết hợp cục bộ được bao nhiêu (engine RDD)?* Shuffle 131 → 52 MB (2,5 lần), thời gian 58,3 → 41,0 s (local[4], 905K).
- *Salting có lợi không?* Task reduce chậm nhất 6,4 → 4,2 s, nhưng tổng thời gian 41,0 → 53,5 s vì thêm pha đếm df và pha gom.
- *Chọn codec nào?* VB: 4,1 MB so với 12,8 MB của raw32; gamma nhỏ nhất (2,6 MB) nhưng giải mã từng bit bằng Python chậm
  khoảng 10 lần khi tắt cache (P50 137,7 ms so với 16,9 ms).
- *Index tái tạo được không?* Có: manifest ghi tokenizer, phiên bản, field, codec, nguồn và sha1 từng file.

---

## 3. BM25 baseline

**Phụ trách:** `src/analysis/tokenizer.py`, `src/retrieval/search.py`, `scripts/tune_bm25.py`; configs B0, B0t, V1, V2 và
`configs/grid_*.yaml`; test `tests/test_tokenizer.py`, `tests/test_index_search.py`.

**Chạy lại**

```bash
python -m eval.run_eval --config configs/b0_bm25_default.yaml --eval random_test_all   # R@10 ≈ 0,100
python scripts/tune_bm25.py --index-config configs/v1_bm25_lean.yaml --grid configs/grid_bm25.yaml   # chỉ trên val
python -m serving.cli --query-id "random:val:0:1" --mode B0 -k 5
python -m pytest -q tests/test_tokenizer.py
```

**Phải giải thích được**
- Công thức BM25, idf dạng không âm của Lucene, vai trò của k1 (bão hòa tf) và b (chuẩn hóa độ dài).
- BM25F: cộng tf đã chuẩn hóa theo từng field (có trọng số) **rồi mới** bão hòa bằng k1, khác với cộng điểm BM25 của từng field.
- `query_tf` nhị phân và đếm: proof state lặp lại tên biến và kiểu nhiều lần; lựa chọn nhị phân được chốt trên val.
- Tokenizer Lean: giữ tên đầy đủ có dấu chấm và các thành phần, tách snake/camelCase, giữ ký hiệu toán, bỏ biến một chữ cái.

**Câu hỏi hay gặp**
- *Baseline có đúng đề?* B0 là BM25 mặc định (k1 = 1,2; b = 0,75) trên statement, tách từ thường, không embedding, không
  dependency graph, cùng dữ liệu, split và máy với phương pháp chính. B0t là baseline mạnh thêm, đã tinh trên val.
- *BM25 tự viết có đúng không?* Lucene (Anserini 2.3.0) dùng đúng token của B0 cho Recall@10, R@50, MRR trùng B0 đến 3 chữ
  số (`scripts/lucene_baseline.py`). Lucene với bộ phân tích tiếng Anh mặc định chỉ đạt 0,076: stemming và stopword làm hỏng
  tên Lean, thêm bằng chứng cho RQ1.
- *Tại sao k1 = 3, b = 1?* Chọn trên val; với V1 điểm tốt nhất nằm ở biên lưới đầu nên lưới được mở rộng (`grid_bm25_ext.yaml`).
- *Tokenizer Lean giúp bao nhiêu?* +0,010 R@10 [+0,002; +0,019] sau khi đã tinh BM25; đây là bước tăng nhỏ nhất của chuỗi.
- *Vì sao chi phí truy vấn không tỉ lệ với số tài liệu?* Chỉ tài liệu có trong posting của term truy vấn mới được cộng điểm;
  chi phí tỉ lệ với tổng độ dài các posting đó.

---

## 4. Structural features (gồm V5)

**Phụ trách:** `src/retrieval/features.py`, `rerank.py`, `src/splits.py`; `scripts/build_dependency_stats.py`,
`scripts/train_reranker.py`; configs V2a, V3, V4; test `tests/test_accessible.py`; tài liệu `docs/ablation_plan.md` (cùng vai trò 6).

**Chạy lại**

```bash
python scripts/build_dependency_stats.py                                   # popularity, PageRank từ train nửa A
python scripts/train_reranker.py --config configs/v4_rerank_dep.yaml --strategy random
python scripts/train_reranker.py --config configs/v4_rerank_dep.yaml --strategy novel_premises
python -m eval.run_eval --config configs/v4_rerank_dep.yaml --eval random_test_all   # R@10 ≈ 0,243
python -m pytest -q tests/test_accessible.py
```

**Phải giải thích được**
- Hai giai đoạn: BM25F lấy 100 ứng viên, hồi quy logistic chấm lại bằng 13 đặc trưng (danh sách ở đầu `features.py`).
- Bộ lọc truy cập được: premise nằm trong bao đóng import của file theorem, hoặc cùng file và **kết thúc** trước dòng bắt đầu theorem.
- Chống rò rỉ: train chia đôi theo md5 tên theorem; nửa A tính thống kê phụ thuộc, nửa B huấn luyện (6.000 truy vấn,
  600.000 cặp, 4.492 cặp dương); `novel_premises` dùng thống kê và mô hình riêng.
- PageRank trên đồ thị theorem → premise của train nửa A: 129.983 cạnh, damping 0,85, hội tụ sau 11 vòng.

**Câu hỏi hay gặp**
- *Lỗi bộ lọc là gì?* Bản đầu so dòng **bắt đầu** nên chính theorem lọt vào ở 1.111/2.641 truy vấn test (vị trí premise tính
  cả docstring). Sửa thành so dòng kết thúc, thêm test hồi quy, huấn luyện và chạy lại V2a/V4.
- *Đặc trưng nào quan trọng nhất?* Bộ lọc (−0,051 R@10 khi bỏ), điểm BM25 (−0,036), popularity + PageRank (−0,018).
  Vị trí file và loại khai báo chỉ có ích cho MRR; namespace không có tác dụng đo được (`ablation_significance.md`).
- *Vì sao dùng mô hình tuyến tính?* Chấm lại 100 ứng viên mất 0,06 ms, trọng số đọc được; hạn chế là chưa thử mô hình phi tuyến.
- *Dùng thống kê từ train có gian lận không?* Không: test không được dùng; chỉ nửa A cho thống kê và nửa B cho huấn luyện.

- *V5 khác V4 thế nào?* Thêm tối đa 250 ứng viên ngoài BM25F: premise dùng nhiều trong cùng tệp, cùng thư mục và phổ biến
  toàn thư viện, cùng hai đặc trưng `file_usage`, `dir_usage`. Số ứng viên chọn trên validation (`scripts/tune_v5.py`).
  Kết quả test `random`: Recall@10 0,243 → 0,281, Recall@50 0,373 → 0,508; độ phủ ứng viên 39,7% → 65,0%.
- *V5 có rò rỉ thông tin không?* Chỉ đếm theorem train-A khai báo trước theorem đang chứng minh (cùng tệp) hoặc ở tệp được
  import (cùng thư mục); có test `tests/test_usage_causal.py`. Trên `novel_premises` mức tăng nhỏ (0,011) vì premise mới chưa
  từng được dùng.

---

## 5. Serving/performance

**Phụ trách:** `src/serving/api.py`, `cli.py`; `src/eval/loadtest.py`; `scripts/run_scale.py --part serve`,
`scripts/profile_latency.py`; môi trường `.venv` và `.venv-spark`.

**Chạy lại**

```bash
uvicorn serving.api:app --port 8000 --workers 4      # cửa sổ 1; chờ /health trả ok (warm-up)
curl -s http://127.0.0.1:8000/health
python -m eval.loadtest --mode V4 --clients 1,10,50 --requests 400 --warmup 50   # cửa sổ 2
python scripts/profile_latency.py                         # thời gian theo bước → results/figures/latency_profile.md
python scripts/run_scale.py --part serve --reps 3         # thí nghiệm tải đầy đủ (lâu)
```

**Phải giải thích được**
- API: `POST /search` nhận proof state và ngữ cảnh (theorem, file, dòng), trả top-k; `/batch_search`; `/health`; warm-up khi khởi động.
- GIL: một tiến trình Python chỉ chạy một luồng bytecode tại một thời điểm, nên dùng 4 worker (4 tiến trình, mỗi tiến trình
  nạp index riêng).
- P50/P95/P99, QPS, bão hòa: khi số client vượt số worker, request phải xếp hàng và P95 tăng mạnh.
- Profile: với V4, cộng điểm BM25F chiếm 88% thời gian một truy vấn; bộ lọc ≈ 0,9 ms, đặc trưng ≈ 0,5 ms, rerank ≈ 0,06 ms.

**Câu hỏi hay gặp**
- *Độ trễ V4?* P95 95 ms, 130 QPS ở 10 client, index đầy đủ. Ở 50 client cả ba hệ thống bão hòa (P95 346–567 ms).
- *Vì sao V4 chỉ chậm hơn V2 khoảng 2 ms?* Phần thêm của V4 chỉ làm trên 100 ứng viên; phần đắt là stage 1, chung cho cả hai.
- *4 lỗi trên 32.400 request là gì?* Lỗi phía client, log server không có exception; loadtest nay ghi `error_types`; ghi
  là hạn chế trong báo cáo.
- *Muốn nhanh hơn thì làm gì?* Cắt sớm MaxScore/WAND, cache kết quả truy vấn, viết stage 1 bằng ngôn ngữ biên dịch hoặc
  dùng Lucene/OpenSearch.

- *Thêm worker có tăng thông lượng không?* Không: 4, 8, 12 worker cho cùng khoảng 230 đến 260 QPS (B0), RAM tăng từ 3,5
  lên 10 GB. Lý do: với 4 worker, tổng CPU của server đã là 8 đến 9,5 trên 12 luồng logic; nhiều tiến trình client cũng không
  tăng QPS nên client không phải nút thắt (`scripts/serve_cpu_check.py`).
- *Vậy muốn nhanh hơn thì sao?* Giảm CPU mỗi truy vấn: BM25 của Lucene cho cùng kết quả nhưng 2,6 ms so với 13,6 ms.

---

## 6. Evaluation/error analysis

**Phụ trách:** `src/eval/metrics.py`, `run_eval.py`; `scripts/summarize_quality.py`, `ablation.py`,
`ablation_significance.py`, `error_analysis.py`, `make_figures.py`, `make_report_tables.py`, `fuse_rrf.py`;
tài liệu `docs/benchmark_spec.md`, `docs/ablation_plan.md`, `report/`.

**Chạy lại**

```bash
python -m pytest -q tests/test_metrics.py
python scripts/summarize_quality.py          # bảng chất lượng + paired bootstrap
python scripts/ablation_significance.py      # KTC cho ablation
python scripts/error_analysis.py             # lát cắt + 30 failure case
bash report/latex/build.sh                   # sinh bảng, hình và biên dịch report/report.pdf
python scripts/fuse_rrf.py --runs results/runs/<run_V4>.tsv <run_nhom4>.tsv --eval random_test_all   # khi có run của Nhóm 4
python scripts/eval_shared.py --external N4-dense=<run_dense>.csv   # N3 và Nhóm 4 trên benchmark chung (n4_shared)
```

**Phải giải thích được**
- Recall@k = |đúng ∩ top-k| / |đúng|, trung bình theo truy vấn; MRR; nDCG@10 với nhãn nhị phân.
- Paired bootstrap: lấy mẫu lại truy vấn, so hai hệ thống trên cùng truy vấn; KTC 95%; p một phía (bảng chất lượng) và
  hai phía (ablation); vấn đề nhiều phép so sánh (Bonferroni).
- Primary metrics: Recall@10 và P95 ở 10 client; các metric khác để giải thích đánh đổi.
- RRF: điểm = Σ 1/(60 + hạng), không cần chuẩn hóa điểm giữa hai hệ thống.

**Câu hỏi hay gặp**
- *Kết quả có tất định không?* Có: hai lần chạy B0 cho run file trùng md5; biến thiên được đo bằng bootstrap trên truy vấn.
- *Error analysis kết luận gì?* 82,7% premise đúng bị bỏ lỡ nằm ngoài top-100 của stage 1, nên reranker không thể cứu;
  hướng tiếp theo là hybrid với dense của Nhóm 4 hoặc tăng K.
- *Tại sao chọn split random làm chính?* Nó phản ánh tình huống tìm trong thư viện có sẵn; `novel_premises` báo kèm để đo
  trường hợp premise đúng chưa từng được dùng trong train.
- *So sánh với Nhóm 4 thế nào?* Trên gói chung `n4_shared` (898 theorem của random/test): `eval_shared.py` chấm run file
  của hai nhóm cùng một quy tắc eligible và kiểm định hoán vị ghép cặp; kết quả N3 đã có ở `results/figures/shared_benchmark.md`.

**Còn phải làm:** hai người gán nhãn độc lập 30 failure case (`results/figures/failure_cases.md`) rồi đối chiếu và ghi tỉ
lệ đồng thuận; bảng so sánh với Nhóm 4 khi có run file của họ; kiểm tra trích dẫn trong báo cáo.
