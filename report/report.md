# Truy hồi premise Lean bằng chỉ mục thưa và tín hiệu cấu trúc ở quy mô thư viện mathlib4

**Nhóm 3 — INT4418 Dữ liệu lớn, PTIT, 2026.** Nguyễn Đức Hoàng, Trần Duy Khánh, Nguyễn Mạnh Cường,
Trần Đình Dũng, Vũ Trung Kiên, Nguyễn Ngọc Minh.

> Bản nháp. Mọi con số lấy từ `results/figures/*.md` (sinh bằng script). Các mục đánh dấu [TODO] cần nhóm hoàn thiện.

## Tóm tắt

Nhóm xây hệ thống truy hồi premise cho Lean 4 trên 180.957 khai báo của mathlib4 (LeanDojo Benchmark 4 v10). Chỉ mục đảo
được build theo lô bằng Python tuần tự và bằng Spark (hai bản trùng từng byte), phục vụ qua FastAPI với BM25F và một reranker
tuyến tính trên tín hiệu cấu trúc: namespace, ký hiệu, vị trí file, bộ lọc import, popularity và PageRank của đồ thị phụ
thuộc. Trên toàn bộ test split `random`, Recall@10 tăng từ 0,100 (BM25 mặc định) lên 0,243 và MRR từ 0,065 lên 0,196; mọi
bước cải tiến có ý nghĩa thống kê. P95 ở 10 client tăng từ 60 lên 95 ms. Thí nghiệm ở 4 kích thước index và 3 mức tải cho
thấy recall giảm và độ trễ tăng gần tuyến tính khi kho lớn hơn; error analysis chỉ ra 82,7% lỗi nằm ở bước tạo ứng viên.

## 1. Giới thiệu

Bộ chứng minh tự động cho Lean cần biết nên dùng những định lý nào trong thư viện. mathlib4 ở snapshot nghiên cứu có
180.957 khai báo; với mỗi bước chứng minh, so truy vấn với toàn bộ thư viện là quá đắt, và đưa toàn bộ thư viện vào
bộ chứng minh là bất khả thi. Bài toán *premise retrieval* là: cho trạng thái chứng minh hiện tại, trả về một danh sách
ngắn các premise có khả năng được dùng ở bước tiếp theo.

Vai trò của dữ liệu lớn nằm ở ba chỗ. Thứ nhất, kho premise đủ lớn để truy vấn tuyến tính tốn kém, nên cần xây chỉ mục
đảo theo lô và phục vụ truy vấn trực tuyến. Thứ hai, phân bố token rất lệch: token phổ biến nhất xuất hiện trong khoảng
110 nghìn statement, tạo hot key khi shuffle và posting list dài khi truy vấn. Thứ ba, hệ thống phải phục vụ nhiều truy
vấn đồng thời với độ trễ đuôi thấp.

Nhóm trả lời hai câu hỏi:
- **RQ1.** BM25 có đủ mạnh cho dữ liệu Lean, hay cần thêm trường ký hiệu, namespace và dependency?
- **RQ2.** Kết hợp sparse score với tín hiệu cấu trúc có tăng Recall@k mà vẫn giữ độ trễ thấp không?

Câu hỏi trung tâm của đề bài là *tín hiệu cấu trúc cải thiện recall hay chỉ làm tăng chi phí*; nhóm trả lời bằng
Recall@10 và P95 latency trên cùng một benchmark.

## 2. Công trình liên quan

- **BM25 và BM25F.** Robertson & Zaragoza (2009) tổng kết khung xác suất cho BM25 và mở rộng nhiều trường BM25F, nền
  tảng của stage 1 trong hệ thống này.
- **LeanDojo / ReProver.** Yang và cộng sự (NeurIPS 2023) công bố LeanDojo Benchmark, định nghĩa bài toán premise
  retrieval trên proof state, hai split `random` và `novel_premises`, và cho thấy giới hạn tìm kiếm trong premise truy
  cập được giúp truy hồi tốt hơn. Nhóm dùng lại benchmark này và ý tưởng bộ lọc truy cập được, nhưng giữ toàn bộ hệ
  thống ở dạng thưa, không dùng mô hình neural.
- **Lọc theo ký hiệu trong ITP.** MePo của Meng & Paulson (2009), dùng trong Sledgehammer của Isabelle, chọn premise
  theo sự trùng ký hiệu với goal; đặc trưng `sym_jaccard` và `ns_in_query` của nhóm cùng tinh thần.
- **Premise selection học sâu.** DeepMath (Irving và cộng sự, 2016) và Magnushammer (Mikuła và cộng sự, 2023) học biểu
  diễn để chọn premise; đây là hướng dense mà Nhóm 4 khảo sát, Nhóm 3 dùng làm điểm so sánh ở tuần 9.
- **Xây chỉ mục phân tán.** MapReduce (Dean & Ghemawat, 2004) và Spark RDD (Zaharia và cộng sự, 2012) là mô hình tính
  toán cho đường build; kỹ thuật kết hợp cục bộ, secondary sort và nén d-gap/VB/gamma theo Manning, Raghavan & Schütze (2008)
  và Lin & Dyer (2010).
- **PageRank** (Page và cộng sự, 1999) được dùng trên đồ thị phụ thuộc theorem → premise làm đặc trưng độ quan trọng.

Khoảng trống: các hệ thống premise retrieval gần đây chủ yếu là dense. Nhóm đo xem riêng chỉ mục thưa cộng các tín
hiệu cấu trúc rẻ đi được bao xa, với chi phí build và độ trễ đo trên nhiều quy mô.

## 3. Dữ liệu

LeanDojo Benchmark 4 v10 (Zenodo 12740403, CC BY 2.0), trích từ mathlib4 commit `29dcec07`. Sau khi trích:
180.957 premise trong 5.674 file (Mathlib 156.947; Lean core 19.619; Batteries 2.883; còn lại Aesop, ProofWidgets, Qq).
Mỗi premise định danh bằng `(full_name, path)` vì có 57 tên trùng giữa Lean core và Mathlib.

Một truy vấn là `state_before` của một tactic có premise được chú thích; premise đúng là các premise đó. 115.736 lượt
premise đúng không có trong corpus (lemma do `@[to_additive]` sinh, constructor, trường structure/class) bị bỏ;
20.544 truy vấn không còn premise đúng bị loại. Tập đánh giá: toàn bộ test của `random` (2.641 truy vấn) và
`novel_premises` (4.141). Tham số chỉ tinh trên val. Train được chia đôi cố định theo md5 tên theorem: nửa A tính thống
kê phụ thuộc, nửa B huấn luyện reranker, để đặc trưng không chứa nhãn của chính truy vấn huấn luyện.

Trước khi chạy thí nghiệm, nhóm kiểm tra định nghĩa trên 20 truy vấn val chọn theo lĩnh vực (`docs/check_queries.md`),
đối chiếu với tactic gốc trong JSON. Cả 20 truy vấn đạt bốn kiểm tra tự động: premise đúng xuất hiện trong tactic, truy
cập được từ theorem, không phải chính theorem, và premise bị bỏ đều nằm trong dead-letter. Khi đọc lại có một đặc điểm
của chú thích cần lưu ý: `rfl` trong mẫu `intro | rfl, rfl` cũng được tính là premise. Phần ký xác nhận của người kiểm
tra: [TODO nhóm]. Mô tả từng cột của mọi bảng nằm
trong `docs/data_dictionary.md`.

Cho thí nghiệm quy mô, nhóm tạo bốn index lồng nhau 10K ⊂ 50K ⊂ 100K ⊂ 181K luôn chứa 870 premise đúng của một mẫu
500 truy vấn test, và một corpus tổng hợp ×5 (905K bản ghi, nhân bản) chỉ để đo build và skew.

## 4. Phương pháp và kiến trúc

### 4.1 Đường build
Đơn vị dữ liệu là một premise với ba trường `name`, `namespace`, `statement` (phần phát biểu đã bỏ attribute và thân
định nghĩa). Map phân tích văn bản, đếm cục bộ trong tài liệu và phát `((field, term), (doc_id, tf))`. Shuffle chia
partition theo `hash((field, term)) mod R`; với hot key (df ≥ ngưỡng), key được tách theo khoảng doc_id
(`salt = doc_id · S / N`) để các phần sau khi gom nối lại vẫn đúng thứ tự mà không phải sắp lại. Reduce sắp posting
theo doc_id, tính df, cf, mã hóa d-gap + Variable-byte và ghi shard trên executor; driver trộn dictionary các shard theo
thứ tự (field, term). Bản tuần tự (Python thuần) và bản Spark sinh index **trùng từng byte** (kiểm tra bằng SHA-1).
Độ phức tạp: map O(tổng số token); shuffle O(số posting); reduce O(P log P) theo độ dài posting.

### 4.2 Đường serve
Stage 1 tính BM25F trên posting của các term trong truy vấn, nên chi phí tỉ lệ với tổng độ dài posting được đọc chứ
không tỉ lệ với số premise. Tùy chọn lọc premise truy cập được: bao đóng import của file chứa theorem cộng các khai báo
kết thúc trước theorem trong cùng file. Stage 2 lấy top-100 và chấm lại bằng logistic regression trên các đặc trưng:
điểm và hạng stage 1, namespace (trùng với truy vấn, tiền tố chung với theorem), cùng file, thư mục chung, độ trùng ký
hiệu toán, loại khai báo, popularity và PageRank. Service FastAPI chạy nhiều tiến trình worker.

### 4.3 Tokenizer Lean
Tách tên đầy đủ theo dấu chấm và giữ cả tên đầy đủ (`Nat.add_comm` → `nat.add_comm`, `nat`, `add_comm`, `add`, `comm`),
tách camelCase, giữ ký hiệu toán làm token (`∀ → ↔ ≤ ∣ ⁻¹ ℕ ℝ`), bỏ biến một chữ cái ASCII. Không dùng NFKC vì NFKC đổi
`ℝ` thành `R`.

## 5. Thiết lập thí nghiệm

**Phần cứng và phần mềm.** Một máy Windows 11, CPU Intel thế hệ Alder Lake 6 nhân / 12 luồng, 32 GB RAM, SSD.
Python 3.13 (numpy 2.5, pyarrow 26, scikit-learn 1.9, FastAPI/uvicorn) cho mọi phần trừ Spark; Spark 4.0.1 chạy local
mode với Python 3.11 và Java 21 (PySpark trên Windows chưa chạy được với Python 3.13). Mỗi run record ghi đủ phiên bản.

**Biến thể.** B0 (baseline bắt buộc, BM25 mặc định k1 = 1,2, b = 0,75), B0t (B0 tinh tham số), V1 (+ tokenizer Lean),
V2 (+ BM25F 3 trường), V2a (V2 + lọc truy cập được), V3 (V2 + reranker cấu trúc), V4 (V3 + popularity, PageRank + lọc).
Tham số stage 1 chọn bằng lưới trên random/val (B0t, V1: k1 ∈ {0,6 … 6}, b ∈ {0,3 … 1}, query_tf ∈ {count, binary};
V2: 20 tổ hợp trọng số/b). Reranker: logistic regression (C = 1), 6.000 truy vấn train nửa B, top-100 ứng viên.

**Quy mô và tải.** Build: 10K / 50K / 100K / 181K / 905K (tổng hợp) × {tuần tự, Spark local[2], local[4], local[8]}
× 3 lần, cộng ablation Spark (không kết hợp cục bộ; salting 4 khoảng với df ≥ 20.000) và codec (raw32 / VB / gamma).
Chất lượng theo quy mô: B0, V2, V4 trên 4 index lồng nhau, mẫu 500 truy vấn test (seed 42). Tải: API 4 worker,
1 / 10 / 50 client, 400 request mỗi mức sau 50 request warm-up, × 3 lần, ở cả 4 kích thước index.

**Metric.** Recall@1/5/10/50, MRR, nDCG@10 trung bình theo truy vấn; paired bootstrap 10.000 mẫu cho chênh lệch;
P50/P95/P99 đầu-cuối và QPS cho tải; wall time, docs/s, RAM đỉnh (cả cây tiến trình), shuffle bytes, spill,
thời gian task reduce max/median cho build. Seed 42 cho mọi lấy mẫu. Báo median và min–max qua các lần lặp.

## 6. Kết quả

### 6.1 Chất lượng truy hồi (RQ1, RQ2)

Toàn bộ test, 180.957 premise ứng viên (`results/figures/quality.md`):

| Biến thể | random R@1 | random R@10 | random R@50 | random MRR | novel R@10 | novel MRR |
|---|---|---|---|---|---|---|
| B0 (baseline bắt buộc) | 0,018 | 0,100 | 0,176 | 0,065 | 0,127 | 0,090 |
| B0t (B0 tinh tham số) | 0,021 | 0,124 | 0,224 | 0,078 | 0,155 | 0,106 |
| V1 (+ tokenizer Lean) | 0,023 | 0,134 | 0,245 | 0,087 | 0,163 | 0,110 |
| V2 (+ BM25F) | 0,023 | 0,147 | 0,270 | 0,093 | 0,183 | 0,124 |
| V3 (+ rerank cấu trúc) | 0,029 | 0,168 | 0,296 | 0,112 | 0,205 | 0,140 |
| V2a (V2 + lọc truy cập được) | 0,053 | 0,183 | 0,328 | 0,148 | 0,221 | 0,176 |
| **V4 (đầy đủ)** | **0,072** | **0,243** | **0,373** | **0,196** | **0,269** | **0,211** |

Mọi bước trong chuỗi B0 → B0t → V1 → V2 → V3 → V4 tăng R@10 có ý nghĩa thống kê (paired bootstrap, p < 0,01 trên cả hai
split). Bước yếu nhất là tokenizer Lean sau khi đã tinh BM25: +0,010 R@10, CI 95% [+0,002; +0,019]. So với B0, V4 tăng
+0,142 R@10 [+0,129; +0,155] và +0,130 MRR, tức gấp 2,4 lần R@10 và 3 lần MRR.

**Trả lời RQ1.** BM25 mặc định chưa đủ: R@10 chỉ 0,100. Riêng việc tinh k1, b và dùng query_tf nhị phân đã thêm 0,023;
ký hiệu Lean và các trường name/namespace thêm 0,023 nữa. Đóng góp lớn nhất đến từ thông tin phụ thuộc: bộ lọc import
(+0,036 khi thêm vào V2) và popularity/PageRank. BM25 cần cả trường ký hiệu, namespace và dependency.

### 6.2 Đánh đổi chất lượng – độ trễ (RQ2 và câu hỏi trung tâm)

Ở index đầy đủ và 10 client đồng thời (`fig_tradeoff.png`, `scale_tables.md`), P95 đầu-cuối là 60 ms (B0), 93 ms (V2) và
95 ms (V4); QPS tương ứng 259, 166 và 130. Phần lớn chi phí tăng thêm nằm ở stage 1 BM25F ba trường với tokenizer Lean
(B0 → V2: +33 ms P95). Reranker cấu trúc và bộ lọc chỉ thêm khoảng 2 ms P95 nhưng nâng R@10 trên mẫu 500 truy vấn từ 0,143
lên 0,228. Profile theo giai đoạn (`results/figures/latency_profile.md`, một luồng, cache đã nóng) xác nhận điều này: với V4,
bước cộng điểm BM25F có trung vị 18,3 ms và chiếm 88% thời gian; bộ lọc import 0,9 ms, tính 13 đặc trưng 0,5 ms (2,3 ms khi
cache ký hiệu còn lạnh), rerank 0,06 ms. **Tín hiệu cấu trúc cải thiện recall thật, với chi phí độ trễ nhỏ so với chi phí của
chính stage 1.** Ở 50 client,
cả ba hệ thống đều bão hòa (P95 346–567 ms, `fig_latency_vs_clients.png`) vì service chỉ có 4 tiến trình Python; ở mức tải này, giới hạn là thông lượng
của service chứ không phải phương pháp.

### 6.3 Quy mô kho premise

Khi index tăng từ 10K lên 181K premise (`fig_quality_vs_size.png`, `fig_latency_vs_size.png`):
- R@10 giảm đều: B0 0,226 → 0,101; V2 0,343 → 0,143; V4 0,454 → 0,228. Kho lớn hơn chứa nhiều premise gần giống nhau hơn,
  nên bài toán khó hơn chứ không chỉ chậm hơn. V4 giữ khoảng cách khoảng gấp đôi B0 ở mọi quy mô.
- P95 một luồng tăng gần tuyến tính theo kích thước: B0 1,2 → 22,6 ms; V4 6,9 → 40,1 ms, vì posting list của token phổ biến
  dài ra. Ở 10 client: B0 10 → 60 ms; V4 28 → 96 ms.

### 6.4 Build index theo lô

| Builder | 10K | 50K | 100K | 181K | 905K (×5) |
|---|---|---|---|---|---|
| Tuần tự (1 nhân) | 1,2 s | 2,5 s | 3,9 s | 6,1 s | 17,8 s |
| Spark local[2] | 16,1 s | 17,5 s | 19,2 s | 22,4 s | 47,8 s |
| Spark local[4] | 23,7 s | 24,7 s | 26,4 s | 27,4 s | 41,0 s |
| Spark local[8] | 39,3 s | 40,1 s | 41,7 s | 42,0 s | 50,4 s |

Median của 3 lần; wall time gồm khởi động Spark (`fig_build_time.png`). Bản tuần tự nhanh nhất ở mọi quy mô đo được vì
toàn bộ dữ liệu vừa RAM một máy (đỉnh 821 MB ở 905K). Spark có overhead cố định khoảng 15–40 giây, chủ yếu do mỗi task phải
khởi động một tiến trình Python trên Windows. Thông lượng phần job của local[4] tăng từ 596 lên 28.324 premise/giây khi dữ
liệu tăng 90 lần, và ở 905K thì local[4] đã nhanh hơn local[2]. local[8] chậm nhất vì máy chỉ có 6 nhân vật lý. Spark cũng
tốn RAM hơn (3,4–3,9 GB ở 905K, `fig_build_memory.png`) vì driver giữ toàn bộ dữ liệu đầu vào khi `parallelize`.

## 7. Ablation, error analysis và threats to validity

### 7.1 Ablation

Kế hoạch ablation (giả thuyết, biến thể, quy tắc kết luận) ở `docs/ablation_plan.md`. V4 đầy đủ trên random/test: R@10
0,243, MRR 0,196. Mỗi biến thể huấn luyện lại reranker trên cùng dữ liệu; Δ kèm KTC 95% paired bootstrap
(`results/figures/ablation.md`, `ablation_significance.md`):

| Bỏ / đổi | ΔR@10 [KTC 95%] | ΔMRR [KTC 95%] |
|---|---|---|
| bộ lọc truy cập được | −0,051 [−0,060; −0,043] | −0,066 [−0,073; −0,058] |
| điểm và hạng BM25 trong reranker | −0,036 [−0,045; −0,028] | −0,046 [−0,056; −0,037] |
| popularity + PageRank | −0,018 [−0,028; −0,009] | −0,015 [−0,022; −0,009] |
| độ trùng ký hiệu | −0,009 [−0,016; −0,002] | −0,008 [−0,013; −0,004] |
| loại khai báo | −0,001 [−0,006; +0,004] | −0,008 [−0,012; −0,005] |
| cùng file / thư mục | +0,000 [−0,007; +0,007] | −0,010 [−0,014; −0,005] |
| namespace | −0,000 [−0,003; +0,003] | −0,000 [−0,002; +0,002] |
| K = 50 ứng viên | −0,009 [−0,016; −0,002] | −0,007 [−0,010; −0,003] |
| K = 200 ứng viên | +0,010 [+0,003; +0,016] | +0,006 [+0,002; +0,010] |

Bộ lọc import là thành phần có giá trị nhất. Đặc trưng vị trí và loại khai báo không đổi R@10 (KTC chứa 0) nhưng giảm MRR
có ý nghĩa khi bỏ, tức chúng giúp đưa premise đúng lên đầu danh sách. Đặc trưng namespace của reranker không có tác dụng đo
được ở cả hai metric; giả thuyết (chưa kiểm tra) là BM25F đã khai thác trường namespace ở stage 1. Tăng K lên 200 còn tăng
R@50 thêm 0,038, khớp với error analysis: giới hạn nằm ở stage 1. V4 giữ K = 100 vì giá trị này đã chốt trước khi chạy test.

**Ablation build (Spark local[4], 905K).** Bỏ kết hợp cục bộ làm shuffle tăng từ 52 MB lên 131 MB (2,5 lần) và thời gian từ
41,0 lên 58,3 giây. Salting 4 khoảng cho hot key (df ≥ 20.000) giảm thời gian task reduce chậm nhất từ 6,4 xuống 4,2 giây,
nhưng thêm một pha đếm df và một pha gom nên tổng thời gian tăng lên 53,5 giây. Ở quy mô này skew chưa đủ lớn để salting có lợi.

**Codec posting (B0, tắt cache).** raw32 12,8 MB; VB 4,1 MB (nhỏ hơn 3,1 lần); gamma 2,6 MB (4,8 lần). P50 truy vấn lần lượt
12,7 / 16,9 / 137,7 ms. VB là điểm cân bằng tốt; gamma giải mã từng bit bằng Python nên chậm khoảng 10 lần. Khi bật cache
posting đã giải mã (mặc định), ba codec có độ trễ như nhau vì token nóng luôn nằm trong cache. Mọi codec cho cùng kết quả
truy hồi (unit test kiểm tra decode(encode(x)) = x).

### 7.2 Error analysis

V4 bỏ lỡ 4.369 trên 5.483 cặp (truy vấn, premise đúng) ở top-10 (`results/figures/error_slices.md`). Phân loại tự động (một
cặp có thể thuộc nhiều nhóm): 82,7% nằm ngoài top-100 của stage 1; 30,9% là premise rất phổ biến (top 1% popularity), thường là
lemma tổng quát về `≤`, `+`, `∈`; 24,7% không trùng từ nào với truy vấn; 16,2% là định nghĩa, instance hoặc structure.

Theo lát cắt, V4 tốt hơn B0 ở mọi chủ đề: tăng mạnh nhất ở LinearAlgebra (+0,201) và FieldTheory (+0,163), yếu nhất ở
Combinatorics (+0,049) và NumberTheory (+0,059). Truy vấn ngắn (≤ 20 token) dễ nhất (R@10 0,378), truy vấn dài (> 160 token)
khó nhất (0,169); truy vấn cần từ 5 premise trở lên chỉ đạt 0,120.

30 failure case lấy mẫu phân tầng nằm trong `results/figures/failure_cases.md` để hai thành viên gán nhãn thủ công
[TODO nhóm]. Chính việc đọc các case này giúp nhóm phát hiện lỗi bộ lọc để lọt chính theorem đang chứng minh (mục 7.3).

### 7.3 Threats to validity

- **Định nghĩa premise đúng.** Benchmark chỉ tính premise có trong corpus; 115.736 lượt premise đúng (lemma sinh bởi
  `@[to_additive]`, constructor, trường structure) bị bỏ. Recall báo cáo vì vậy là recall trên phần tìm được, cao hơn
  recall "thật" đối với bộ chứng minh. So sánh giữa các biến thể vẫn công bằng vì cùng định nghĩa.
- **Lỗi bộ lọc phát hiện trong quá trình làm.** Phiên bản đầu của bộ lọc truy cập được để lọt chính theorem đang chứng
  minh ở 42% truy vấn test (so dòng bắt đầu thay vì dòng kết thúc). Lỗi được phát hiện nhờ đọc failure case, đã sửa và
  có test hồi quy (`tests/test_accessible.py`); mọi số liệu V2a/V4 trong báo cáo là sau khi sửa. Các biến thể không lọc
  (B0–V3) vẫn có thể trả về chính theorem; điều này chỉ làm chúng kém đi, không làm kết quả đẹp giả.
- **Tinh tham số.** Lưới k1, b, trọng số có giới hạn và chỉ trên val; điểm tối ưu của V1 nằm ở biên lưới đầu tiên nên
  lưới đã được mở rộng. B0 được báo cả bản mặc định (đúng đề) và bản đã tinh (B0t) để so sánh công bằng.
- **Một máy, Windows, local mode.** Mọi số đo thời gian trên một máy 12 luồng, 32 GB. PySpark trên Windows không dùng
  được daemon fork, nên mỗi task khởi động một tiến trình Python mới; overhead này không đại diện cho cụm Linux và làm
  Spark thua bản tuần tự ở mọi quy mô đo được. Cần chạy lại trên Linux/WSL hoặc cụm thật trước khi kết luận về Spark.
- **Corpus tổng hợp ×5.** Nhân bản văn bản giữ nguyên phân phối term và nhân df đúng 5 lần; posting list dài và đều
  hơn dữ liệu thật, nên chỉ dùng để đo build/skew, không dùng cho chất lượng.
- **Độ trễ phụ thuộc cài đặt.** Service viết bằng Python/numpy; mỗi tiến trình bị GIL giới hạn. Số đo phản ánh cài đặt
  này, không phải giới hạn của BM25. Các lần chạy chất lượng song song (tag `quality-parallel`) không dùng để báo độ trễ.
- **novel_premises.** Thống kê phụ thuộc và reranker cho split này được tính lại từ `novel_premises/train` để tránh rò
  rỉ; nếu dùng chung mô hình của `random` thì kết quả sẽ bị thổi phồng.
- **Nhiều phép so sánh.** Ablation có 9 phép kiểm định. Với hiệu chỉnh Bonferroni (ngưỡng ≈ 0,0056), hai kết luận yếu
  nhất về R@10 không còn đứng: độ trùng ký hiệu (p = 0,010) và K = 50 (p = 0,013) (`docs/ablation_plan.md` §3).
- **Một lần chạy cho chất lượng.** Chất lượng là tất định với cùng dữ liệu và tham số (hai lần chạy B0 trên random/test
  cho run file trùng md5); độ biến thiên được đo bằng bootstrap trên truy vấn thay vì lặp lại.

## 8. Kết luận

Chỉ mục thưa cộng các tín hiệu cấu trúc rẻ đưa Recall@10 của premise retrieval trên mathlib4 từ 0,100 lên 0,243 (random)
và từ 0,127 lên 0,269 (novel_premises), MRR gấp ba. Chi phí thêm khoảng 35 ms P95 ở 10 client, phần lớn nằm ở stage 1 chứ
không ở tín hiệu cấu trúc. Thông tin phụ thuộc (bộ lọc import, popularity, PageRank) đóng góp nhiều nhất. Về hệ thống, chỉ
mục build bằng Spark trùng từng byte với bản tuần tự; ở quy mô 181K–905K premise trên một máy, bản tuần tự vẫn nhanh hơn vì
overhead của Spark, và kết hợp cục bộ quan trọng hơn salting.

**Sản phẩm bàn giao:** benchmark split và spec, inverted index tái tạo được, search API/CLI, BM25 baseline, reranker cấu
trúc, script đánh giá/ablation/error analysis, run record và biểu đồ sinh bằng script.

**Hướng tiếp theo.** (1) Trộn với dense retrieval của Nhóm 4 bằng RRF (`scripts/fuse_rrf.py`) để giảm 82,7% lỗi do stage 1;
(2) tăng K hoặc dùng cắt sớm MaxScore/WAND để giữ độ trễ khi K lớn; (3) chạy lại thí nghiệm build trên Linux/WSL hoặc cụm
nhiều máy, nơi PySpark dùng được daemon fork; (4) thử với bộ chứng minh Lean để đo proof success rate.

## Tài liệu tham khảo
1. S. Robertson, H. Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in IR, 2009.
2. K. Yang et al. LeanDojo: Theorem Proving with Retrieval-Augmented Language Models. NeurIPS 2023 (Datasets and Benchmarks).
3. J. Meng, L. C. Paulson. Lightweight relevance filtering for machine-generated resolution problems. Journal of Applied Logic, 2009.
4. G. Irving et al. DeepMath — Deep Sequence Models for Premise Selection. NeurIPS 2016.
5. M. Mikuła et al. Magnushammer: A Transformer-Based Approach to Premise Selection. arXiv:2303.04488, 2023.
6. J. Dean, S. Ghemawat. MapReduce: Simplified Data Processing on Large Clusters. OSDI 2004.
7. M. Zaharia et al. Resilient Distributed Datasets. NSDI 2012.
8. C. D. Manning, P. Raghavan, H. Schütze. Introduction to Information Retrieval. Cambridge University Press, 2008.
9. J. Lin, C. Dyer. Data-Intensive Text Processing with MapReduce. Morgan & Claypool, 2010.
10. L. Page, S. Brin, R. Motwani, T. Winograd. The PageRank Citation Ranking: Bringing Order to the Web. Stanford, 1999.

[TODO] Nhóm mở từng nguồn để kiểm tra trích dẫn trước khi nộp (xem docs/ai_use_statement.md).

## Phụ lục
- AI-use statement: `docs/ai_use_statement.md`
- Đóng góp: `contributions.md`
- Lệnh tái lập: `bash scripts/reproduce.sh --with-tuning --with-scale`
