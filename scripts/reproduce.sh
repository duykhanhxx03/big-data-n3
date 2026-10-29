#!/usr/bin/env bash
# Tái tạo kết quả của báo cáo bằng một lệnh:   bash scripts/reproduce.sh [--with-tuning] [--with-scale] [--report]
#
#   mặc định        đánh giá lại các cấu hình đã chốt trong configs/ (không chạy lại quá trình chọn tham số)
#   --with-tuning   tái tạo cả quá trình chọn: lưới BM25 cơ bản và mở rộng, lưới BM25F, lưới ứng viên của V5 (~1 giờ)
#   --with-scale    thí nghiệm quy mô, tải và Lucene (~4 giờ; phần Linux chạy riêng bằng docker/run_build2_linux.sh)
#   --report        sinh bảng, hình và biên dịch báo cáo LaTeX (cần Tectonic hoặc xelatex)
#
# Môi trường: requirements/main.lock.txt (.venv, Python 3.13) và requirements/spark.lock.txt (.venv-spark, Python 3.11).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python; [ -x "$PY" ] || PY=./.venv/bin/python
TUNING=0; SCALE=0; REPORT=0
for a in "$@"; do case $a in --with-tuning) TUNING=1;; --with-scale) SCALE=1;; --report) REPORT=1;; esac; done

echo "== 1. Dữ liệu"
$PY scripts/download_data.py
$PY -m extract.leandojo
$PY scripts/make_subsets.py
$PY scripts/make_holdout.py
$PY scripts/build_dependency_stats.py
$PY scripts/export_n4_shared.py   # gói benchmark chung N3/N4, dừng nếu SHA-256 khác bản ghim

if [ $TUNING = 1 ]; then
  echo "== 2. Tinh tham số trên val (chỉ validation, không chạm tập kiểm tra)"
  for c in b0_bm25_default v1_bm25_lean; do
    $PY scripts/tune_bm25.py --index-config configs/$c.yaml --grid configs/grid_bm25.yaml
    $PY scripts/tune_bm25.py --index-config configs/$c.yaml --grid configs/grid_bm25_ext.yaml   # điểm tốt nhất ở biên lưới
  done
  $PY scripts/tune_bm25.py --index-config configs/v2_bm25f_lean.yaml --grid configs/grid_bm25f.yaml
fi

echo "== 3. Huấn luyện reranker (train nửa B)"
for c in v3_rerank_struct v4_rerank_dep; do
  for s in random novel_premises; do $PY scripts/train_reranker.py --config configs/$c.yaml --strategy $s; done
done
if [ $TUNING = 1 ]; then $PY scripts/tune_v5.py; fi   # chọn số ứng viên bổ sung của V5 trên validation
for s in random novel_premises; do $PY scripts/train_reranker.py --config configs/v5_rerank_expand.yaml --strategy $s; done

echo "== 4. Đánh giá trên val và test"
for c in b0_bm25_default b0t_bm25_tuned v1_bm25_lean v2_bm25f_lean v2a_bm25f_accessible v3_rerank_struct v4_rerank_dep v5_rerank_expand; do
  for e in random_val_all random_test_all novel_premises_test_all; do
    $PY -m eval.run_eval --config configs/$c.yaml --eval $e
  done
done
$PY scripts/review_experiments.py        # đối chứng cùng ngân sách ứng viên, ablation phụ thuộc và namespace, holdout mới
$PY scripts/pin_runs.py                  # ghim các lần chạy dùng cho báo cáo
$PY scripts/summarize_quality.py > /dev/null
$PY scripts/make_check_queries.py > /dev/null
$PY scripts/make_data_dictionary.py > /dev/null   # cần index V2 đến V4 đã build ở bước 4
$PY scripts/profile_latency.py > /dev/null
$PY scripts/eval_shared.py         # B0, V2, V4, V5 trên benchmark chung → results/figures/shared_benchmark.md

echo "== 5. Ablation, phân tích lỗi và độ bền"
$PY scripts/ablation.py
$PY scripts/ablation_significance.py > /dev/null
$PY scripts/bench_codec.py
$PY scripts/error_analysis.py > /dev/null
$PY scripts/robustness.py > /dev/null
$PY scripts/candidate_analysis.py > /dev/null

if [ $SCALE = 1 ]; then
  echo "== 6. Thí nghiệm quy mô (cần .venv-spark với Python 3.11)"
  $PY scripts/run_scale.py --part build --reps 3
  $PY scripts/run_scale.py --part build2 --reps 3 --sizes 10k,full,x5,x20   # phần Linux: docker/run_build2_linux.sh
  $PY scripts/run_scale.py --part quality
  $PY scripts/run_scale.py --part serve --reps 3
  $PY scripts/run_scale.py --part serve2 --reps 3
  $PY scripts/run_scale.py --part serve_v5_k240 --reps 3
  $PY scripts/serve_cpu_check.py
  $PY scripts/lucene_baseline.py   # cần Java 21 và tệp jar của Anserini 2.3.0 (xem README)
fi
$PY scripts/make_figures.py > /dev/null

if [ $REPORT = 1 ]; then
  echo "== 7. Báo cáo"
  # các lần chạy vừa tái tạo đã được ghim lại ở bước 4, nên cập nhật báo cáo thay vì đối chiếu với bản đã phát hành
  bash report/latex/build.sh --update   # sinh bảng, ví dụ xuyên suốt, hình PDF và biên dịch report/report.pdf
fi
echo "Xong. Xem results/figures/ và report/report.pdf"
