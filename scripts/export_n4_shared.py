"""Dựng lại gói benchmark chung N3/N4 (n4_shared) từ LeanDojo Benchmark 4 gốc và kiểm tra SHA-256 đã ghim.

Gói gồm corpus.jsonl (180.907 premise, định danh full_name) và queries.jsonl (898 theorem của random/test), đúng schema
đã thống nhất với Nhóm 4 (docs/n4_shared_benchmark.md). Script tạo lại đúng từng byte bản đã gửi Nhóm 4, nên không cần
giữ tệp zip: nếu SHA-256 khác giá trị ghim trong src/shared.py thì script dừng và không ghi đè gói cũ.

Dùng:  python scripts/export_n4_shared.py [--src data/raw/leandojo_benchmark_4_v10/leandojo_benchmark_4]
                                          [--out data/n4_shared] [--zip]
Ra:    data/n4_shared/{corpus.jsonl, queries.jsonl, checksums.txt, eligible_rule.txt}, thêm data/n4_shared.zip nếu --zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared import EXPECTED_SHA256, build_records, verify_directory, write_bundle  # noqa: E402

ELIGIBLE_RULE = """QUY TẮC ELIGIBLE PREMISE (N3 xác nhận, khớp đề nghị của N4):
1. Theorem truy vấn KHÔNG được xuất hiện trong kết quả truy hồi của chính nó: loại nó trước khi cắt top_k
   (shared.eligible; scripts/eval_shared.py áp dụng cho mọi hệ thống, kể cả run file của Nhóm 4).
2. Premise pool = toàn bộ corpus.jsonl trừ chính theorem truy vấn (đóng băng trước mọi run, pool_id: shared-frozen-pool).
3. Theorem test KHÁC vẫn là premise hợp lệ và được giữ nguyên trong relevant_ids.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/leandojo_benchmark_4_v10/leandojo_benchmark_4")
    ap.add_argument("--out", default="data/n4_shared")
    ap.add_argument("--zip", action="store_true", help="đóng gói thêm <out>.zip để gửi nhóm khác")
    args = ap.parse_args()
    out = ROOT / args.out
    corpus, queries = build_records(ROOT / args.src)
    with tempfile.TemporaryDirectory(dir=out.parent if out.parent.exists() else None) as tmp:
        digests = write_bundle(Path(tmp), corpus, queries)
        bad = [n for n, d in digests.items() if d != EXPECTED_SHA256[n]]
        if bad:
            for n in bad:
                print(f"SHA-256 lệch {n}: {digests[n]} (ghim {EXPECTED_SHA256[n]})", file=sys.stderr)
            return 1
        (Path(tmp) / "eligible_rule.txt").write_text(ELIGIBLE_RULE, encoding="utf-8", newline="\n")
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(tmp, out)
    report = verify_directory(out)
    print(f"{out.relative_to(ROOT).as_posix()}: {report['corpus_count']} premise, {report['query_count']} truy vấn, "
          f"{report['queries_without_direct_premises']} truy vấn không có premise đúng; SHA-256 khớp bản ghim")
    if args.zip:
        archive = out.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.iterdir()):
                z.write(p, p.name)
        print(f"{archive.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
