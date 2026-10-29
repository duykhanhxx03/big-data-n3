"""Data dictionary (tuần 2): mô tả từng cột của mọi bảng do pipeline tạo ra, kèm kiểu, số dòng, giá trị rỗng,
miền giá trị và ví dụ lấy thẳng từ dữ liệu. Mô tả viết tay ở DESC; số liệu tính lại mỗi lần chạy.

Dùng:  python scripts/make_data_dictionary.py
Ra:    docs/data_dictionary.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
IDX = ROOT / "indexes" / "lean__name+namespace+statement__vb__full"

TABLES = {
    "premises": (P / "premises.parquet", "Một dòng = một khai báo trong corpus.jsonl (tập ứng viên truy hồi).", {
        "premise_id": "Khóa chính. Thứ tự của (full_name, path) sau khi sắp xếp; ổn định với cùng snapshot.",
        "full_name": "Tên đầy đủ của khai báo, gồm namespace (ví dụ `Nat.succ_le_of_lt`). Không duy nhất: 57 tên "
                     "trùng giữa Lean core và Mathlib, nên định danh là cặp (full_name, path).",
        "name": "Trường `name` của LeanDojo; trong snapshot này luôn trùng full_name.",
        "path": "File chứa khai báo, tương đối so với gốc repo mathlib4 (thư viện phụ nằm dưới `.lake/packages/`).",
        "namespace": "Phần trước thành phần cuối của full_name; rỗng nếu khai báo ở gốc.",
        "short_name": "Thành phần cuối của full_name.",
        "decl_kind": "Loại khai báo: từ khóa đầu tiên của mã nguồn sau khi bỏ docstring/attribute.",
        "statement": "Phát biểu: mã nguồn cắt tại `:=` cấp ngoài cùng, `where` hoặc nhánh `|` đầu dòng. Field chính của index.",
        "code": "Mã nguồn đầy đủ của khai báo (gồm docstring, attribute, chứng minh). Không dùng để index.",
        "library": "Thư viện gốc suy từ path: Mathlib, lean4 (core), batteries, aesop, ...",
        "start_line": "Dòng bắt đầu của khai báo trong file (tính cả docstring/attribute).",
        "end_line": "Dòng kết thúc. Dùng cho bộ lọc truy cập được: premise cùng file phải có end_line < dòng bắt đầu theorem.",
    }),
    "queries": (P / "queries.parquet", "Một dòng = một tactic có ít nhất một premise đúng trong corpus.", {
        "query_id": "Khóa chính, dạng `{split_strategy}:{split}:{theorem_idx}:{tactic_idx}`.",
        "split_strategy": "Cách chia của LeanDojo: `random` (chính) hoặc `novel_premises` (khó).",
        "split": "train / val / test.",
        "theorem_full_name": "Theorem đang được chứng minh. Được phép dùng lúc truy vấn (ngữ cảnh).",
        "file_path": "File chứa theorem. Được phép dùng lúc truy vấn.",
        "theorem_idx": "Vị trí theorem trong file JSON của split (để truy ngược dữ liệu gốc).",
        "theorem_start_line": "Dòng bắt đầu của theorem. Được phép dùng lúc truy vấn (bộ lọc truy cập được).",
        "tactic_idx": "Vị trí tactic trong `traced_tactics` của theorem.",
        "state_before": "Proof state trước tactic = **văn bản truy vấn**. Text của tactic không được lưu (chứa tên premise).",
        "n_gold": "Số premise đúng của truy vấn sau khi bỏ premise không có trong corpus.",
    }),
    "qrels": (P / "qrels.parquet", "Một dòng = một cặp (truy vấn, premise đúng). Nhãn nhị phân.", {
        "query_id": "Khóa ngoại tới queries.",
        "premise_id": "Khóa ngoại tới premises.",
    }),
    "files": (P / "files.parquet", "Một dòng = một file Lean trong corpus.", {
        "path": "Đường dẫn file, cùng quy ước với premises.path.",
        "imports": "Danh sách file được import trực tiếp. Bao đóng bắc cầu = tập file truy cập được.",
    }),
    "dead_letter": (P / "dead_letter.parquet", "Bản ghi bị loại khi trích xuất, kèm lý do. Không có bản ghi nào mất lặng lẽ.", {
        "entity": "Loại bản ghi: `premise` (trùng), `gold` (premise đúng không có trong corpus), `query` (truy vấn không còn premise đúng).",
        "key": "Khóa của bản ghi; với `gold` là `{query_id}:{full_name}`.",
        "reason": "Lý do loại.",
    }),
    "dep_stats_random": (P / "dep_stats_random.parquet", "Thống kê phụ thuộc tính từ train nửa A của split random "
                         "(`scripts/build_dependency_stats.py`); bản `_novel_premises` cùng cấu trúc, tính từ train của novel_premises.", {
        "premise_id": "Khóa ngoại tới premises (đủ 180.957 dòng).",
        "popularity": "Số theorem train-A dùng premise ít nhất một lần.",
        "pagerank": "PageRank trên đồ thị theorem→premise của train-A (damping 0,85, tổng = 1).",
    }),
}

INDEX_FILES = {
    "dictionary": (IDX / "dictionary.parquet", "Từ điển của index: một dòng = một cặp (field, term).", {
        "field": "Field của tài liệu: name / namespace / statement.",
        "term": "Term sau tokenizer.",
        "df": "Số tài liệu chứa term trong field này.",
        "cf": "Tổng số lần xuất hiện (collection frequency).",
        "offset": "Vị trí byte bắt đầu của posting list trong postings.bin.",
        "nbytes": "Số byte của posting list sau nén.",
    }),
    "terms_any": (IDX / "terms_any.parquet", "df gộp mọi field, dùng tính idf của BM25F.", {
        "term": "Term.",
        "df_any": "Số tài liệu chứa term ở ít nhất một field.",
    }),
    "doc_lengths": (IDX / "doc_lengths.parquet", "Độ dài từng field của từng tài liệu (chuẩn hóa độ dài BM25).", {
        "doc_id": "Mã tài liệu nội bộ của index (0..n_docs−1).",
        "len_name": "Số token field name (mỗi field index có một cột `len_<field>`).",
        "len_namespace": "Số token field namespace.",
        "len_statement": "Số token field statement.",
    }),
    "doc_map": (IDX / "doc_map.parquet", "Ánh xạ doc_id của index → premise_id (index con chỉ chứa một phần corpus).", {
        "doc_id": "Mã tài liệu nội bộ.",
        "premise_id": "Khóa ngoại tới premises.",
    }),
}


def fmt(v, n=70) -> str:
    if isinstance(v, (list, np.ndarray)):
        v = list(v)[:4]
    s = " ".join(str(v).split())
    s = s if len(s) <= n else s[:n] + " …"
    return "`" + s.replace("|", "\\|").replace("`", "ˋ") + "`" if s else "(rỗng)"


def describe(name: str, path: Path, intro: str, desc: dict) -> list[str]:
    t = pq.read_table(path)
    lines = [f"### `{path.relative_to(ROOT).as_posix()}`", "", intro + f" **{t.num_rows:,} dòng.**", "",
             "| Cột | Kiểu | Rỗng | Số giá trị khác nhau | Ví dụ / miền giá trị | Ý nghĩa |", "|---|---|---|---|---|---|"]
    missing = set(desc) ^ set(t.column_names)
    assert not missing, f"{name}: mô tả không khớp cột {missing}"
    for col in t.column_names:
        s = t.column(col).to_pandas()
        typ = str(t.schema.field(col).type)
        if typ.startswith("list"):
            empty = int(s.map(len).eq(0).sum())
            nuniq = "—"
            ex = fmt(next((x for x in s if len(x)), []))
        else:
            text = pd.api.types.is_string_dtype(s) or s.dtype == object
            empty = int(s.isna().sum() + (s.astype(str) == "").sum()) if text else int(s.isna().sum())
            u = s.nunique()
            nuniq = f"{u:,}"
            if text and u <= 12:
                vc = s.value_counts()
                ex = ", ".join(f"{fmt(k, 30)} {c:,}" for k, c in vc.items())
            elif not text:
                ex = f"{s.min():,} … {s.max():,}" if pd.api.types.is_integer_dtype(s) else f"{s.min():.3g} … {s.max():.3g}"
            else:
                ex = fmt(next((x for x in s if x), ""))
        lines.append(f"| `{col}` | {typ} | {empty:,} | {nuniq} | {ex} | {desc[col]} |")
    return lines + [""]


def main() -> None:
    stats = json.loads((P / "extract_stats.json").read_text(encoding="utf-8"))
    sub = json.loads((P / "subsets" / "subsets_info.json").read_text(encoding="utf-8"))
    out = ["# Data dictionary — Nhóm 3 (sinh bởi scripts/make_data_dictionary.py)", "",
           "Mọi bảng tạo bằng `python -m extract.leandojo` và các script trong `scripts/` từ LeanDojo Benchmark 4 v10 "
           "(xem `docs/data_card.md`). Định nghĩa truy vấn và premise đúng: `docs/benchmark_spec.md`. Bản mẫu nhỏ có thể "
           "đưa vào Git nằm trong `data_sample/` với cùng tên cột. Số liệu dưới đây tính lại mỗi lần chạy script.", "",
           "Quan hệ: `qrels.query_id → queries.query_id`, `qrels.premise_id → premises.premise_id`, "
           "`premises.path → files.path`, `queries.file_path → files.path`, `dep_stats_*.premise_id → premises.premise_id`, "
           "`doc_map.premise_id → premises.premise_id`.", "",
           "## 1. Bảng sau trích xuất (`data/processed/`)", ""]
    for name, (path, intro, desc) in TABLES.items():
        out += describe(name, path, intro, desc)
    out += ["### Tập đánh giá và index con", "",
            "| File | Nội dung |", "|---|---|"]
    for f in sorted((P / "eval_sets").glob("*.npy")):
        a = np.load(f, allow_pickle=False)
        out.append(f"| `data/processed/eval_sets/{f.name}` | {len(a):,} `query_id` (mảng chuỗi numpy), ví dụ {fmt(a[0])} |")
    for f in sorted((P / "subsets").glob("*.npy"), key=lambda x: int(x.stem.split("_")[1].rstrip("k"))):
        a = np.load(f)
        out.append(f"| `data/processed/subsets/{f.name}` | {len(a):,} `premise_id` của index con (lồng nhau, chứa mọi "
                   f"premise đúng của `scale_random_test_500`) |")
    out += [f"| `data/processed/subsets/subsets_info.json` | seed ({sub['seed']}), kích thước tập đánh giá và, với mỗi index "
            f"con, số premise đúng bắt buộc ({sub['subsets']['10k']['gold_in_subset']}) và tỉ lệ của chúng |",
            "| `data/processed/synthetic/premises_x5.parquet` | Corpus nhân 5 lần cho thí nghiệm build: 4 cột `premise_id` "
            "(đánh lại 0..5N−1), `name`, `namespace`, `statement`; nội dung lặp nguyên văn |",
            f"| `data/processed/extract_stats.json` | Số liệu trích xuất; ví dụ decl_kind: "
            f"{fmt(json.dumps(stats['decl_kind'], ensure_ascii=False), 120)} |", ""]
    out += ["## 2. Index (`indexes/<tokenizer>__<fields>__<codec>__<subset>/`)", "",
            "Ví dụ dưới đây là index của V2–V4 (`lean__name+namespace+statement__vb__full`). Index tạo lại được bằng "
            "`index.build_seq` hoặc `index.build_spark` (kết quả trùng từng byte).", ""]
    for name, (path, intro, desc) in INDEX_FILES.items():
        out += describe(name, path, intro, desc)
    man = json.loads((IDX / "index_manifest.json").read_text(encoding="utf-8"))
    out += ["### `postings.bin` và `index_manifest.json`", "",
            f"- `postings.bin`: các posting list nối liền, mỗi list = dãy doc_id (lưu d-gap) và dãy tf, nén bằng codec "
            f"ghi trong manifest (`raw32`, `vb` hoặc `gamma`; xem `src/index/codec.py`). Index này: "
            f"{man['n_postings']:,} posting, {man['postings_bytes']:,} byte.",
            "- `index_manifest.json`: " + ", ".join(f"`{k}`" for k in man) + ". `checksums` là sha1 của từng file; "
            "`build` ghi builder và thời gian.", "",
            "## 3. Kết quả (`results/`)", "",
            "| File | Cột / khóa | Ý nghĩa |", "|---|---|---|",
            "| `results/runs/<run_id>.tsv` | `query_id`, `premise_id`, `rank`, `score`, `system` | Top-100 mỗi truy vấn; "
            "định dạng chung với Nhóm 4 để tính metric và trộn RRF |",
            "| `results/raw/eval/<run_id>.perquery.parquet` | `query_id`, `R@1`, `R@5`, `R@10`, `R@50`, `MRR`, `nDCG@10`, "
            "`latency_ms` | Metric từng truy vấn, đầu vào của paired bootstrap |",
            "| `results/raw/**/<run_id>.json` | `run_id`, `kind`, `timestamp`, `git_commit`, `code_hash`, `data_version`, "
            "`machine`, `software`, `parameters`, `seed`, `timeout_s`, `metrics`, `status`, `log` | Run record theo Phụ lục B "
            "của hướng dẫn |",
            "| `results/models/*.json` | `type`, `features`, `mean`, `std`, `coef`, `intercept`, `C`, `train` | Trọng số "
            "reranker hồi quy logistic |", ""]
    (ROOT / "docs" / "data_dictionary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
