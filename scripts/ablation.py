"""Ablation tuần 8 cho V4 (huấn luyện lại reranker cho từng biến thể, đánh giá trên random_test_all).

1. Bỏ từng nhóm đặc trưng (leave-one-group-out)
2. Số ứng viên stage 1: K ∈ {50, 100, 200}
3. Bộ lọc premise truy cập được: bật / tắt (giữ nguyên đặc trưng)
4. Codec posting (B0, mẫu 500 truy vấn, chạy tuần tự để đo latency): raw32 / vb / gamma

Kết quả: results/raw/eval/*.json (tag ablation-*) và bảng results/figures/ablation.md
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
BASE = yaml.safe_load((ROOT / "configs" / "v4_rerank_dep.yaml").read_text(encoding="utf-8"))
GROUPS = {
    "bm25": ["bm25_rel", "bm25_rr"],
    "namespace": ["ns_in_query", "ns_thm"],
    "location": ["same_file", "dir_overlap"],
    "symbols": ["sym_jaccard"],
    "kind": ["kind_thm", "kind_def", "kind_inst", "kind_type"],
    "dependency": ["popularity", "pagerank"],
}


def variant(name: str, features=None, candidates=None, accessible=None) -> Path:
    cfg = copy.deepcopy(BASE)
    cfg["name"] = name
    if features is not None:
        cfg["rerank"]["features"] = features
    if candidates is not None:
        cfg["rerank"]["candidates"] = candidates
    if accessible is not None:
        cfg["rerank"]["accessible_filter"] = accessible
    cfg["rerank"]["model"] = f"results/models/ablation/{name}.json"
    path = ROOT / "configs" / "ablation" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Sinh bởi scripts/ablation.py\n" + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    return path


def train_and_eval(path: Path):
    subprocess.run([PY, "scripts/train_reranker.py", "--config", str(path), "--workers", "3"], cwd=ROOT, check=True,
                   env=ENV, stdout=subprocess.DEVNULL)
    out = subprocess.run([PY, "-m", "eval.run_eval", "--config", str(path), "--eval", "random_test_all",
                          "--tag", "ablation"], cwd=ROOT, check=True, env=ENV, capture_output=True, text=True)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    # stdout đã làm tròn 4 chữ số; lấy metric gốc trong run record để tránh làm tròn hai lần.
    res.update(json.loads(Path(res["record"]).read_text(encoding="utf-8"))["metrics"])
    print(path.stem, {k: res[k] for k in ("R@1", "R@10", "R@50", "MRR")}, flush=True)
    return path.stem, res


def main() -> None:
    feats = BASE["rerank"]["features"]
    paths = [variant("abl_full", feats)]
    for g, fs in GROUPS.items():
        paths.append(variant(f"abl_no_{g}", [f for f in feats if f not in fs]))
    for k in (50, 200):
        paths.append(variant(f"abl_k{k}", feats, candidates=k))
    paths.append(variant("abl_no_filter", feats, accessible=False))
    with ThreadPoolExecutor(3) as ex:
        results = dict(ex.map(train_and_eval, paths))

    codec_rows = []
    for codec in ("raw32", "vb", "gamma"):
        cfg = yaml.safe_load((ROOT / "configs" / "b0_bm25_default.yaml").read_text(encoding="utf-8"))
        cfg["name"] = f"B0-{codec}"
        cfg["index"]["codec"] = codec
        p = ROOT / "configs" / "ablation" / f"b0_{codec}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        out = subprocess.run([PY, "-m", "eval.run_eval", "--config", str(p), "--eval", "scale_random_test_500",
                              "--tag", "ablation-codec"], cwd=ROOT, check=True, env=ENV, capture_output=True, text=True)
        res = json.loads(out.stdout.strip().splitlines()[-1])
        rec = json.loads(Path(res["record"]).read_text(encoding="utf-8"))
        res.update(rec["metrics"])
        codec_rows.append((codec, rec["index"]["postings_bytes"], res["R@10"], res["latency_ms_p50"], res["latency_ms_p95"]))

    full = results["abl_full"]
    md = ["# Ablation (sinh bởi scripts/ablation.py; random_test_all, n = 2641)\n",
          "| Biến thể | R@1 | R@10 | R@50 | MRR | ΔR@10 so với đầy đủ |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        md.append(f"| {name} | {r['R@1']:.3f} | {r['R@10']:.3f} | {r['R@50']:.3f} | {r['MRR']:.3f} | {r['R@10'] - full['R@10']:+.3f} |")
    md += ["\n## Codec posting (B0, mẫu 500 truy vấn, một luồng)\n",
           "| Codec | Kích thước postings (byte) | R@10 | P50 (ms) | P95 (ms) |", "|---|---|---|---|---|"]
    for c, b, r10, p50, p95 in codec_rows:
        md.append(f"| {c} | {b:,} | {r10:.3f} | {p50:.1f} | {p95:.1f} |")
    (ROOT / "results" / "figures" / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
