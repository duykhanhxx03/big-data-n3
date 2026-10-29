"""Điều phối thí nghiệm quy mô (tuần 7). Mỗi cấu hình lặp --reps lần; mỗi lần ghi một run record.

Phần build:   builder ∈ {sequential, spark local[2|4|8]} × kích thước ∈ {10k, 100k, full, x5}
              + ablation Spark: không kết hợp cục bộ; salting hot key; codec raw32/vb/gamma
              (engine RDD đầu tiên của build_spark, giữ nguyên để tái lập số liệu cũ)
Phần build2:  engine Spark tối ưu (arrow) so với bản tuần tự, trên hai loại chỉ mục: b0 (tách từ mặc định, 1 trường)
              và lean (tách từ Lean, 3 trường, chỉ mục thật của V2 đến V4); kích thước tới x20 (3,6 triệu bản ghi).
              Chạy được cả trên Windows lẫn trong container Linux (docker/spark-linux.Dockerfile).
Phần quality: B0 / V2 / V4 × {10k, 100k, full} trên mẫu 500 truy vấn test (chất lượng tất định, chạy 1 lần)
Phần serve:   với mỗi kích thước: khởi động API (4 worker) → load test 1/10/50 client cho B0, V2, V4
Phần serve2:  index đầy đủ, số worker của API ∈ {4, 8, 12} → load test 1/10/50/100 client cho B0, V2, V4; ghi RAM của server

Dùng:  python scripts/run_scale.py --part build --reps 3
       python scripts/run_scale.py --part quality
       python scripts/run_scale.py --part serve --reps 3
       python scripts/run_scale.py --part build2 --reps 3 [--sizes full,x5,x20] [--indexes lean] [--masters 4,8]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import runlog  # noqa: E402

PY = os.environ.get("N3_PY") or (str(ROOT / ".venv" / "Scripts" / "python.exe") if os.name == "nt"
                                  else str(ROOT / ".venv" / "bin" / "python"))
PY_SPARK = os.environ.get("N3_PY_SPARK") or (str(ROOT / ".venv-spark" / "Scripts" / "python.exe") if os.name == "nt"
                                             else str(ROOT / ".venv-spark" / "bin" / "python"))
INDEX_ARGS = {"b0": [], "lean": ["--fields", "name,namespace,statement", "--tokenizer", "lean"]}
SIZES = ["10k", "100k", "full", "x5"]
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run_tracked(cmd: list[str], log: Path, timeout: float | None = None) -> tuple[int, float, float]:
    """Chạy tiến trình con, lấy mẫu tổng RSS của cả cây tiến trình. Trả (mã thoát, thời gian, RSS đỉnh MB).
    Quá timeout thì dừng cả cây tiến trình và trả mã -9."""
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=ENV)
        proc = psutil.Process(p.pid)
        peak = 0
        while p.poll() is None:
            total = 0
            try:
                for c in [proc, *proc.children(recursive=True)]:
                    total += c.memory_info().rss
            except psutil.Error:
                pass
            peak = max(peak, total)
            if timeout and time.perf_counter() - t0 > timeout:
                for c in proc.children(recursive=True):
                    c.kill()
                p.kill()
                p.wait()
                return -9, time.perf_counter() - t0, peak / 2**20
            time.sleep(0.1)
    return p.returncode, time.perf_counter() - t0, peak / 2**20


def size_args(size: str) -> list[str]:
    if size in ("x5", "x20"):
        return ["--premises", f"data/processed/synthetic/premises_{size}.parquet"]
    if size == "full":
        return []
    return ["--subset", f"data/processed/subsets/premises_{size}.npy"]


def build_one(builder: str, size: str, rep: int, extra: list[str], label: str, *, experiment: str = "build",
              timeout: float | None = None):
    out = ROOT / "indexes" / "_scale" / f"{label}_{size}_r{rep}"
    if out.exists():
        shutil.rmtree(out)
    logs = ROOT / "results" / "logs" / "scale"
    logs.mkdir(parents=True, exist_ok=True)
    if builder == "sequential":
        cmd = [PY, "-m", "index.build_seq", "--out", str(out), *size_args(size), *extra]
    else:
        cmd = [PY_SPARK, "-m", "index.build_spark", "--out", str(out), "--master", builder, *size_args(size), *extra]
    code, wall, peak = run_tracked(cmd, logs / f"{label}_{size}_r{rep}.log", timeout)
    metrics = {"wall_s": round(wall, 2), "peak_rss_mb": round(peak, 1), "exit_code": code}
    status = "ok" if code == 0 else ("timeout" if code == -9 else "failed")
    if code == 0:
        m = json.loads((out / "index_manifest.json").read_text(encoding="utf-8"))
        metrics.update({k: m[k] for k in ("n_docs", "n_terms", "n_postings", "max_df", "postings_bytes")})
        metrics.update({k: v for k, v in m["build"].items() if k != "stages"})
        metrics["index_bytes"] = sum(p.stat().st_size for p in out.glob("*") if p.is_file())
        metrics["checksums"] = m.get("checksums")
        stages = m["build"].get("stages")
    else:
        stages = None
    runlog.write_record("build", parameters={"builder": builder, "size": size, "rep": rep, "label": label, "args": extra,
                                             "experiment": experiment, "platform": platform.system()},
                        metrics=metrics, status=status, log_path=f"results/logs/scale/{label}_{size}_r{rep}.log",
                        extra={"spark_stages": stages})
    shutil.rmtree(out, ignore_errors=True)
    print(f"[build] {label} {size} r{rep}: {status} {metrics.get('wall_s')}s peak {metrics.get('peak_rss_mb')}MB", flush=True)


def part_build(reps: int):
    if not (ROOT / "data/processed/synthetic/premises_x5.parquet").exists():
        subprocess.run([PY, "scripts/make_synthetic.py", "--factor", "5"], cwd=ROOT, check=True, env=ENV)
    for rep in range(1, reps + 1):
        for size in SIZES:
            build_one("sequential", size, rep, [], "seq")
            for master in ("local[2]", "local[4]", "local[8]"):
                build_one(master, size, rep, ["--engine", "rdd"], f"spark{master[6:-1]}")
        for size in ("full", "x5"):  # ablation Spark ở local[4]
            build_one("local[4]", size, rep, ["--engine", "rdd", "--no-local-combine"], "spark4-nocombine")
            build_one("local[4]", size, rep, ["--engine", "rdd", "--salt", "4", "--hot-df", "20000"], "spark4-salt4")
        for codec in ("raw32", "gamma"):  # vb đã có ở các lần chạy chính
            build_one("sequential", "full", rep, ["--codec", codec], f"seq-{codec}")


def part_build2(reps: int, sizes: list[str], indexes: list[str], masters: list[int], rdd_x20: bool):
    """Engine Spark tối ưu so với bản tuần tự. Thứ tự xen kẽ builder trong cùng kích thước để giảm ảnh hưởng của trôi máy."""
    for factor in ("x5", "x20"):
        if factor in sizes and not (ROOT / f"data/processed/synthetic/premises_{factor}.parquet").exists():
            subprocess.run([PY, "scripts/make_synthetic.py", "--factor", factor[1:]], cwd=ROOT, check=True, env=ENV)
    plat = "linux" if platform.system() == "Linux" else "win"
    for rep in range(1, reps + 1):
        for index in indexes:
            for size in sizes:
                build_one("sequential", size, rep, INDEX_ARGS[index], f"{plat}-{index}-seq", experiment="build2")
                for n in masters:
                    build_one(f"local[{n}]", size, rep, ["--engine", "arrow", *INDEX_ARGS[index]],
                              f"{plat}-{index}-arrow{n}", experiment="build2")
        if rdd_x20 and rep == 1:   # engine RDD cũ ở x20: driver phải giữ toàn bộ 3,6 triệu bản ghi
            build_one("local[4]", "x20", rep, ["--engine", "rdd"], f"{plat}-b0-rdd4", experiment="build2", timeout=1800)


def part_quality():
    for cfg in ("b0_bm25_default", "v2_bm25f_lean", "v4_rerank_dep"):
        for size in ("10k", "100k", "full"):
            subprocess.run([PY, "-m", "eval.run_eval", "--config", f"configs/{cfg}.yaml", "--eval",
                            "scale_random_test_500", "--subset", size, "--tag", "scale-quality"], cwd=ROOT, check=True, env=ENV)


def wait_health(url: str, timeout: float = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                return json.load(r)
        except OSError:
            time.sleep(2)
    raise TimeoutError("API không lên")


def part_serve(reps: int, workers: int):
    for size in ("10k", "100k", "full"):
        env = {**ENV, "N3_SUBSET": size}
        log = (ROOT / "results" / "logs" / "scale" / f"api_{size}.log").open("w", encoding="utf-8")
        srv = subprocess.Popen([PY, "-m", "uvicorn", "serving.api:app", "--host", "127.0.0.1", "--port", "8001",
                                "--workers", str(workers), "--log-level", "warning"], cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT)
        try:
            wait_health("http://127.0.0.1:8001")
            for rep in range(1, reps + 1):
                for mode in ("B0", "V2", "V4"):
                    subprocess.run([PY, "-m", "eval.loadtest", "--url", "http://127.0.0.1:8001", "--mode", mode,
                                    "--clients", "1,10,50", "--requests", "400", "--server-workers", str(workers),
                                    "--tag", f"scale-{size}-r{rep}"], cwd=ROOT, check=True, env=ENV)
        finally:
            for c in psutil.Process(srv.pid).children(recursive=True):
                c.kill()
            srv.kill()
            log.close()
            time.sleep(3)


def server_rss_mb(pid: int) -> float:
    total = 0
    for c in [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]:
        try:
            total += c.memory_info().rss
        except psutil.Error:
            pass
    return total / 2**20


def part_serve_v5(reps: int, clients: str, with_k240: bool = False):
    """V4 so với V5 dưới tải: cùng một server 4 worker nạp các cấu hình, xen kẽ thứ tự mode giữa các lần lặp.
    with_k240: thêm V4-K240 (cùng ngân sách ứng viên với V5), ghi tag v5cmp2 để tách khỏi đợt đo đầu."""
    configs = "configs/v4_rerank_dep.yaml,configs/v5_rerank_expand.yaml"
    modes = ["V4", "V5"]
    tag = "v5cmp"
    if with_k240:
        configs += ",configs/review/v4k240.yaml"
        modes, tag = ["V4", "V4-K240", "V5"], "v5cmp2"
    for rep in range(1, reps + 1):
        log = (ROOT / "results" / "logs" / "scale" / f"api_v5cmp_r{rep}.log").open("w", encoding="utf-8")
        srv = subprocess.Popen([PY, "-m", "uvicorn", "serving.api:app", "--host", "127.0.0.1", "--port", "8004",
                                "--workers", "4", "--log-level", "warning"], cwd=ROOT,
                               env={**ENV, "N3_SUBSET": "full", "N3_CONFIGS": configs}, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_health("http://127.0.0.1:8004")
            shift = (rep - 1) % len(modes)
            for mode in modes[shift:] + modes[:shift]:
                subprocess.run([PY, "-m", "eval.loadtest", "--url", "http://127.0.0.1:8004", "--mode", mode,
                                "--clients", clients, "--requests", "400", "--server-workers", "4",
                                "--tag", f"{tag}-r{rep}"], cwd=ROOT, check=True, env=ENV)
        finally:
            for c in psutil.Process(srv.pid).children(recursive=True):
                c.kill()
            srv.kill()
            log.close()
            time.sleep(3)


def part_serve2(reps: int, workers_list: list[int], clients: str):
    """Mở rộng dịch vụ theo số tiến trình worker. Mỗi (lần lặp, số worker) khởi động server mới."""
    for rep in range(1, reps + 1):
        for w in workers_list:
            log = (ROOT / "results" / "logs" / "scale" / f"api_workers{w}_r{rep}.log").open("w", encoding="utf-8")
            srv = subprocess.Popen([PY, "-m", "uvicorn", "serving.api:app", "--host", "127.0.0.1", "--port", "8002",
                                    "--workers", str(w), "--log-level", "warning"], cwd=ROOT,
                                   env={**ENV, "N3_SUBSET": "full"}, stdout=log, stderr=subprocess.STDOUT)
            try:
                health = wait_health("http://127.0.0.1:8002")
                time.sleep(5)   # để mọi worker nạp xong rồi mới đo RAM
                rss = server_rss_mb(srv.pid)
                runlog.write_record("serve-memory", parameters={"workers": w, "rep": rep, "subset": "full"},
                                    metrics={"server_rss_mb": round(rss, 1)}, extra={"health": health})
                for mode in ("B0", "V2", "V4"):
                    subprocess.run([PY, "-m", "eval.loadtest", "--url", "http://127.0.0.1:8002", "--mode", mode,
                                    "--clients", clients, "--requests", "400", "--server-workers", str(w),
                                    "--tag", f"workers-{w}-r{rep}"], cwd=ROOT, check=True, env=ENV)
                print(f"[serve2] workers={w} r{rep}: RAM server {rss:,.0f} MB", flush=True)
            finally:
                for c in psutil.Process(srv.pid).children(recursive=True):
                    c.kill()
                srv.kill()
                log.close()
                time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["build", "build2", "quality", "serve", "serve2", "serve_v5", "serve_v5_k240"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sizes", default="10k,100k,full,x5,x20", help="chỉ cho build2")
    ap.add_argument("--indexes", default="b0,lean", help="chỉ cho build2")
    ap.add_argument("--masters", default="2,4,8", help="chỉ cho build2: số luồng của Spark local[n]")
    ap.add_argument("--no-rdd-x20", action="store_true", help="build2: bỏ lần chạy engine RDD cũ ở x20")
    ap.add_argument("--server-workers", default="4,8,12", help="chỉ cho serve2")
    ap.add_argument("--clients", default="1,10,50,100", help="chỉ cho serve2")
    args = ap.parse_args()
    {"build": lambda: part_build(args.reps), "quality": part_quality,
     "serve": lambda: part_serve(args.reps, args.workers),
     "serve_v5": lambda: part_serve_v5(args.reps, "1,10,50"),
     "serve_v5_k240": lambda: part_serve_v5(args.reps, "1,10,50", with_k240=True),
     "serve2": lambda: part_serve2(args.reps, [int(x) for x in args.server_workers.split(",")], args.clients),
     "build2": lambda: part_build2(args.reps, args.sizes.split(","), args.indexes.split(","),
                                   [int(x) for x in args.masters.split(",")], not args.no_rdd_x20)}[args.part]()


if __name__ == "__main__":
    main()
