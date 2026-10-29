"""Chẩn đoán vì sao thêm worker không tăng thông lượng (bổ sung cho run_scale.py --part serve2).

Với mỗi số worker: khởi động API trên index đầy đủ, chạy tải B0 ở 48 client đồng thời bằng một tiến trình client và bằng
bốn tiến trình client (12 client mỗi tiến trình), đồng thời lấy mẫu CPU của từng tiến trình worker trong suốt thời gian
tải. Nếu nhiều tiến trình client không tăng QPS thì client không phải nút thắt; nếu tổng CPU của worker gần số luồng
logic của máy thì dịch vụ bị giới hạn bởi CPU.

Dùng:  python scripts/serve_cpu_check.py [--workers 4,8,12]
Ra:    run record kind "serve-cpu" trong results/raw/serve-cpu/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import runlog  # noqa: E402

PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "N3_SUBSET": "full"}
URL = "http://127.0.0.1:8003"


def wait_health():
    for _ in range(300):
        try:
            urllib.request.urlopen(URL + "/health", timeout=5).read()
            return
        except OSError:
            time.sleep(2)
    raise TimeoutError("API không lên")


def load(n_proc: int, clients: int) -> tuple[float, list[float]]:
    procs = [subprocess.Popen([PY, "-m", "eval.loadtest", "--url", URL, "--mode", "B0", "--clients", str(clients),
                               "--requests", "1200", "--warmup", "20"], cwd=ROOT, env=ENV, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True) for _ in range(n_proc)]
    outs = [p.communicate()[0] for p in procs]
    lv = [json.loads([ln for ln in o.splitlines() if ln.startswith("{")][-1]) for o in outs]
    return sum(x["qps"] for x in lv), [x["e2e_ms_p95"] for x in lv]


def sample_cpu(workers: list[psutil.Process], stop: threading.Event, out: dict):
    for w in workers:
        w.cpu_percent(None)
    samples = []
    while not stop.wait(1.0):
        samples.append([w.cpu_percent(None) for w in workers])
    out["per_worker"] = [round(sum(s[i] for s in samples) / max(len(samples), 1), 1) for i in range(len(workers))]
    out["total"] = round(sum(out["per_worker"]), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="4,8,12")
    args = ap.parse_args()
    for w in [int(x) for x in args.workers.split(",")]:
        srv = subprocess.Popen([PY, "-m", "uvicorn", "serving.api:app", "--host", "127.0.0.1", "--port", "8003",
                                "--workers", str(w), "--log-level", "warning"], cwd=ROOT, env=ENV,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_health()
            time.sleep(5)
            workers = [c for c in psutil.Process(srv.pid).children(recursive=True) if "python" in c.name().lower()]
            res = {}
            for label, n_proc, clients in (("1x48", 1, 48), ("4x12", 4, 12)):
                stop, cpu = threading.Event(), {}
                th = threading.Thread(target=sample_cpu, args=(workers, stop, cpu))
                th.start()
                qps, p95 = load(n_proc, clients)
                stop.set()
                th.join()
                res[label] = {"qps": round(qps, 1), "p95_ms": [round(x, 1) for x in p95], "worker_cpu_pct": cpu}
            metrics = {"workers": w, "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                       **res}
            runlog.write_record("serve-cpu", parameters={"workers": w, "mode": "B0", "subset": "full"}, metrics=metrics)
            print(json.dumps({"workers": w, **{k: {"qps": v["qps"], "cpu_total": v["worker_cpu_pct"]["total"]}
                                               for k, v in res.items()}}), flush=True)
        finally:
            for c in psutil.Process(srv.pid).children(recursive=True):
                c.kill()
            srv.kill()
            time.sleep(3)


if __name__ == "__main__":
    main()
