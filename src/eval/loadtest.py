"""Load test search service với C client đồng thời (mặc định 1, 10, 50).

Mỗi client là một luồng giữ kết nối HTTP keep-alive, gửi lần lượt các truy vấn của tập đánh giá.
Đo độ trễ đầu-cuối phía client (gồm HTTP) và server_ms do service trả về.
Trước mỗi mức tải có warm-up (không tính vào kết quả).

Dùng:  python -m eval.loadtest --url http://127.0.0.1:8000 --mode V4 --clients 1,10,50 --requests 600
"""
from __future__ import annotations

import argparse
import http.client
from collections import Counter
import json
import threading
import time
from urllib.parse import urlparse

import numpy as np

import runlog
from eval.run_eval import load_eval, query_contexts


def _client(url: str, payloads: list[bytes], out: list, errors: list, barrier: threading.Barrier):
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=120)
    barrier.wait()
    for body in payloads:
        t = time.perf_counter()
        try:
            conn.request("POST", "/search", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                errors.append(resp.status)
                continue
            out.append(((time.perf_counter() - t) * 1000, json.loads(data)["server_ms"]))
        except (OSError, http.client.HTTPException) as e:
            errors.append(repr(e))
            conn.close()
            conn = http.client.HTTPConnection(u.hostname, u.port, timeout=120)
    conn.close()


def run_level(url: str, bodies: list[bytes], clients: int, n_requests: int) -> dict:
    per = max(1, n_requests // clients)
    results: list = []
    errors: list = []
    barrier = threading.Barrier(clients + 1)
    threads = []
    for c in range(clients):
        chunk = [bodies[(c * per + i) % len(bodies)] for i in range(per)]
        th = threading.Thread(target=_client, args=(url, chunk, results, errors, barrier), daemon=True)
        th.start()
        threads.append(th)
    barrier.wait()
    t0 = time.perf_counter()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    lat = np.array([r[0] for r in results])
    srv = np.array([r[1] for r in results])
    error_types = dict(Counter(str(e).split("(")[0] for e in errors))
    return {"clients": clients, "requests_ok": int(lat.size), "errors": len(errors), "error_types": error_types,
            "wall_s": round(wall, 3),
            "qps": round(lat.size / wall, 2),
            "e2e_ms_p50": float(np.percentile(lat, 50)), "e2e_ms_p95": float(np.percentile(lat, 95)),
            "e2e_ms_p99": float(np.percentile(lat, 99)), "server_ms_p50": float(np.percentile(srv, 50)),
            "server_ms_p95": float(np.percentile(srv, 95))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", default="V4")
    ap.add_argument("--eval", default="scale_random_test_500")
    ap.add_argument("--clients", default="1,10,50")
    ap.add_argument("--requests", type=int, default=600, help="tổng số request mỗi mức tải")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--server-workers", type=int, default=None, help="chỉ để ghi vào run record")
    args = ap.parse_args()
    q, _ = load_eval(args.eval)
    bodies = [json.dumps({"query": t, "k": args.k, "mode": args.mode, **c}).encode()
              for t, c in zip(q.state_before, query_contexts(q))]
    with urllib_health(args.url) as health:
        pass
    levels = []
    for c in map(int, args.clients.split(",")):
        run_level(args.url, bodies, min(c, 10), args.warmup)  # warm-up
        res = run_level(args.url, bodies, c, args.requests)
        print(json.dumps(res))
        levels.append(res)
    rec = runlog.write_record("loadtest", parameters={"url": args.url, "mode": args.mode, "eval_set": args.eval,
                                                      "requests_per_level": args.requests, "k": args.k, "tag": args.tag,
                                                      "server_workers": args.server_workers, "server": health},
                              metrics={"levels": levels}, seed=None)
    print("record:", rec)


class urllib_health:
    """Đọc /health để ghi phiên bản index vào run record."""

    def __init__(self, url: str):
        self.url = url

    def __enter__(self):
        import urllib.request
        with urllib.request.urlopen(self.url + "/health", timeout=30) as r:
            return json.load(r)

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
