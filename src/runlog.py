"""Ghi run record theo Phụ lục B của hướng dẫn: mỗi lần chạy một file JSON trong results/raw/."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import uuid
from importlib import metadata
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "uncommitted"


def code_hash() -> str:
    """Băm toàn bộ src/: vẫn định danh được phiên bản mã khi chưa commit."""
    h = hashlib.sha1()
    for p in sorted((ROOT / "src").rglob("*.py")):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def file_sha256(path: str | Path) -> str:
    """SHA-256 dùng để ghim đầu vào của báo cáo. Tệp văn bản được băm ở dạng xuống dòng LF vì Git lưu LF và đổi xuống dòng
    khi checkout theo core.autocrlf; tệp nhị phân (có byte NUL trong 8000 byte đầu, cùng cách Git nhận biết) băm nguyên
    byte. Nhờ vậy giá trị như nhau trên Windows, Linux và với mọi cấu hình Git."""
    data = Path(path).read_bytes()
    if b"\0" not in data[:8000]:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def machine_info() -> dict:
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "hostname_hash": hashlib.sha1(platform.node().encode()).hexdigest()[:8],
    }


def software_info(extra: tuple[str, ...] = ()) -> dict:
    out = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "pyarrow", "pandas", "pyspark", "fastapi", *extra):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def data_version() -> str:
    manifest = ROOT / "data" / "raw" / "MANIFEST.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        return f"leandojo_benchmark_4@{m.get('version')}#md5:{m['archive']['md5'][:8]}"
    return "unknown"


def new_run_id(kind: str) -> str:
    return f"{kind}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def write_record(kind: str, parameters: dict, metrics: dict, *, status: str = "ok", seed: int | None = None,
                 timeout_s: float | None = None, log_path: str | None = None, extra: dict | None = None,
                 run_id: str | None = None, out_dir: Path | None = None) -> Path:
    run_id = run_id or new_run_id(kind)
    record = {
        "run_id": run_id,
        "kind": kind,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "code_hash": code_hash(),
        "data_version": data_version(),
        "machine": machine_info(),
        "software": software_info(),
        "parameters": parameters,
        "seed": seed,
        "timeout_s": timeout_s,
        "metrics": metrics,
        "status": status,
        "log": log_path,
    }
    if extra:
        record.update(extra)
    out_dir = out_dir or ROOT / "results" / "raw" / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
