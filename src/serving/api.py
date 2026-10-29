"""Search service (vai trò Serving/performance).

Chạy:  uvicorn serving.api:app --port 8000 --workers 4
       N3_CONFIGS=configs/b0_bm25_default.yaml,configs/v2_bm25f_lean.yaml,configs/v4_rerank_dep.yaml
       N3_SUBSET=full   (hoặc 10k / 100k cho thí nghiệm quy mô)

Endpoint:
  GET  /health                       trạng thái, phiên bản index đã nạp
  POST /search        {query, k, mode, theorem_full_name?, file_path?, theorem_start_line?}
  POST /batch_search  {queries: [...], k, mode}
Mỗi phản hồi kèm thời gian xử lý phía server (server_ms) để tách độ trễ mạng khỏi độ trễ tìm kiếm.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from eval.run_eval import PROCESSED, ensure_index, load_config, make_searcher
from retrieval.search import InvertedIndex

DEFAULT_CONFIGS = "configs/b0_bm25_default.yaml,configs/v2_bm25f_lean.yaml,configs/v4_rerank_dep.yaml"
STATE: dict = {}


class SearchRequest(BaseModel):
    query: str = Field(..., description="proof state, ví dụ 'n m : ℕ\\n⊢ n + m = m + n'")
    k: int = Field(10, ge=1, le=100)
    mode: str = "V4"
    theorem_full_name: str = ""
    file_path: str = ""
    theorem_start_line: int = 10**9


class BatchRequest(BaseModel):
    queries: list[SearchRequest]


def _load():
    root = Path(__file__).resolve().parents[2]
    subset = os.environ.get("N3_SUBSET", "full")
    searchers, versions = {}, {}
    for path in os.environ.get("N3_CONFIGS", DEFAULT_CONFIGS).split(","):
        cfg = load_config(root / path)
        idx_path = ensure_index(cfg["index"], subset)
        idx = InvertedIndex(idx_path)
        searchers[cfg["name"]] = make_searcher(cfg, idx)
        versions[cfg["name"]] = {"index": idx_path.name, "n_docs": idx.n_docs,
                                 "tokenizer_version": idx.manifest["tokenizer_version"],
                                 "checksum_postings": idx.manifest["checksums"].get("postings.bin", "")[:12]}
    prem = pq.read_table(PROCESSED / "premises.parquet", columns=["full_name", "statement", "path"]).to_pandas()
    STATE.update(searchers=searchers, versions=versions, subset=subset, prem=prem)
    for s in searchers.values():  # warm-up: nạp cache và các bảng phụ trước khi nhận tải
        s.search("⊢ a + b = b + a", k=10, ctx={})


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load()
    yield


app = FastAPI(title="Nhóm 3 Lean premise search", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "subset": STATE["subset"], "modes": STATE["versions"], "pid": os.getpid()}


def _search(req: SearchRequest) -> dict:
    s = STATE["searchers"].get(req.mode)
    if s is None:
        raise HTTPException(400, f"mode '{req.mode}' không có; chọn một trong {sorted(STATE['searchers'])}")
    ctx = {"theorem_full_name": req.theorem_full_name, "file_path": req.file_path,
           "theorem_start_line": req.theorem_start_line}
    t = time.perf_counter()
    hits = s.search(req.query, k=req.k, ctx=ctx)
    ms = (time.perf_counter() - t) * 1000
    prem = STATE["prem"]
    return {"mode": req.mode, "server_ms": round(ms, 3), "results": [
        {"rank": i, "premise_id": pid, "full_name": prem.full_name.iat[pid], "score": round(score, 4),
         "statement": prem.statement.iat[pid][:300], "path": prem.path.iat[pid]}
        for i, (pid, score) in enumerate(hits, 1)]}


# Hàm đồng bộ: FastAPI chạy trong threadpool. Tính toán chủ yếu là numpy nên vẫn bị GIL giới hạn
# trong một tiến trình; tăng thông lượng bằng --workers (nhiều tiến trình).
@app.post("/search")
def search(req: SearchRequest):
    return _search(req)


@app.post("/batch_search")
def batch_search(req: BatchRequest):
    t = time.perf_counter()
    res = [_search(q) for q in req.queries]
    return {"server_ms": round((time.perf_counter() - t) * 1000, 3), "responses": res}
