"""Tín hiệu cấu trúc cho reranker (vai trò Structural features).

Mọi đặc trưng chỉ dùng thông tin có sẵn lúc truy vấn: proof state, tên theorem đang chứng minh,
file và vị trí của nó. Thống kê phụ thuộc lấy từ nửa A của train (scripts/build_dependency_stats.py).

Đặc trưng cho cặp (truy vấn, premise ứng viên):
  bm25_rel      điểm stage 1 / điểm cao nhất của truy vấn
  bm25_rr       1 / hạng ở stage 1
  ns_in_query   tỉ lệ thành phần namespace của premise xuất hiện trong truy vấn
  ns_thm        độ dài tiền tố namespace chung với theorem đang chứng minh / độ dài namespace premise
  same_file     premise cùng file với theorem
  dir_overlap   số thư mục chung đầu đường dẫn / độ sâu thư mục của file theorem
  sym_jaccard   Jaccard của tập ký hiệu toán (không phải chữ) giữa goal và statement
  kind_*        loại khai báo: theorem/lemma, def/abbrev, instance, structure/class/inductive
  popularity    log(1 + số theorem train-A dùng premise) đã chuẩn hóa
  pagerank      log PageRank đã chuẩn hóa về [0, 1]
  accessible    premise truy cập được qua import hoặc khai báo xong trước theorem trong cùng file
  file_usage    log(1 + số theorem train-A cùng file, khai báo TRƯỚC theorem đang chứng minh, dùng premise), chuẩn hóa
  dir_usage     như file_usage, cộng thêm theorem train-A ở các file cùng thư mục mà file hiện tại import (bắc cầu)
                (chỉ đếm những gì người viết định lý đã có tại vị trí đó, nên không dùng thông tin "tương lai")

Ngoài BM25F, reranker có thể lấy thêm ứng viên từ ba nguồn cấu trúc (V5): premise được dùng nhiều nhất trong cùng file,
trong cùng thư mục, và premise phổ biến nhất toàn thư viện (đều tính trên train nửa A và lọc theo truy cập được).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from analysis.tokenizer import lean_tokens

ALL_FEATURES = ("bm25_rel", "bm25_rr", "ns_in_query", "ns_thm", "same_file", "dir_overlap", "sym_jaccard",
                "kind_thm", "kind_def", "kind_inst", "kind_type", "popularity", "pagerank", "accessible",
                "file_usage", "dir_usage")
DEPENDENCY_FEATURES = ("popularity", "pagerank", "accessible")
_KIND_GROUP = {"theorem": "thm", "lemma": "thm", "def": "def", "abbrev": "def", "opaque": "def", "irreducible_def": "def",
               "instance": "inst", "structure": "type", "class": "type", "inductive": "type"}


def _goal(state: str) -> str:
    i = state.rfind("⊢")
    return state[i + 1:] if i >= 0 else state


def _symbols(text: str) -> frozenset[str]:
    return frozenset(t for t in lean_tokens(text) if not any(c.isalnum() for c in t))


class PremiseMeta:
    def __init__(self, processed: str | Path, strategy: str = "random"):
        root = Path(processed)
        p = pq.read_table(root / "premises.parquet",
                          columns=["premise_id", "namespace", "path", "start_line", "end_line", "decl_kind",
                                   "statement"]).to_pandas()
        self.n = len(p)
        self.namespace = p.namespace.tolist()
        self.ns_parts = [tuple(c.lower() for c in ns.split(".")) if ns else () for ns in self.namespace]
        self.path = p.path.tolist()
        self.statement = p.statement.tolist()
        self.start_line = p.start_line.to_numpy()
        self.end_line = p.end_line.to_numpy()
        kinds = p.decl_kind.map(lambda k: _KIND_GROUP.get(k, "other")).to_numpy()
        self.kind = {g: (kinds == g).astype(np.float64) for g in ("thm", "def", "inst", "type")}
        files = pq.read_table(root / "files.parquet").to_pandas()
        self.file_index = {f: i for i, f in enumerate(files.path)}
        self.imports = [[self.file_index[x] for x in imps if x in self.file_index] for imps in files.imports]
        self.prem_file = np.array([self.file_index.get(x, -1) for x in self.path])
        dep = pq.read_table(root / f"dep_stats_{strategy}.parquet").to_pandas().sort_values("premise_id")
        pop = np.log1p(dep.popularity.to_numpy().astype(np.float64))
        self.popularity = pop / max(pop.max(), 1e-9)
        lpr = np.log(dep.pagerank.to_numpy() * self.n)
        self.pagerank = (lpr - lpr.min()) / max(lpr.max() - lpr.min(), 1e-9)
        raw_pop = dep.popularity.to_numpy()
        self.pop_order = np.argsort(-raw_pop, kind="stable")[: int((raw_pop > 0).sum())]
        # Mỗi cặp (theorem train-A, premise nó dùng) kèm file và dòng bắt đầu của theorem (build_dependency_stats.py).
        # Chỉ đếm những gì người viết định lý đã có: theorem cùng file khai báo trước, hoặc theorem ở file được import.
        self.u_file = self.u_line = self.u_prem = np.zeros(0, np.int64)
        self.u_by_file: dict[int, np.ndarray] = {}
        self.u_by_dir: dict[str, np.ndarray] = {}
        use_path = root / f"dep_usage_{strategy}.parquet"
        if use_path.exists():
            u = pq.read_table(use_path).to_pandas()
            u["file_idx"] = u.file_path.map(lambda f: self.file_index.get(f, -1))
            u = u[u.file_idx >= 0].sort_values(["file_idx", "line", "premise_id"]).reset_index(drop=True)
            self.u_file = u.file_idx.to_numpy(np.int64)
            self.u_line = u.line.to_numpy(np.int64)
            self.u_prem = u.premise_id.to_numpy(np.int64)
            self.u_by_file = {int(f): g.to_numpy() for f, g in u.groupby("file_idx").groups.items()}
            dirs = u.file_path.map(self.dir_of)
            self.u_by_dir = {d: g.to_numpy() for d, g in u.groupby(dirs).groups.items()}
        self._usage = lru_cache(maxsize=4096)(self._usage_uncached)
        self._sym_cache: dict[int, frozenset] = {}
        self._closure = lru_cache(maxsize=2048)(self._closure_uncached)

    @staticmethod
    def dir_of(path: str) -> str:
        return path.rsplit("/", 1)[0] if "/" in path else ""

    def _usage_uncached(self, file_path: str, start_line: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        fi = self.file_index.get(file_path)
        if fi is None:
            return {"file": empty, "dir": empty}
        rows = self.u_by_file.get(fi, np.zeros(0, np.int64))
        same = rows[self.u_line[rows] < start_line]
        drows = self.u_by_dir.get(self.dir_of(file_path), np.zeros(0, np.int64))
        files_ok = self._closure(fi)
        other = drows[(self.u_file[drows] != fi) & files_ok[self.u_file[drows]]]
        out = {}
        for scope, r in (("file", same), ("dir", np.concatenate([same, other]))):
            if r.size == 0:
                out[scope] = empty
                continue
            ids, cnt = np.unique(self.u_prem[r], return_counts=True)
            order = np.lexsort((ids, -cnt))
            out[scope] = (ids[order], cnt[order])
        return out

    def usage_of(self, scope: str, file_path: str, start_line: int = 10**9) -> tuple[np.ndarray, np.ndarray]:
        """(premise_id, số theorem dùng) giảm dần theo số lần dùng, chỉ tính các theorem train-A 'có trước'."""
        return self._usage(file_path, int(start_line))[scope]

    def extra_candidates(self, file_path: str, start_line: int, access: np.ndarray | None, exclude: set[int],
                         n_file: int = 0, n_dir: int = 0, n_pop: int = 0) -> list[int]:
        """Ứng viên bổ sung theo thứ tự: dùng nhiều trong cùng file, cùng thư mục, phổ biến toàn thư viện.
        Chỉ lấy premise truy cập được (nếu có mặt nạ) và chưa có trong danh sách."""
        out: list[int] = []
        seen = set(exclude)
        sources = [(self.usage_of("file", file_path, start_line)[0], n_file),
                   (self.usage_of("dir", file_path, start_line)[0], n_dir), (self.pop_order, n_pop)]
        for ranked, n in sources:
            if n <= 0 or ranked.size == 0:
                continue
            ok = ranked if access is None else ranked[access[ranked]]
            taken = 0
            for p in ok:
                p = int(p)
                if p in seen:
                    continue
                out.append(p)
                seen.add(p)
                taken += 1
                if taken >= n:
                    break
        return out

    def _closure_uncached(self, file_idx: int) -> np.ndarray:
        seen = np.zeros(len(self.imports), dtype=bool)
        stack = list(self.imports[file_idx])
        while stack:
            f = stack.pop()
            if not seen[f]:
                seen[f] = True
                stack.extend(self.imports[f])
        return seen

    def accessible_mask(self, file_path: str, theorem_start_line: int) -> np.ndarray:
        """Mảng bool theo premise_id. File không có trong corpus → cho phép tất cả.
        Premise cùng file phải KẾT THÚC trước dòng bắt đầu của theorem: so dòng bắt đầu là chưa đủ, vì vị trí của
        premise tính cả docstring/attribute nên chính theorem đang chứng minh có thể lọt qua."""
        fi = self.file_index.get(file_path)
        if fi is None:
            return np.ones(self.n, dtype=bool)
        files_ok = self._closure(fi)
        mask = np.zeros(self.n, dtype=bool)
        valid = self.prem_file >= 0
        mask[valid] = files_ok[self.prem_file[valid]]
        mask |= (self.prem_file == fi) & (self.end_line < theorem_start_line)
        return mask

    def symbols(self, pid: int) -> frozenset[str]:
        s = self._sym_cache.get(pid)
        if s is None:
            s = self._sym_cache[pid] = _symbols(self.statement[pid])
        return s


def build_features(meta: PremiseMeta, cand: np.ndarray, stage1: np.ndarray, state: str, ctx: dict,
                   names: tuple[str, ...] = ALL_FEATURES, access: np.ndarray | None = None,
                   n_stage1: int | None = None) -> np.ndarray:
    """cand: premise_id, n_stage1 phần tử đầu theo thứ tự stage 1, phần còn lại là ứng viên bổ sung (V5);
    stage1: điểm BM25F tương ứng (ứng viên bổ sung cũng được tra điểm). Trả ma trận (len(cand), len(names))."""
    k = len(cand)
    n1 = k if n_stage1 is None else n_stage1
    cols: dict[str, np.ndarray] = {}
    if "bm25_rel" in names:
        cols["bm25_rel"] = stage1 / max(stage1.max(), 1e-9) if k else stage1
    if "bm25_rr" in names:   # ứng viên bổ sung nằm ngoài top-K của stage 1 nên nhận 0
        cols["bm25_rr"] = np.concatenate([1.0 / np.arange(1, n1 + 1), np.zeros(k - n1)])
    for scope in ("file", "dir"):
        if f"{scope}_usage" in names:
            ids, cnt = meta.usage_of(scope, ctx.get("file_path", ""), ctx.get("theorem_start_line", 10**9))
            lookup = dict(zip(ids.tolist(), cnt.tolist()))
            top = np.log1p(cnt.max()) if cnt.size else 1.0
            cols[f"{scope}_usage"] = np.array([np.log1p(lookup.get(int(p), 0)) / top for p in cand]) if k else np.zeros(0)
    qtok = set(lean_tokens(state))
    thm_ns = tuple(c.lower() for c in ctx.get("theorem_full_name", "").split(".")[:-1])
    if "ns_in_query" in names:
        cols["ns_in_query"] = np.array([sum(c in qtok for c in meta.ns_parts[p]) / len(meta.ns_parts[p])
                                        if meta.ns_parts[p] else 0.0 for p in cand])
    if "ns_thm" in names:
        def prefix(a, b):
            n = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                n += 1
            return n
        cols["ns_thm"] = np.array([prefix(meta.ns_parts[p], thm_ns) / len(meta.ns_parts[p]) if meta.ns_parts[p] else 0.0
                                   for p in cand])
    qfile = ctx.get("file_path", "")
    if "same_file" in names:
        cols["same_file"] = np.array([float(meta.path[p] == qfile) for p in cand])
    if "dir_overlap" in names:
        qdir = qfile.split("/")[:-1]
        def overlap(path):
            d = path.split("/")[:-1]
            n = 0
            for x, y in zip(d, qdir):
                if x != y:
                    break
                n += 1
            return n / max(len(qdir), 1)
        cols["dir_overlap"] = np.array([overlap(meta.path[p]) for p in cand])
    if "sym_jaccard" in names:
        gs = _symbols(_goal(state))
        def jac(p):
            ps = meta.symbols(p)
            u = len(gs | ps)
            return len(gs & ps) / u if u else 0.0
        cols["sym_jaccard"] = np.array([jac(p) for p in cand])
    for g in ("thm", "def", "inst", "type"):
        if f"kind_{g}" in names:
            cols[f"kind_{g}"] = meta.kind[g][cand]
    if "popularity" in names:
        cols["popularity"] = meta.popularity[cand]
    if "pagerank" in names:
        cols["pagerank"] = meta.pagerank[cand]
    if "accessible" in names:
        if access is None:
            access = meta.accessible_mask(qfile, ctx.get("theorem_start_line", 10**9))
        cols["accessible"] = access[cand].astype(np.float64)
    return np.column_stack([cols[n] for n in names]) if k else np.zeros((0, len(names)))
