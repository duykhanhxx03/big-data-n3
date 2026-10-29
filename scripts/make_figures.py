"""Sinh mọi biểu đồ và bảng quy mô từ results/raw (không chỉnh tay).

Đầu ra: results/figures/*.png và results/figures/scale_tables.md
Với --report: report/latex/figures/*.pdf cho báo cáo LaTeX (nền trắng, không tiêu đề trong hình vì tiêu đề nằm ở
caption, dấu phẩy thập phân và dấu chấm phân cách hàng nghìn theo quy ước tiếng Việt); không ghi lại bảng markdown.
Mỗi điểm = median qua các lần lặp; thanh lỗi = min–max.
Màu: bảng màu phân loại mặc định của skill dataviz (thứ tự cố định; 3 slot đầu an toàn mọi cặp với CVD).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "results" / "raw"
FIG = ROOT / "results" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SIZE_ORDER = ["10k", "100k", "full", "x5"]
INDEX_SIZES = SIZE_ORDER[:3]   # các mức chỉ mục của thí nghiệm quy mô: 10K, 100K, toàn bộ (181K)

REPORT = False   # đặt bởi --report
EXT = ".png"
FIGSIZE = (6.4, 4)      # báo cáo dùng hình nhỏ hơn để chữ giữ khoảng 8pt khi in
LABEL_FS, LEGEND_FS = 9, 8


def vn(v: float, fmt: str = "g") -> str:
    """Định dạng số; ở chế độ báo cáo dùng dấu phẩy thập phân và dấu chấm hàng nghìn."""
    s = format(v, fmt)
    return s.replace(",", "_").replace(".", ",").replace("_", ".") if REPORT else s


def title(ax, text: str):
    if not REPORT:
        ax.set_title(text, loc="left")


def save(fig, name: str):
    # PDF không ghi ngày tạo, để cùng dữ liệu luôn cho cùng tệp hình
    fig.savefig(FIG / (name + EXT), **({"metadata": {"CreationDate": None}} if EXT == ".pdf" else {}))


plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 6, "figure.dpi": 150,
})


INPUTS: set[Path] = set()


def load(kind: str) -> list[dict]:
    """Run record của một loại. Ở chế độ báo cáo chỉ đọc danh sách đã ghim trong results/figures/report_records.json
    (scripts/pin_runs.py), để một lần chạy phát sinh sau khi phát hành không làm hình thay đổi."""
    files = sorted((RAW / kind).glob("*.json"))
    pins = ROOT / "results" / "figures" / "report_records.json"
    if REPORT and pins.exists():
        ids = json.loads(pins.read_text(encoding="utf-8"))["records"].get("eval-scale" if kind == "eval" else kind)
        if ids is not None:
            files = [RAW / kind / f"{i}.json" for i in ids]
            INPUTS.add(pins)
    INPUTS.update(files)
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def agg(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    return (np.median(a), a.min(), a.max(), a.size) if a.size else (np.nan, np.nan, np.nan, 0)


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def style(color: str) -> dict:
    """Báo cáo in được đen trắng: mỗi chuỗi có màu, kiểu điểm và kiểu nét riêng (theo thứ tự màu cố định)."""
    i = SERIES.index(color) if color in SERIES else 0
    if not REPORT:
        return {"marker": "o", "linestyle": "-"}
    return {"marker": MARKERS[i], "linestyle": LINESTYLES[i]}


def line_with_range(ax, xs, stats, color, label, direct_label=True):
    med = np.array([s[0] for s in stats])
    lo = np.array([s[1] for s in stats])
    hi = np.array([s[2] for s in stats])
    ax.plot(xs, med, color=color, label=label, markeredgecolor=SURFACE, markeredgewidth=1.2, **style(color))
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
    if direct_label and len(xs):
        ax._end_labels = getattr(ax, "_end_labels", []) + [(xs[-1], med[-1], label)]


def place_end_labels(ax, min_gap_px: float = 15.0):
    """Nhãn trực tiếp ở cuối mỗi đường, giãn theo chiều dọc để không chồng nhau."""
    items = getattr(ax, "_end_labels", [])
    if not items or REPORT:   # hình trong báo cáo chỉ dùng chú giải, tránh lặp tên chuỗi
        return
    ax.figure.canvas.draw()
    pts = [(ax.transData.transform((x, y)), x, y, lab) for x, y, lab in items]
    pts.sort(key=lambda t: t[0][1])
    ys = [pt[0][1] for pt in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap_px)
    for (disp, x, y, lab), ynew in zip(pts, ys):
        ax.annotate(lab, (x, y), xytext=(8, ynew - disp[1]), textcoords="offset points", va="center",
                    color=INK2, fontsize=LABEL_FS)


def build_section(md: list[str]):
    recs = [r for r in load("build") if r["status"] == "ok"]
    if not recs:
        return
    groups = defaultdict(list)
    for r in recs:
        groups[(r["parameters"]["label"], r["parameters"]["size"])].append(r["metrics"])
    labels_main = ["seq", "spark2", "spark4", "spark8"]
    names = {"seq": "Tuần tự (1 nhân)", "spark2": "Spark local[2]", "spark4": "Spark local[4]", "spark8": "Spark local[8]"}
    md += ["## Build index (B0: default tokenizer, field statement, codec VB)\n",
           "Median (min–max) qua các lần lặp. Thời gian là wall time của cả tiến trình, gồm khởi động Spark.\n",
           "| Builder | Kích thước | n_docs | Lần | Wall (s) | Docs/s (job) | RAM đỉnh (MB) | Shuffle ghi (MB) | Task reduce max/median (ms) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for lab in labels_main + ["spark4-nocombine", "spark4-salt4", "seq-raw32", "seq-gamma"]:
        for size in SIZE_ORDER:
            ms = groups.get((lab, size))
            if not ms:
                continue
            w, wlo, whi, n = agg(m["wall_s"] for m in ms)
            ram = agg(m["peak_rss_mb"] for m in ms)
            dps = agg(m.get("docs_per_s") or (m["n_docs"] / m["total_s"] if m.get("total_s") else None) for m in ms)
            sh = agg((m.get("shuffle_write_bytes") or 0) / 2**20 for m in ms) if lab.startswith("spark") else None
            tmx = agg(m.get("reduce_task_ms_max") for m in ms)
            tmd = agg(m.get("reduce_task_ms_median") for m in ms)
            md.append(f"| {lab} | {size} | {ms[0]['n_docs']:,} | {n} | {w:.1f} ({wlo:.1f}–{whi:.1f}) | {dps[0]:,.0f} | "
                      f"{ram[0]:,.0f} | {sh[0]:.1f} |" if sh else
                      f"| {lab} | {size} | {ms[0]['n_docs']:,} | {n} | {w:.1f} ({wlo:.1f}–{whi:.1f}) | {dps[0]:,.0f} | {ram[0]:,.0f} | — |")
            md[-1] += f" {tmx[0]:.0f} / {tmd[0]:.0f} |" if lab.startswith("spark") and tmx[3] else " — |"
    md.append("")

    for metric, ttl, fname, ylabel in [("wall_s", "Thời gian build theo kích thước corpus", "fig_build_time", "Wall time (s, log)"),
                                         ("peak_rss_mb", "RAM đỉnh khi build", "fig_build_memory", "RAM đỉnh (MB, log)")]:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for i, lab in enumerate(labels_main):
            pts = [(groups[(lab, s)][0]["n_docs"], agg(m[metric] for m in groups[(lab, s)])) for s in SIZE_ORDER if (lab, s) in groups]
            if pts:
                line_with_range(ax, [p[0] for p in pts], [p[1] for p in pts], SERIES[i], names[lab])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([200, 500, 1000, 2000, 4000] if metric == "peak_rss_mb" else [1, 2, 5, 10, 20, 50])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: vn(v, ",.0f") if v >= 1 else vn(v)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Số bản ghi trong corpus (log)" if REPORT else
                      "Số premise trong index (log; mức cuối là corpus tổng hợp ×5)")
        ax.set_ylabel(ylabel)
        title(ax, ttl)
        ax.legend(loc="upper left" if metric == "peak_rss_mb" else "lower right", fontsize=LEGEND_FS)
        ax.margins(x=0.08 if REPORT else 0.3)
        fig.tight_layout()
        if not REPORT:          # hình nhỏ của báo cáo chỉ dùng chú giải, nhãn cuối đường sẽ bị chồng
            place_end_labels(ax)
        save(fig, fname)
        plt.close(fig)


def quality_scale_section(md: list[str]):
    rows = []
    for r in load("eval"):
        p = r["parameters"]
        if p.get("tag") == "scale-quality":
            rows.append({"name": p["config"]["name"], "subset": p["subset"], "R@10": r["metrics"]["R@10"],
                         "MRR": r["metrics"]["MRR"], "ts": r["timestamp"], "n_docs": r["index"]["n_docs"]})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("ts").groupby(["name", "subset"]).tail(1)
    md += ["## Chất lượng theo kích thước index (mẫu 500 truy vấn random/test)\n",
           "| Biến thể | " + " | ".join(INDEX_SIZES) + " |", "|---" * (len(INDEX_SIZES) + 1) + "|"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, name in enumerate(["B0", "V2", "V4"]):
        d = df[df.name == name].set_index("subset").reindex(INDEX_SIZES).dropna()
        md.append(f"| {name} | " + " | ".join(f"{d.loc[s, 'R@10']:.3f}" if s in d.index else "—" for s in INDEX_SIZES) + " |")
        ax.plot(d.n_docs, d["R@10"], color=SERIES[i], label=name, markeredgecolor=SURFACE, markeredgewidth=1.2,
                **style(SERIES[i]))
        if not REPORT:
            ax.annotate(name, (d.n_docs.iloc[-1], d["R@10"].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=LABEL_FS)
    ax.set_xscale("log")
    ax.set_xlabel("Số premise trong index (log)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: vn(v, ".1f")))
    title(ax, "Recall@10 giảm khi kho lớn hơn")
    ax.legend(loc="lower left", fontsize=LEGEND_FS)
    ax.margins(x=0.2)
    fig.tight_layout()
    save(fig, "fig_quality_vs_size")
    plt.close(fig)
    md.append("")
    return df


def serve_section(md: list[str], qdf):
    rows = []
    for r in load("loadtest"):
        tag = r["parameters"].get("tag") or ""
        if not tag.startswith("scale-"):
            continue
        size = tag.split("-")[1]
        if size not in INDEX_SIZES:   # các lần đo ở mức 50K cũ vẫn còn trong results/raw nhưng không đưa vào báo cáo
            continue
        for lv in r["metrics"]["levels"]:
            rows.append({"mode": r["parameters"]["mode"], "size": size, "clients": lv["clients"], "p50": lv["e2e_ms_p50"],
                         "p95": lv["e2e_ms_p95"], "qps": lv["qps"], "errors": lv["errors"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    g = df.groupby(["mode", "size", "clients"])
    md += ["## Độ trễ và thông lượng (API 4 worker, mỗi mức 400 request)\n",
           "| Mode | Index | Client | Lần | P50 (ms) | P95 (ms) | QPS | Lỗi |", "|---|---|---|---|---|---|---|---|"]
    for (mode, size, c), d in sorted(g, key=lambda x: (x[0][0], SIZE_ORDER.index(x[0][1]), x[0][2])):
        p95 = agg(d.p95)
        md.append(f"| {mode} | {size} | {c} | {len(d)} | {np.median(d.p50):.1f} | {p95[0]:.1f} ({p95[1]:.1f}–{p95[2]:.1f}) | "
                  f"{np.median(d.qps):.1f} | {int(d.errors.sum())} |")
    md.append("")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, mode in enumerate(["B0", "V2", "V4"]):
        d = df[(df["mode"] == mode) & (df["size"] == "full")]
        if d.empty:
            continue
        cs = sorted(d.clients.unique())
        line_with_range(ax, cs, [agg(d[d.clients == c].p95) for c in cs], SERIES[i], mode)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([20, 50, 100, 200, 500])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: vn(v, ",.0f")))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([1, 10, 50], ["1", "10", "50"])
    ax.set_xlabel("Số client đồng thời")
    ax.set_ylabel("P95 đầu cuối (ms, log)")
    title(ax, "P95 theo tải, index đầy đủ 181K premise")
    ax.legend(loc="upper left", fontsize=LEGEND_FS)
    ax.margins(x=0.25)
    fig.tight_layout()
    place_end_labels(ax)
    save(fig, "fig_latency_vs_clients")
    plt.close(fig)

    if qdf is not None:
        n_docs = qdf.groupby("subset").n_docs.first()
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for i, mode in enumerate(["B0", "V2", "V4"]):
            d = df[(df["mode"] == mode) & (df.clients == 10)]
            sizes = [s for s in INDEX_SIZES if s in set(d["size"]) and s in n_docs.index]
            if sizes:
                line_with_range(ax, [n_docs[s] for s in sizes], [agg(d[d["size"] == s].p95) for s in sizes], SERIES[i], mode)
        ax.set_xscale("log")
        ax.set_xlabel("Số premise trong index (log)")
        ax.set_ylabel("P95 ở 10 client (ms)")
        ax.set_ylim(bottom=0)
        title(ax, "P95 tăng theo kích thước index")
        ax.legend(loc="upper left", fontsize=LEGEND_FS)
        ax.margins(x=0.25)
        fig.tight_layout()
        place_end_labels(ax)
        save(fig, "fig_latency_vs_size")
        plt.close(fig)

        # Đánh đổi chất lượng – chi phí ở index đầy đủ: trả lời câu hỏi trung tâm của đề
        fig, ax = plt.subplots(figsize=FIGSIZE)
        pts = []
        for i, mode in enumerate(["B0", "V2", "V4"]):
            d = df[(df["mode"] == mode) & (df["size"] == "full") & (df.clients == 10)]
            q = qdf[(qdf.name == mode) & (qdf.subset == "full")]
            if d.empty or q.empty:
                continue
            m, lo, hi, _ = agg(d.p95)
            r10 = q["R@10"].iloc[0]
            ax.errorbar([m], [r10], xerr=[[m - lo], [hi - m]], fmt=style(SERIES[i])["marker"], color=SERIES[i],
                        markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=0, elinewidth=2, label=mode)
            ax.annotate(f"{mode}: R@10 = {vn(r10, '.3f')}; P95 = {vn(m, '.1f')} ms", (m, r10), xytext=(0, 10),
                        textcoords="offset points", ha="center", va="bottom", color=INK2, fontsize=LABEL_FS)
            pts.append((hi, r10))
        ax.set_xlabel("P95 đầu cuối ở 10 client (ms)")
        ax.set_ylabel("Recall@10 (mẫu 500 truy vấn)")
        if pts:
            ax.set_xlim(0, max(x for x, _ in pts) * 1.1)
            ax.set_ylim(0, max(y for _, y in pts) * 1.2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: vn(v, ".2f")))
        title(ax, "Đánh đổi chất lượng và độ trễ, index đầy đủ")
        ax.legend(loc="lower right", fontsize=LEGEND_FS)
        fig.tight_layout()
        save(fig, "fig_tradeoff")
        plt.close(fig)


PLAT_NAME = {"win": "Windows", "linux": "Linux"}
INDEX_NAME = {"b0": "chỉ mục B0 (1 trường, tách từ mặc định)", "lean": "chỉ mục V2 đến V4 (3 trường, tách từ Lean)"}


def build2_section(md: list[str]):
    """Engine Spark tối ưu (arrow) so với bản tuần tự, trên Windows và Linux (run_scale.py --part build2)."""
    recs = [r for r in load("build") if r["parameters"].get("experiment") == "build2"]
    if not recs:
        return
    groups: dict = defaultdict(list)
    for r in recs:
        plat, index, kind = r["parameters"]["label"].split("-", 2)
        groups[(plat, index, kind, r["parameters"]["size"])].append(r)
    plats = [p for p in ("win", "linux") if any(k[0] == p for k in groups)]
    sizes = [s for s in ("10k", "100k", "full", "x5", "x20") if any(k[3] == s for k in groups)]
    masters = sorted({int(k[2][5:]) for k in groups if k[2].startswith("arrow")})

    def ok(key):
        return [r for r in groups.get(key, []) if r["status"] == "ok"]

    def wall(key):
        return agg(r["metrics"]["wall_s"] for r in ok(key))

    def n_docs(index, size):
        for p in plats:
            for r in ok((p, index, "seq", size)):
                return r["metrics"]["n_docs"]
        return None

    md += ["## Engine Spark tối ưu so với bản tuần tự (run_scale.py --part build2)\n",
           "Wall time của cả tiến trình (gồm khởi động Spark), trung vị (nhỏ nhất đến lớn nhất) qua các lần lặp. "
           "Tăng tốc = trung vị tuần tự / trung vị Spark nhanh nhất. Cột cuối kiểm tra mọi lần chạy cùng chỉ mục và kích "
           "thước (cả hai nền tảng, mọi builder) cho cùng mã SHA-1 của từng tệp chỉ mục.\n",
           "| Nền tảng | Chỉ mục | Kích thước | n_docs | Lần | Tuần tự (s) | "
           + " | ".join(f"Spark local[{m}] (s)" for m in masters) + " | Tăng tốc | Trùng byte |",
           "|---" * (7 + len(masters) + 1) + "|"]
    for p in plats:
        for index in ("b0", "lean"):
            for size in sizes:
                seq = wall((p, index, "seq", size))
                if not seq[3]:
                    continue
                cells, best = [], np.inf
                for m in masters:
                    w = wall((p, index, f"arrow{m}", size))
                    cells.append(f"{w[0]:.1f} ({w[1]:.1f}; {w[2]:.1f})" if w[3] else "—")
                    if w[3]:
                        best = min(best, w[0])
                sums = {json.dumps(r["metrics"].get("checksums"), sort_keys=True)
                        for pp in plats for kind in ["seq", *[f"arrow{m}" for m in masters]]
                        for r in ok((pp, index, kind, size))}
                md.append(f"| {PLAT_NAME[p]} | {index} | {size} | {n_docs(index, size):,} | {seq[3]} | "
                          f"{seq[0]:.1f} ({seq[1]:.1f}; {seq[2]:.1f}) | " + " | ".join(cells)
                          + f" | {seq[0] / best:.2f} | {'có' if len(sums) == 1 else 'KHÔNG'} |")
    md.append("")

    md += ["### Bộ nhớ và shuffle ở kích thước lớn nhất (trung vị)\n",
           "| Nền tảng | Chỉ mục | Builder | RAM đỉnh (MB) | Bản ghi shuffle | Shuffle ghi (MB) | Số posting |",
           "|---|---|---|---|---|---|---|"]
    big = sizes[-1]
    for p in plats:
        for index in ("b0", "lean"):
            for kind in ["seq", *[f"arrow{m}" for m in masters], "rdd4"]:
                rs = ok((p, index, kind, big))
                if not rs:
                    continue
                med = lambda k: float(np.median([r["metrics"].get(k) or 0 for r in rs]))  # noqa: E731
                md.append(f"| {PLAT_NAME[p]} | {index} | {kind} | {med('peak_rss_mb'):,.0f} | "
                          f"{med('shuffle_write_records'):,.0f} | {med('shuffle_write_bytes') / 2**20:,.1f} | "
                          f"{med('n_postings'):,.0f} |")
    for p in plats:  # engine RDD cũ ở x20: có thể hết giờ hoặc lỗi bộ nhớ
        for r in groups.get((p, "b0", "rdd4", "x20"), []):
            if r["status"] != "ok":
                md.append(f"| {PLAT_NAME[p]} | b0 | rdd4 | {r['metrics'].get('peak_rss_mb', 0):,.0f} | "
                          f"trạng thái: {r['status']} sau {r['metrics']['wall_s']:.0f} s | | |")
    md.append("")

    # Hình 1: thời gian theo kích thước, mỗi loại chỉ mục một hình
    big_m = max(masters) if masters else None
    for index in ("b0", "lean"):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        i = 0
        for p in plats:
            for kind, name in (("seq", "tuần tự"), (f"arrow{big_m}", f"Spark local[{big_m}]")):
                pts = [(n_docs(index, s), wall((p, index, kind, s))) for s in sizes if wall((p, index, kind, s))[3]]
                if pts:
                    line_with_range(ax, [x for x, _ in pts], [st for _, st in pts], SERIES[i],
                                    f"{PLAT_NAME[p]}, {name}", direct_label=False)
                i += 1
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 500])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: vn(v, ",.0f")))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Số bản ghi trong corpus (log)")
        ax.set_ylabel("Wall time (s, log)")
        title(ax, f"Thời gian xây {INDEX_NAME[index]}")
        ax.legend(loc="upper left", fontsize=LEGEND_FS)
        fig.tight_layout()
        save(fig, f"fig_build2_time_{index}")
        plt.close(fig)

    # Hình 2: tăng tốc so với bản tuần tự cùng nền tảng, chỉ mục lean
    fig, ax = plt.subplots(figsize=FIGSIZE)
    i = 0
    series = [("linux", m) for m in masters] + [("win", big_m)] if "linux" in plats else [("win", m) for m in masters]
    for p, m in series:
        pts = []
        for s in sizes:
            seq, sp = wall((p, "lean", "seq", s)), wall((p, "lean", f"arrow{m}", s))
            if seq[3] and sp[3]:
                pts.append((n_docs("lean", s), seq[0] / sp[0]))
        if pts:
            ax.plot([x for x, _ in pts], [y for _, y in pts], color=SERIES[i], markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label=f"{PLAT_NAME[p]}, Spark local[{m}]", **style(SERIES[i]))
        i += 1
    ax.set_xscale("log")
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
    ax.annotate("bằng bản tuần tự", xy=(0.98, 1.0), xycoords=("axes fraction", "data"), xytext=(0, -3),
                textcoords="offset points", ha="right", va="top", color=INK2, fontsize=LABEL_FS)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: vn(v, ".1f")))
    ax.set_xlabel("Số bản ghi trong corpus (log)")
    ax.set_ylabel("Tăng tốc so với bản tuần tự (lần)")
    title(ax, "Tăng tốc của Spark, chỉ mục 3 trường")
    ax.legend(loc="upper left", fontsize=LEGEND_FS)
    fig.tight_layout()
    save(fig, "fig_build2_speedup")
    plt.close(fig)


def serve2_section(md: list[str]):
    """Mở rộng dịch vụ theo số worker (run_scale.py --part serve2) và kiểm tra CPU (serve_cpu_check.py)."""
    rows = []
    for r in load("loadtest"):
        tag = r["parameters"].get("tag") or ""
        if not tag.startswith("workers-"):
            continue
        w = int(tag.split("-")[1])
        for lv in r["metrics"]["levels"]:
            rows.append({"mode": r["parameters"]["mode"], "workers": w, "clients": lv["clients"],
                         "p95": lv["e2e_ms_p95"], "qps": lv["qps"], "errors": lv["errors"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    mem = defaultdict(list)
    for r in load("serve-memory"):
        mem[r["parameters"]["workers"]].append(r["metrics"]["server_rss_mb"])
    clients = sorted(df.clients.unique())
    md += ["## Mở rộng dịch vụ theo số worker (index đầy đủ, run_scale.py --part serve2)\n",
           "P95 đầu cuối (ms) và QPS, trung vị của 3 lần; RAM là tổng RSS của mọi tiến trình server sau khi nạp xong.\n",
           "| Mode | Worker | RAM server (MB) | " + " | ".join(f"{c} client: P95 / QPS" for c in clients) + " | Lỗi |",
           "|---" * (4 + len(clients)) + "|"]
    for mode in ("B0", "V2", "V4"):
        for w in sorted(df.workers.unique()):
            d = df[(df["mode"] == mode) & (df.workers == w)]
            if d.empty:
                continue
            cells = [f"{np.median(d[d.clients == c].p95):.0f} / {np.median(d[d.clients == c].qps):.0f}" for c in clients]
            md.append(f"| {mode} | {w} | {np.median(mem[w]) if mem[w] else float('nan'):,.0f} | " + " | ".join(cells)
                      + f" | {int(d.errors.sum())} |")
    md.append("")
    v5 = []
    for r in load("loadtest"):
        if (r["parameters"].get("tag") or "").startswith("v5cmp-"):
            for lv in r["metrics"]["levels"]:
                v5.append({"mode": r["parameters"]["mode"], "clients": lv["clients"], "p50": lv["e2e_ms_p50"],
                           "p95": lv["e2e_ms_p95"], "qps": lv["qps"]})
    if v5:
        d5 = pd.DataFrame(v5)
        md += ["### V4 so với V5 dưới tải (4 worker, cùng server, run_scale.py --part serve_v5)\n",
               "| Mode | Client | Lần | P50 (ms) | P95 (ms) | QPS |", "|---|---|---|---|---|---|"]
        for (mode, c), g in d5.groupby(["mode", "clients"]):
            p95 = agg(g.p95)
            md.append(f"| {mode} | {c} | {len(g)} | {np.median(g.p50):.1f} | {p95[0]:.1f} ({p95[1]:.1f}; {p95[2]:.1f}) | "
                      f"{np.median(g.qps):.0f} |")
        md.append("")
    cpu = load("serve-cpu")
    if cpu:
        md += ["### Kiểm tra CPU khi chịu tải (B0, 48 client, scripts/serve_cpu_check.py)\n",
               "| Worker | QPS, 1 tiến trình client | QPS, 4 tiến trình client | Tổng CPU của worker (%) | Luồng logic của máy |",
               "|---|---|---|---|---|"]
        for r in sorted(cpu, key=lambda x: x["metrics"]["workers"]):
            m = r["metrics"]
            md.append(f"| {m['workers']} | {m['1x48']['qps']:.0f} | {m['4x12']['qps']:.0f} | "
                      f"{m['1x48']['worker_cpu_pct']['total']:.0f} đến {m['4x12']['worker_cpu_pct']['total']:.0f} | "
                      f"{m['logical_cpus']} |")
        md.append("")


def main() -> None:
    global REPORT, EXT, FIG, SURFACE, FIGSIZE, LABEL_FS, LEGEND_FS
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="xuất PDF cho báo cáo LaTeX vào report/latex/figures")
    if ap.parse_args().report:
        REPORT, EXT, FIG, SURFACE = True, ".pdf", ROOT / "report" / "latex" / "figures", "#ffffff"
        # hình in ở nửa bề rộng trang (khoảng 0,87 lần kích thước gốc): chữ 10pt còn khoảng 8,7pt trên giấy
        FIGSIZE, LABEL_FS, LEGEND_FS = (3.6, 2.8), 9, 8.5
        plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                             "pdf.fonttype": 42, "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                             "ytick.labelsize": 9, "lines.linewidth": 1.6, "lines.markersize": 5})
    FIG.mkdir(parents=True, exist_ok=True)
    md = ["# Kết quả quy mô (sinh bởi scripts/make_figures.py)\n"]
    build_section(md)
    build2_section(md)
    qdf = quality_scale_section(md)
    serve_section(md, qdf)
    serve2_section(md)
    if not REPORT:
        (FIG / "scale_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))
    else:   # nguồn của từng hình kèm SHA-256, đối chiếu bằng scripts/verify_report_inputs.py
        from runlog import file_sha256
        lines = ["# Tệp đầu vào của các hình (sinh bởi scripts/make_figures.py --report)", "",
                 "| Tệp | SHA-256 |", "|---|---|"]
        lines += [f"| `{x.relative_to(ROOT).as_posix()}` | `{file_sha256(x)}` |"
                  for x in sorted(INPUTS)]
        (FIG / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
