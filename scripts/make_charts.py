#!/usr/bin/env python3
"""Generate the SVG charts for the search-time / caching sections of REPORT.md
from the JSON results (same hand-made style as the existing charts:
light/dark aware via CSS variables, no external dependencies).

  docs/charts/pareto-latency-space.svg  latency at recall 0.97 vs bits/edge per store
  docs/charts/hit-rate-policy.svg       hot-set hit rate vs node budget per policy
"""

import argparse
import json
import os

STYLE = """<style>
:root{--ink:#0b0b0b;--sec:#52514e;--mut:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--blue:#2a78d6;--orange:#eb6834;--aqua:#1baf7a;--gray:#898781;--purple:#8a5cd6;--red:#d63a3a}
@media (prefers-color-scheme: dark){:root{--ink:#ffffff;--sec:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--blue:#3987e5;--orange:#d95926;--aqua:#199e70;--purple:#9b74e0;--red:#e05252}}
text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif}
.t{fill:var(--ink);font-size:15px;font-weight:600}
.s{fill:var(--sec);font-size:12.5px}
.m{fill:var(--mut);font-size:12px}
.v{fill:var(--ink);font-size:12px}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
</style>
"""


def at_recall(runs, target):
    rs = sorted(((x["recall"], x["mean_us"]) for x in runs), key=lambda p: p[0])
    for (r0, t0), (r1, t1) in zip(rs, rs[1:]):
        if r0 <= target <= r1 and r1 > r0:
            return t0 + (t1 - t0) * (target - r0) / (r1 - r0)
    return None


def load(path):
    with open(path) as f:
        return json.load(f)


def pareto_chart(points, out, title, subtitle):
    """points: list of (label, bits_per_edge, latency_us, color)."""
    W, H = 720, 360
    x0, x1, y0, y1 = 80, 700, 300, 80
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    xmin, xmax = 16, 36
    ymin, ymax = 0, max(ys) * 1.15
    sx = lambda x: x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
    sy = lambda y: y0 - (y - ymin) / (ymax - ymin) * (y0 - y1)
    o = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{title}">', STYLE]
    o.append(f'<text class="t" x="16" y="28">{title}</text>')
    o.append(f'<text class="s" x="16" y="48">{subtitle}</text>')
    for gy in range(0, int(ymax) + 1, 50):
        o.append(f'<line class="grid" x1="{x0}" y1="{sy(gy):.1f}" x2="{x1}" y2="{sy(gy):.1f}"/>')
        o.append(f'<text class="m" x="{x0 - 8}" y="{sy(gy) + 4:.1f}" text-anchor="end">{gy}</text>')
    for gx in range(16, 37, 4):
        o.append(f'<line class="grid" x1="{sx(gx):.1f}" y1="{y1}" x2="{sx(gx):.1f}" y2="{y0}"/>')
        o.append(f'<text class="m" x="{sx(gx):.1f}" y="{y0 + 18}" text-anchor="middle">{gx}</text>')
    o.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}"/>')
    o.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}"/>')
    o.append(f'<text class="s" x="{(x0 + x1) / 2}" y="{H - 8}" text-anchor="middle">graph space, bits/edge (incl. offsets) — smaller is better</text>')
    o.append(f'<text class="s" x="18" y="{(y0 + y1) / 2}" text-anchor="middle" transform="rotate(-90 18 {(y0 + y1) / 2})">latency at recall@10 = 0.97, µs</text>')
    for label, bx, ly, color, dx, dy in points:
        o.append(f'<circle cx="{sx(bx):.1f}" cy="{sy(ly):.1f}" r="5.5" fill="var(--{color})"/>')
        o.append(f'<text class="v" x="{sx(bx) + dx:.1f}" y="{sy(ly) + dy:.1f}">{label}</text>')
    o.append('</svg>')
    with open(out, "w") as f:
        f.write("\n".join(o) + "\n")


def hitrate_chart(hot, out, title, subtitle):
    W, H = 720, 360
    x0, x1, y0, y1 = 80, 700, 300, 80
    budgets = sorted(float(b) for b in hot["eval_hit_rate"]["freq"])
    budgets = [b for b in budgets if b <= 0.5]
    sx = lambda b: x0 + b / 0.5 * (x1 - x0)
    sy = lambda v: y0 - v * (y0 - y1)
    o = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{title}">', STYLE]
    o.append(f'<text class="t" x="16" y="28">{title}</text>')
    o.append(f'<text class="s" x="16" y="48">{subtitle}</text>')
    for gy in range(0, 101, 20):
        o.append(f'<line class="grid" x1="{x0}" y1="{sy(gy / 100):.1f}" x2="{x1}" y2="{sy(gy / 100):.1f}"/>')
        o.append(f'<text class="m" x="{x0 - 8}" y="{sy(gy / 100) + 4:.1f}" text-anchor="end">{gy}%</text>')
    for gx in range(0, 51, 10):
        o.append(f'<line class="grid" x1="{sx(gx / 100):.1f}" y1="{y1}" x2="{sx(gx / 100):.1f}" y2="{y0}"/>')
        o.append(f'<text class="m" x="{sx(gx / 100):.1f}" y="{y0 + 18}" text-anchor="middle">{gx}%</text>')
    o.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}"/>')
    o.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}"/>')
    o.append(f'<text class="s" x="{(x0 + x1) / 2}" y="{H - 8}" text-anchor="middle">hot set size, % of nodes</text>')
    o.append(f'<text class="s" x="18" y="{(y0 + y1) / 2}" text-anchor="middle" transform="rotate(-90 18 {(y0 + y1) / 2})">share of held-out expansions served by the hot set</text>')
    series = [("oracle", "gray", "5 4"), ("freq", "orange", None), ("indeg", "aqua", None),
              ("outdeg", "purple", None), ("bfs", "blue", None), ("random", "red", None)]
    labels = {"oracle": "oracle (ranked on the test queries)", "freq": "frequency (200k sampled base queries)",
              "indeg": "in-degree (hubs)", "outdeg": "out-degree", "bfs": "BFS levels from the medoid (DiskANN)",
              "random": "random"}
    ly = 66
    for name, color, dash in series:
        pts = " ".join(f"{sx(b):.1f},{sy(hot['eval_hit_rate'][name][str(b) if str(b) in hot['eval_hit_rate'][name] else repr(b)]):.1f}" for b in budgets)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<polyline points="{pts}" fill="none" stroke="var(--{color})" stroke-width="2.2"{da}/>')
        o.append(f'<line x1="{x1 - 250}" y1="{ly}" x2="{x1 - 226}" y2="{ly}" stroke="var(--{color})" stroke-width="2.2"{da}/>')
        o.append(f'<text class="s" x="{x1 - 220}" y="{ly + 4}">{labels[name]}</text>')
        ly += 17
    o.append('</svg>')
    with open(out, "w") as f:
        f.write("\n".join(o) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--charts", default="docs/charts")
    args = ap.parse_args()
    R = args.results
    # (results file, label, color, label dx, dy)
    spec = [
        ("seq_r32_bfs_raw", "raw CSR", "blue", 8, 4),
        ("seq_r32_bfs_packed", "packed (DiskANN layout)", "blue", 8, 16),
        ("seq_r32_bfs_gef", "U-GEF", "orange", 8, 4),
        ("codec_bfs_rowef", "row EF", "aqua", 8, 4),
        ("codec_bfs_rowbic", "row BIC", "aqua", 8, -8),
        ("codec_bfs_rowpack", "row packed gaps", "aqua", 8, 16),
        ("hyb_freq_100000", "hybrid 10% hot + U-GEF", "purple", 8, 4),
        ("hyb_freq_200000", "hybrid 20% hot + U-GEF", "purple", 8, 4),
        ("hybrowef_freq_100000", "hybrid 10% hot + row EF", "purple", 8, 16),
    ]
    points = []
    for name, label, color, dx, dy in spec:
        p = os.path.join(R, name + ".json")
        if not os.path.exists(p):
            continue
        r = load(p)
        bpe = r["graph_bytes"] * 8 / r["edges"]
        lat = at_recall(r["runs"], 0.97)
        if lat:
            points.append((label, bpe, lat, color, dx, dy))
    pareto_chart(points, os.path.join(args.charts, "pareto-latency-space.svg"),
                 "Search latency vs graph space per adjacency store",
                 "SIFT-1M, DiskANN R=32, BFS labeling, single thread; latency interpolated at recall@10 = 0.97")
    hot = load(os.path.join(R, "hotness_r32_bfs_bsample200k.json"))
    hitrate_chart(hot, os.path.join(args.charts, "hit-rate-policy.svg"),
                  "Hot-set hit rate vs budget by caching policy",
                  "SIFT-1M, DiskANN R=32, L=64; policies ranked on the training profile, hit rate on 5,000 held-out queries")
    print("charts written:", len(points), "pareto points")


if __name__ == "__main__":
    main()
