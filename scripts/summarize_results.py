#!/usr/bin/env python3
"""Aggregate search_bench JSON results into markdown tables.

For each result file: graph bytes (bits/edge incl. offsets), and per L:
recall@10, QPS (best of repeats), mean latency, hops. Also interpolates the
latency at fixed recall targets so stores can be compared at equal recall.
"""

import argparse
import glob
import json
import os



def load(path):
    with open(path) as f:
        r = json.load(f)
    r["name"] = os.path.basename(path)[:-5]
    return r


def at_recall(runs, target):
    """Linear interpolation of mean latency (us) at a recall target along L."""
    rs = [(x["recall"], x["mean_us"]) for x in sorted(runs, key=lambda x: x["L"])]
    for (r0, t0), (r1, t1) in zip(rs, rs[1:]):
        if r0 <= target <= r1 and r1 > r0:
            return t0 + (t1 - t0) * (target - r0) / (r1 - r0)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--targets", default="0.95,0.97,0.99")
    ap.add_argument("--L", default="", help="only these L values (comma list)")
    args = ap.parse_args()
    targets = [float(t) for t in args.targets.split(",")]
    Ls = {int(x) for x in args.L.split(",")} if args.L else None

    files = []
    for f in args.files:
        files += sorted(glob.glob(f))
    rows = [load(f) for f in files]

    print("| run | store | bits/edge | " + " | ".join(f"us@R{t}" for t in targets) + " |")
    print("|---|---|---:|" + "---:|" * len(targets))
    for r in rows:
        bpe = (r["graph_bytes"] + r.get("hier_bytes", 0)) * 8 / r["edges"]
        store = r["backend"] + ("" if r["backend"] != "gef" else f"/{r['offsets']}")
        if r["backend"] == "hybrid":
            store += f" h={r['hot_count']}"
        cells = []
        for t in targets:
            v = at_recall(r["runs"], t)
            cells.append(f"{v:.0f}" if v else "-")
        print(f"| {r['name']} | {store} | {bpe:.2f} | " + " | ".join(cells) + " |")

    print()
    print("| run | L | recall@10 | QPS | mean us | p99 us | hops |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        for x in r["runs"]:
            if Ls and x["L"] not in Ls:
                continue
            print(f"| {r['name']} | {x['L']} | {x['recall']:.4f} | {x['qps']:.0f} | {x['mean_us']:.1f} | "
                  f"{x['p99_us']:.1f} | {x['hops']:.1f} |")


if __name__ == "__main__":
    main()
