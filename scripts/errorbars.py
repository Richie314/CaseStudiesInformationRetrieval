#!/usr/bin/env python3
"""Mean ± standard deviation of the per-run mean latency (and QPS) over
interleaved repeats of the same configuration, e.g. results/rep{1,2,3}_raw.json.

usage: errorbars.py --L 64 results/rep*_raw.json results/rep*_gef.json ...
Groups files by the name after the first underscore (rep1_raw -> raw).
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--L", type=int, default=64)
    args = ap.parse_args()
    files = []
    for f in args.files:
        files += sorted(glob.glob(f))
    groups = defaultdict(list)
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        run = next(x for x in r["runs"] if x["L"] == args.L)
        key = os.path.basename(f)[:-5].split("_", 1)[1]
        groups[key].append((run["mean_us"], run["qps"], r["graph_bytes"] * 8 / r["edges"], run["recall"]))
    print(f"| store | bits/edge | mean µs at L={args.L} (± sd, n) | QPS (± sd) | recall |")
    print("|---|---:|---:|---:|---:|")
    for key, vals in groups.items():
        us = [v[0] for v in vals]
        qps = [v[1] for v in vals]
        sd_us = statistics.stdev(us) if len(us) > 1 else 0.0
        sd_q = statistics.stdev(qps) if len(qps) > 1 else 0.0
        print(f"| {key} | {vals[0][2]:.2f} | {statistics.mean(us):.1f} ± {sd_us:.1f} (n={len(us)}) | "
              f"{statistics.mean(qps):.0f} ± {sd_q:.0f} | {vals[0][3]:.4f} |")


if __name__ == "__main__":
    main()
