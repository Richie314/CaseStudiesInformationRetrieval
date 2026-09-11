#!/usr/bin/env python3
"""Compare the measured latency of the two-tier (hybrid) stores with the
prediction of the additive access-cost model

    t_hybrid(h) = t_raw + (1 - hit(h)) * (t_gef - t_raw)

where hit(h) is the fraction of node expansions served by the hot tier
(from hotness_analysis.py on held-out queries) and t_raw / t_gef are the
measured mean latencies of the all-raw / all-compressed stores at the same
L. Also prints the space of each store in bits/edge.
"""

import argparse
import glob
import json
import os


def load(path):
    with open(path) as f:
        return json.load(f)


def runs_by_L(r):
    return {x["L"]: x for x in r["runs"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="results json of the all-raw store")
    ap.add_argument("--gef", required=True, help="results json of the all-compressed store")
    ap.add_argument("--hotness", required=True, help="hotness json (eval_hit_rate)")
    ap.add_argument("--policy", default="freq")
    ap.add_argument("--hybrids", nargs="+", required=True, help="results json files of hybrid runs")
    ap.add_argument("--L", type=int, default=64)
    args = ap.parse_args()

    raw = load(args.raw)
    gef = load(args.gef)
    hot = load(args.hotness)
    N = hot["N"]
    L = args.L
    t_raw = runs_by_L(raw)[L]["mean_us"]
    t_gef = runs_by_L(gef)[L]["mean_us"]
    print(f"L={L}: t_raw={t_raw:.1f}us t_gef={t_gef:.1f}us  (decode overhead {t_gef - t_raw:.1f}us = "
          f"{(t_gef / t_raw - 1) * 100:.0f}%)")
    print("| hot nodes | budget | bits/edge | hit rate (held-out) | predicted us | measured us | measured/raw |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    files = []
    for f in args.hybrids:
        files += sorted(glob.glob(f))
    rows = []
    for f in files:
        r = load(f)
        h = r["hot_count"]
        b = h / N
        key = None
        for k in hot["eval_hit_rate"][args.policy]:
            if abs(float(k) - b) < 1e-9:
                key = k
        hit = hot["eval_hit_rate"][args.policy][key] if key else float("nan")
        pred = t_raw + (1 - hit) * (t_gef - t_raw)
        meas = runs_by_L(r)[L]["mean_us"]
        bpe = r["graph_bytes"] * 8 / r["edges"]
        rows.append((h, b, bpe, hit, pred, meas))
    for h, b, bpe, hit, pred, meas in sorted(rows):
        print(f"| {h} | {b * 100:.0f}% | {bpe:.2f} | {hit * 100:.1f}% | {pred:.1f} | {meas:.1f} | {meas / t_raw:.2f} |")


if __name__ == "__main__":
    main()
