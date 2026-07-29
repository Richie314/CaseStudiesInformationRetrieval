#!/usr/bin/env python3
"""Run the labeling x compression sweep over one or more exported CSR graphs.

For every input prefix, applies each relabeling (bfs, cm-dir, rcm; plus the
original ids), runs the U-GEF compressor on each, parses its report, and
aggregates everything into a JSON file and a markdown table.

The BFS / directed-C-M start node is taken from <prefix>_meta.json
("start" key) when present, else 0.

Example:
  ./venv/bin/python scripts/run_experiments.py \
      --binary ./build/information_retrieval \
      --out ./data/sweep_results.json \
      ./data/graph ./data/graph_R16 ./data/graph_R64 ./data/graph_nsg ./data/graph_hnsw
"""

import argparse
import json
import os
import re
import subprocess
import sys

LABELINGS = ("orig", "bfs", "cm-dir", "rcm")

LINE_RE = re.compile(
    r"^(offsets|neighbors|TOTAL)\s+raw=\s*(\d+) B\s+compressed=\s*(\d+) B\s+"
    r"ratio=([\d.]+) \(([\d.]+)%\)\s+bits/edge=([\d.]+)", re.M)


def compress(binary: str, prefix: str) -> dict:
    proc = subprocess.run([binary, prefix], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} {prefix} failed:\n{proc.stdout}{proc.stderr}")
    if "OK: all rows reconstruct exactly." not in proc.stdout:
        raise RuntimeError(f"verification failed for {prefix}:\n{proc.stdout}")
    out = {}
    for m in LINE_RE.finditer(proc.stdout):
        out[m.group(1).lower()] = {
            "raw_bytes": int(m.group(2)),
            "compressed_bytes": int(m.group(3)),
            "ratio": float(m.group(4)),
            "bits_per_edge": float(m.group(6)),
        }
    if set(out) != {"offsets", "neighbors", "total"}:
        raise RuntimeError(f"unparseable compressor output for {prefix}:\n{proc.stdout}")
    return out


def relabel(prefix: str, out_prefix: str, strategy: str, start: int):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "relabel_graph.py")
    subprocess.run(
        [sys.executable, script, "--prefix", prefix, "--out-prefix", out_prefix,
         "--strategy", strategy, "--start", str(start)],
        check=True)


def start_node(prefix: str) -> int:
    try:
        with open(f"{prefix}_meta.json") as f:
            return int(json.load(f).get("start", 0))
    except FileNotFoundError:
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefixes", nargs="+", help="CSR prefixes to sweep")
    parser.add_argument("--binary", default="./build/information_retrieval")
    parser.add_argument("--out", default="./data/sweep_results.json")
    parser.add_argument("--skip-relabel", action="store_true",
                        help="Assume relabeled CSR files already exist")
    args = parser.parse_args()

    results = {}
    for prefix in args.prefixes:
        name = os.path.basename(prefix)
        start = start_node(prefix)
        results[name] = {"start": start, "labelings": {}}
        for labeling in LABELINGS:
            if labeling == "orig":
                target = prefix
            else:
                suffix = labeling.replace("-", "")
                target = f"{prefix}_{suffix}"
                if not args.skip_relabel:
                    print(f"--- relabel {name} [{labeling}] (start={start})")
                    relabel(prefix, target, labeling if labeling != "cm-dir"
                            else "cm-dir", start)
            print(f"--- compress {name} [{labeling}]")
            entry = compress(args.binary, target)
            gap_file = f"{target}_stats.json"
            if os.path.exists(gap_file):
                with open(gap_file) as f:
                    gs = json.load(f)
                entry["gap_median"] = gs.get("gap_median")
                entry["gap_mean"] = gs.get("gap_mean")
            results[name]["labelings"][labeling] = entry

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {args.out}\n")

    print("| Graph | Labeling | Neighbors bits/edge | Total ratio |")
    print("|---|---|---:|---:|")
    for name, r in results.items():
        for labeling, e in r["labelings"].items():
            print(f"| {name} | {labeling} | {e['neighbors']['bits_per_edge']:.2f} "
                  f"| {e['total']['ratio']:.4f} |")


if __name__ == "__main__":
    main()
