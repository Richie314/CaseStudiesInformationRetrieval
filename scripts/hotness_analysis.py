#!/usr/bin/env python3
"""Analyse the per-node access-frequency profile of beam search.

Input: an expansion-count file written by search_bench --profile (one uint32
per node, in the graph's id space), the graph CSR, and its perm/entry point.
Reports how skewed the accesses are (share of expansions captured by the
top x% nodes, Zipf slope), and compares *caching policies* at equal node
budgets: pick the hot set by
  freq     measured expansion count (from a training query set)
  bfs      BFS hop distance from the entry point (DiskANN's cache policy)
  indeg    in-degree (hub-ness), a workload-free proxy
  outdeg   out-degree
  random   control
and measure the fraction of expansions from an *evaluation* profile (held-out
queries) that hit the hot set.
"""

import argparse
import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, shortest_path


def load_csr(prefix):
    off = np.fromfile(f"{prefix}_offsets.bin", dtype="<u8").astype(np.int64)
    nbr = np.fromfile(f"{prefix}_neighbors.bin", dtype="<u4").astype(np.int64)
    return off, nbr


def bfs_levels(off, nbr, entry):
    """Hop distance from entry over out-edges (unreached = large)."""
    N = len(off) - 1
    A = csr_matrix((np.ones(len(nbr), dtype=np.int8), nbr, off), shape=(N, N))
    dist = shortest_path(A, method="D", directed=True, unweighted=True, indices=entry)
    dist[~np.isfinite(dist)] = N
    return dist.astype(np.int64)


def coverage(counts_eval, hot_ids):
    return float(counts_eval[hot_ids].sum() / max(1, counts_eval.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True, help="CSR prefix (graph id space of the profiles)")
    ap.add_argument("--train-expand", required=True, help="expansion counts from the training queries")
    ap.add_argument("--eval-expand", required=True, help="expansion counts from held-out queries")
    ap.add_argument("--entry", type=int, required=True, help="entry point in the graph id space")
    ap.add_argument("--budgets", default="0.0001,0.001,0.005,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    off, nbr = load_csr(args.graph)
    N = len(off) - 1
    tr = np.fromfile(args.train_expand, dtype="<u4").astype(np.int64)
    ev = np.fromfile(args.eval_expand, dtype="<u4").astype(np.int64)
    assert len(tr) == N and len(ev) == N

    out = {"N": N, "train_total": int(tr.sum()), "eval_total": int(ev.sum()),
           "train_touched": int((tr > 0).sum()), "eval_touched": int((ev > 0).sum())}

    # --- skew of the training profile
    order = np.argsort(-tr, kind="stable")
    cum = np.cumsum(tr[order]) / tr.sum()
    budgets = [float(b) for b in args.budgets.split(",")]
    out["train_top_share"] = {str(b): float(cum[max(1, int(b * N)) - 1]) for b in budgets}
    ranks = np.arange(1, N + 1)
    m = (tr[order] > 0) & (ranks <= 100000)
    slope, _ = np.polyfit(np.log(ranks[m]), np.log(tr[order][m]), 1)
    out["zipf_slope_top1e5"] = float(slope)
    p = tr / tr.sum()
    p = p[p > 0]
    out["train_entropy_bits"] = float(-(p * np.log2(p)).sum())
    out["train_perplexity"] = float(2 ** out["train_entropy_bits"])
    out["train_max_count"] = int(tr.max())
    out["train_top10"] = [(int(i), int(tr[i])) for i in order[:10]]

    # --- policies
    lvl = bfs_levels(off, nbr, args.entry)
    out["bfs_level_sizes"] = {int(l): int((lvl == l).sum()) for l in range(0, int(min(lvl.max(), 12)) + 1)}
    out["bfs_level_eval_share"] = {int(l): float(ev[lvl == l].sum() / ev.sum())
                                   for l in range(0, int(min(lvl.max(), 12)) + 1)}
    indeg = np.bincount(nbr, minlength=N)
    outdeg = np.diff(off)
    rng = np.random.default_rng(args.seed)
    rand_order = rng.permutation(N)
    # bfs policy: by level, ties broken by id (BFS labelings already order by level)
    bfs_order = np.lexsort((np.arange(N), lvl))
    indeg_order = np.argsort(-indeg, kind="stable")
    outdeg_order = np.argsort(-outdeg, kind="stable")
    policies = {"freq": order, "bfs": bfs_order, "indeg": indeg_order,
                "outdeg": outdeg_order, "random": rand_order}
    out["eval_hit_rate"] = {}
    for name, po in policies.items():
        out["eval_hit_rate"][name] = {str(b): coverage(ev, po[: max(1, int(b * N))]) for b in budgets}
    # oracle: hot set chosen from the eval profile itself
    ev_order = np.argsort(-ev, kind="stable")
    out["eval_hit_rate"]["oracle"] = {str(b): coverage(ev, ev_order[: max(1, int(b * N))]) for b in budgets}
    # correlation between hotness and structural proxies
    out["spearman_freq_indeg"] = float(np.corrcoef(np.argsort(np.argsort(-tr)), np.argsort(np.argsort(-indeg)))[0, 1])
    out["spearman_freq_level"] = float(np.corrcoef(np.argsort(np.argsort(-tr)), np.argsort(np.argsort(lvl)))[0, 1])

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"N={N} train expansions={out['train_total']} touched={out['train_touched']} "
          f"zipf slope={slope:.2f} perplexity={out['train_perplexity']:.0f}")
    print("| budget (nodes) | " + " | ".join(out["eval_hit_rate"]) + " |")
    print("|---|" + "---:|" * len(out["eval_hit_rate"]))
    for b in budgets:
        h = max(1, int(b * N))
        print(f"| {b*100:g}% ({h}) | " + " | ".join(f"{out['eval_hit_rate'][p][str(b)]*100:.1f}%"
                                                for p in out["eval_hit_rate"]) + " |")
    print("BFS levels (size, eval share):", {l: (out["bfs_level_sizes"][l], round(out["bfs_level_eval_share"][l], 4))
                                             for l in out["bfs_level_sizes"]})


if __name__ == "__main__":
    main()
