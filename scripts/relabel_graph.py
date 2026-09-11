#!/usr/bin/env python3
"""Relabel graph node ids to reduce adjacency-matrix bandwidth.

Elias-Fano-style encoders exploit small gaps between consecutive sorted
neighbor ids, so a permutation that clusters each node's neighbors around
its own id (bandwidth minimization) directly improves compression.

Three strategies:
  rcm     Reverse Cuthill-McKee on the symmetrized adjacency matrix
  bfs     BFS visit order from a start node -- a cheap approximation of
          Cuthill-McKee (same idea, no by-degree tie-breaking)
  cm-dir  Cuthill-McKee over the directed graph: BFS following out-edges
          only, enqueuing each node's unvisited out-neighbors in order of
          increasing out-degree (no standard directed C-M exists; this
          transplants its by-degree tie-breaking onto the directed BFS)

Reads <prefix>_offsets.bin / <prefix>_neighbors.bin, writes the same pair
under <out-prefix>, plus <out-prefix>_stats.json with gap statistics.
"""

import argparse
import json
import time
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, reverse_cuthill_mckee


def load_csr(prefix: str):
    offsets = np.fromfile(f"{prefix}_offsets.bin", dtype="<u8").astype(np.int64)
    neighbors = np.fromfile(f"{prefix}_neighbors.bin", dtype="<u4").astype(np.int64)
    if len(offsets) == 0 or offsets[-1] != len(neighbors):
        raise ValueError("offsets/neighbors mismatch")
    return offsets, neighbors


def as_scipy(offsets, neighbors):
    N = len(offsets) - 1
    data = np.ones(len(neighbors), dtype=np.int8)
    return csr_matrix((data, neighbors, offsets), shape=(N, N))


def rcm_permutation(A):
    """perm[k] = old id that receives new id k."""
    sym = ((A + A.T) > 0).tocsr()
    return np.asarray(reverse_cuthill_mckee(sym, symmetric_mode=True), dtype=np.int64)


def bfs_permutation(A, start: int):
    order = np.asarray(
        breadth_first_order(A, start, directed=True, return_predecessors=False),
        dtype=np.int64,
    )
    if len(order) < A.shape[0]:
        seen = np.zeros(A.shape[0], dtype=bool)
        seen[order] = True
        rest = np.flatnonzero(~seen)
        print(f"BFS reached {len(order)} nodes; appending {len(rest)} unreached")
        order = np.concatenate([order, rest])
    return order


def cm_directed_permutation(offsets, neighbors, start: int):
    """Cuthill-McKee transplanted onto the directed graph: BFS over
    out-edges, children enqueued by increasing out-degree. Unreached
    components are entered at their minimum-out-degree node, per C-M
    convention."""
    N = len(offsets) - 1
    deg = np.diff(offsets)
    visited = np.zeros(N, dtype=bool)
    order = np.empty(N, dtype=np.int64)
    pos = 0
    q = deque()
    visited[start] = True
    q.append(start)
    while pos < N:
        while q:
            u = q.popleft()
            order[pos] = u
            pos += 1
            nbrs = neighbors[offsets[u]:offsets[u + 1]]
            # unique also dedupes: rows may repeat a neighbor (faiss NSG does)
            fresh = np.unique(nbrs[~visited[nbrs]])
            if len(fresh):
                fresh = fresh[np.argsort(deg[fresh], kind="stable")]
                visited[fresh] = True
                q.extend(fresh.tolist())
        if pos < N:
            rest = np.flatnonzero(~visited)
            s = rest[np.argmin(deg[rest])]
            print(f"cm-dir: {N - pos} nodes unreached; continuing from {s}")
            visited[s] = True
            q.append(s)
    return order


def relabel(offsets, neighbors, perm):
    """Apply perm to rows and columns; rows come out sorted ascending."""
    N = len(offsets) - 1
    inv = np.empty(N, dtype=np.int64)
    inv[perm] = np.arange(N, dtype=np.int64)
    deg = np.diff(offsets)
    new_deg = deg[perm]
    new_offsets = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(new_deg, out=new_offsets[1:])
    edge_idx = np.concatenate(
        [np.arange(offsets[p], offsets[p + 1]) for p in perm]
    )
    new_neighbors = inv[neighbors[edge_idx]]
    row_ids = np.repeat(np.arange(N, dtype=np.int64), new_deg)
    order = np.argsort(row_ids * N + new_neighbors, kind="stable")
    return new_offsets, new_neighbors[order]


def gap_stats(offsets, neighbors):
    deg = np.diff(offsets)
    starts = offsets[:-1]
    diffs = np.diff(neighbors)
    row_start_mask = np.zeros(len(neighbors), dtype=bool)
    row_start_mask[starts[deg > 0]] = True
    inner = diffs[~row_start_mask[1:]]
    g = inner[inner > 0]
    log2g = np.floor(np.log2(g)).astype(int)
    return {
        "gap_mean": float(g.mean()),
        "gap_median": float(np.median(g)),
        "gap_log2_hist": np.bincount(log2g, minlength=20).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Input CSR prefix")
    parser.add_argument("--out-prefix", required=True, help="Output CSR prefix")
    parser.add_argument("--strategy", choices=["rcm", "bfs", "cm-dir"], required=True)
    parser.add_argument("--start", type=int, default=0,
                        help="Start node for bfs/cm-dir (e.g. the graph entry point)")
    args = parser.parse_args()

    offsets, neighbors = load_csr(args.prefix)

    t0 = time.time()
    if args.strategy == "rcm":
        perm = rcm_permutation(as_scipy(offsets, neighbors))
    elif args.strategy == "bfs":
        perm = bfs_permutation(as_scipy(offsets, neighbors), args.start)
    else:
        perm = cm_directed_permutation(offsets, neighbors, args.start)
    t_perm = time.time() - t0
    print(f"{args.strategy} permutation computed in {t_perm:.1f}s")

    t0 = time.time()
    new_offsets, new_neighbors = relabel(offsets, neighbors, perm)
    print(f"relabeled in {time.time() - t0:.1f}s")

    new_offsets.astype("<u8").tofile(f"{args.out_prefix}_offsets.bin")
    new_neighbors.astype("<u4").tofile(f"{args.out_prefix}_neighbors.bin")
    # perm[k] = old id that received new id k; needed to permute vectors and
    # map search results back to the original ids.
    perm.astype("<u4").tofile(f"{args.out_prefix}_perm.bin")

    stats = {"strategy": args.strategy, "perm_seconds": t_perm,
             **gap_stats(new_offsets, new_neighbors)}
    if args.strategy in ("bfs", "cm-dir"):
        stats["start"] = args.start
    with open(f"{args.out_prefix}_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "gap_log2_hist"},
                     indent=1))


if __name__ == "__main__":
    main()
