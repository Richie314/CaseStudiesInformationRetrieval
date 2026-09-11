#!/usr/bin/env python3
"""Build navigation layers ("upper layers") over a flat base graph so that
search_bench can descend HNSW-style before running beam search on layer 0.

The node sets are nested, S_1 ⊃ S_2 ⊃ ..., chosen by one of
  random   uniform sample (HNSW's coin flip, in expectation)
  indeg    highest in-degree in the base graph (HM-ANN-style hub promotion)
  freq     highest measured expansion count (access-biased, biased-skip-list
           style: heavy items get promoted)
with sizes given explicitly (e.g. N/16, N/256 to mimic M=16 HNSW). Each
layer's graph is the base layer of a faiss HNSW built on the layer's vectors
(a proper navigable graph on the subset), exported in layer-local ids.

Ids are in the *graph's* id space (after any relabeling): --perm maps them
to the original vector ids. Writes <out>_hier.json listing, top level first,
{n, nodes(u32 graph ids), offsets(u64), neighbors(u32)} and the entry
(local id in the top level = the medoid of that level).
"""

import argparse
import json

import faiss
import numpy as np


def read_fbin(p):
    with open(p, "rb") as f:
        n, d = np.fromfile(f, dtype="<u4", count=2)
        return np.fromfile(f, dtype="<f4").reshape(n, d)


def layer_graph(X, M, efc, metric):
    index = faiss.IndexHNSWFlat(X.shape[1], M, faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efc
    index.add(X)
    h = index.hnsw
    offsets = faiss.vector_to_array(h.offsets)
    neighbors = faiss.vector_to_array(h.neighbors)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level)
    w = int(cum[1] - cum[0])
    n = X.shape[0]
    rows = []
    for i in range(n):
        r = neighbors[offsets[i]:offsets[i] + w]
        rows.append(np.sort(r[r >= 0]).astype(np.uint32))
    off = np.zeros(n + 1, dtype=np.uint64)
    off[1:] = np.cumsum([len(r) for r in rows])
    return off, np.concatenate(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True, help="base CSR prefix (graph id space)")
    ap.add_argument("--perm", required=True)
    ap.add_argument("--base", required=True, help="vectors .fbin (original ids)")
    ap.add_argument("--policy", choices=["random", "indeg", "freq"], required=True)
    ap.add_argument("--expand", help="expansion counts (graph id space) for policy=freq")
    ap.add_argument("--sizes", required=True, help="comma list, largest first, e.g. 62500,3906,244")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--metric", default="l2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    off = np.fromfile(f"{args.graph}_offsets.bin", dtype="<u8").astype(np.int64)
    nbr = np.fromfile(f"{args.graph}_neighbors.bin", dtype="<u4").astype(np.int64)
    perm = np.fromfile(args.perm, dtype="<u4").astype(np.int64)
    N = len(off) - 1
    X = read_fbin(args.base)
    assert len(X) == N

    if args.policy == "random":
        order = np.random.default_rng(args.seed).permutation(N)
    elif args.policy == "indeg":
        order = np.argsort(-np.bincount(nbr, minlength=N), kind="stable")
    else:
        cnt = np.fromfile(args.expand, dtype="<u4").astype(np.int64)
        order = np.argsort(-cnt, kind="stable")

    sizes = [int(s) for s in args.sizes.split(",")]
    levels = []
    for li, n in enumerate(sizes):
        S = np.sort(order[:n])                    # nested: prefix of the same ranking
        XS = np.ascontiguousarray(X[perm[S]])
        loff, lnbr = layer_graph(XS, args.M, args.efc, args.metric)
        S.astype("<u4").tofile(f"{args.out}_L{li}_nodes.u32")
        loff.astype("<u8").tofile(f"{args.out}_L{li}_offsets.bin")
        lnbr.astype("<u4").tofile(f"{args.out}_L{li}_neighbors.bin")
        levels.append({"n": int(n), "nodes": f"{args.out}_L{li}_nodes.u32",
                       "offsets": f"{args.out}_L{li}_offsets.bin",
                       "neighbors": f"{args.out}_L{li}_neighbors.bin",
                       "edges": int(len(lnbr))})
        print(f"level {li}: n={n} edges={len(lnbr)} avgdeg={len(lnbr)/n:.1f}")
    # entry: medoid of the top (smallest) level
    top = np.sort(order[:sizes[-1]])
    XT = X[perm[top]]
    entry = int(np.argmin(((XT - XT.mean(0)) ** 2).sum(1)))
    hier = {"policy": args.policy, "levels": levels[::-1], "entry": entry}
    with open(f"{args.out}_hier.json", "w") as f:
        json.dump(hier, f, indent=1)
    print(json.dumps({k: v for k, v in hier.items() if k != "levels"}))


if __name__ == "__main__":
    main()
