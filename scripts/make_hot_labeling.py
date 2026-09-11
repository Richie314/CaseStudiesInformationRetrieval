#!/usr/bin/env python3
"""Build a *hot-first* labeling for the two-tier (hybrid) graph store.

Given a base-labeled CSR graph (e.g. the BFS relabeling), its perm (new->
original ids) and an expansion-count profile in the base id space, choose a
hot set of h nodes by a policy and emit a new labeling in which the hot
nodes take ids [0, h) (in decreasing frequency) and the cold nodes keep
their relative base order after them. This lets the hybrid store test tiers
with `u < h` and keeps the hot rows contiguous in memory.

Policies: freq (expansion count), indeg, outdeg, random. The BFS-level
policy needs no relabeling on a BFS-labeled graph (ids [0,h) already are
the h nodes closest to the entry point in BFS order).

Also writes <out-prefix>_hot.u32: the hot ids in the *base* id space, for
search_bench --hot-ids on the base labeling.
"""

import argparse
import json

import numpy as np

from relabel_graph import relabel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", required=True, help="base-labeled CSR prefix")
    ap.add_argument("--perm", required=True, help="base perm (new->orig), uint32")
    ap.add_argument("--expand", required=True, help="expansion counts in the base id space")
    ap.add_argument("--policy", choices=["freq", "indeg", "outdeg", "random"], default="freq")
    ap.add_argument("--hot-count", type=int, required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    off = np.fromfile(f"{args.prefix}_offsets.bin", dtype="<u8").astype(np.int64)
    nbr = np.fromfile(f"{args.prefix}_neighbors.bin", dtype="<u4").astype(np.int64)
    base_perm = np.fromfile(args.perm, dtype="<u4").astype(np.int64)
    N = len(off) - 1
    cnt = np.fromfile(args.expand, dtype="<u4").astype(np.int64)
    assert len(cnt) == N == len(base_perm)

    if args.policy == "freq":
        score = cnt
    elif args.policy == "indeg":
        score = np.bincount(nbr, minlength=N)
    elif args.policy == "outdeg":
        score = np.diff(off)
    else:
        score = np.random.default_rng(args.seed).permutation(N)
    order = np.argsort(-score, kind="stable")
    hot = order[: args.hot_count]
    is_hot = np.zeros(N, dtype=bool)
    is_hot[hot] = True
    cold = np.flatnonzero(~is_hot)            # keeps base order
    perm_base = np.concatenate([hot, cold])   # new id -> base id

    new_off, new_nbr = relabel(off, nbr, perm_base)
    new_off.astype("<u8").tofile(f"{args.out_prefix}_offsets.bin")
    new_nbr.astype("<u4").tofile(f"{args.out_prefix}_neighbors.bin")
    base_perm[perm_base].astype("<u4").tofile(f"{args.out_prefix}_perm.bin")   # new -> orig
    hot.astype("<u4").tofile(f"{args.out_prefix}_hot.u32")
    share = float(cnt[hot].sum() / max(1, cnt.sum()))
    meta = {"policy": args.policy, "hot_count": args.hot_count, "train_share": share,
            "base_prefix": args.prefix}
    with open(f"{args.out_prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
