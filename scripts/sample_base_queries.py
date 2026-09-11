#!/usr/bin/env python3
"""Sample base vectors as a workload-free query set (with exact ground truth
computed by brute force), to test whether access-frequency profiles derived
without a real query log predict the hot set of the real queries."""
import argparse
import numpy as np, faiss

def read_fbin(p):
    with open(p, "rb") as f:
        n, d = np.fromfile(f, dtype="<u4", count=2)
        return np.fromfile(f, dtype="<f4").reshape(n, d)

def write_bin(p, a):
    with open(p, "wb") as f:
        np.array(a.shape, dtype="<u4").tofile(f); a.tofile(f)

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out-prefix", required=True)
ap.add_argument("--n", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--metric", default="l2"); ap.add_argument("--k", type=int, default=100)
ap.add_argument("--self-gt", action="store_true", help="skip brute force; gt = the sampled id repeated (profiling only)")
a = ap.parse_args()
X = read_fbin(a.base)
rng = np.random.default_rng(a.seed)
ids = rng.choice(len(X), a.n, replace=False)
Q = np.ascontiguousarray(X[ids])
if a.self_gt:
    I = np.repeat(ids[:, None], a.k, axis=1)
else:
    index = faiss.IndexFlatL2(X.shape[1]) if a.metric == "l2" else faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    _, I = index.search(Q, a.k)
write_bin(f"{a.out_prefix}_query.fbin", Q.astype("<f4"))
write_bin(f"{a.out_prefix}_gt.ibin", I.astype("<i4"))
print(f"wrote {a.n} sampled base queries + gt@{a.k} -> {a.out_prefix}_*")
