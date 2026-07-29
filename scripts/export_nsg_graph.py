#!/usr/bin/env python3

import argparse
import json

import faiss
import numpy as np


def extract_nsg_adjacency(index) -> tuple[np.ndarray, np.ndarray]:
    """
    Pull the raw (N, K) adjacency matrix out of a faiss NSG-family index.

    Returns
    -------
    adj : (N, K) int32 array. adj[i, j] is the j-th neighbor of node i, or
          a value >= N for an unused/padding slot (out-degree < K).
    N   : number of nodes (== index.ntotal)
    """
    if not hasattr(index, "nsg"):
        raise TypeError(
            f"{type(index)} has no `.nsg` attribute. This script expects an "
            "IndexNSGFlat / IndexNSGPQ / IndexNSGSQ built and trained with faiss."
        )
    g = index.nsg.get_final_graph()
    N, K = g.N, g.K
    adj = faiss.rev_swig_ptr(g.data, N * K).reshape(N, K).copy()
    return adj, N


def adjacency_to_csr(adj: np.ndarray, N: int, sort_rows: bool = True):
    """
    Convert the fixed-width (N, K) adjacency (with >= N sentinel padding)
    into a variable-degree CSR representation: row_ptr (N+1,) uint64 and
    neighbors (nnz,) uint32.
    """
    row_ptr = np.zeros(N + 1, dtype=np.uint64)
    rows = []
    total = 0
    for i in range(N):
        row = adj[i]
        valid = row[row < N].astype(np.uint32)
        if sort_rows:
            valid = np.sort(valid)
        rows.append(valid)
        total += len(valid)
        row_ptr[i + 1] = total
    neighbors = np.concatenate(rows) if rows else np.empty(0, dtype=np.uint32)
    return row_ptr, neighbors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Path to a faiss NSG index file")
    parser.add_argument("--prefix", default="graph", help="Output file prefix")
    parser.add_argument("--no-sort", action="store_true",
                         help="Keep original neighbor order (use B_GEF/B_STAR_GEF downstream instead of U_GEF)")
    args = parser.parse_args()

    index = faiss.read_index(args.index)
    adj, N = extract_nsg_adjacency(index)
    print(f"Loaded index: N={N}, max out-degree K={adj.shape[1]}")

    row_ptr, neighbors = adjacency_to_csr(adj, N, sort_rows=not args.no_sort)
    nnz = len(neighbors)
    avg_degree = nnz / N if N else 0
    print(f"CSR built: nnz={nnz}, avg out-degree={avg_degree:.2f}")

    row_ptr.tofile(f"{args.prefix}_offsets.bin")
    neighbors.tofile(f"{args.prefix}_neighbors.bin")
    with open(f"{args.prefix}_meta.json", "w") as f:
        json.dump({"N": int(N), "nnz": int(nnz), "sorted": not args.no_sort}, f, indent=2)

    print(f"Wrote {args.prefix}_offsets.bin ({row_ptr.nbytes} bytes raw)")
    print(f"Wrote {args.prefix}_neighbors.bin ({neighbors.nbytes} bytes raw)")
    print(f"Wrote {args.prefix}_meta.json")


if __name__ == "__main__":
    main()