#!/usr/bin/env python3

import argparse
import time

import faiss
import h5py
import numpy as np


def load_vectors(hdf5_path: str, dataset_key: str = "train", normalize: bool = False) -> np.ndarray:
    """Load the base vector set from an ann-benchmarks-style hdf5 file."""
    with h5py.File(hdf5_path, "r") as f:
        if dataset_key not in f:
            available = list(f.keys())
            raise KeyError(
                f"Dataset key '{dataset_key}' not found in {hdf5_path}. "
                f"Available keys: {available}"
            )
        vectors = f[dataset_key][:]
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    return vectors


def build_nsg_index(
    vectors: np.ndarray,
    R: int = 32,
    L: int = 64,
    C: int = 100,
    metric: str = "l2",
) -> faiss.IndexNSGFlat:
    """
    Build a faiss IndexNSG over `vectors`.

    Parameters
    ----------
    vectors : (N, d) float32 array
    R : int
        Max out-degree of each node in the final NSG graph
        (faiss param: GK / index.nsg.R). Analogous to DiskANN's R.
    L : int
        Search list size used while probing during construction. Analogous
        to DiskANN's Lbuild / L search width.
    C : int
        Candidate pool size limit during graph construction
        (faiss param: index.nsg.search_L / GK-related knobs vary by version;
        C controls the KNN graph fan-out fed into NSG).
    metric : {"l2", "ip"}
        Distance metric.
    """
    n, d = vectors.shape
    faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT

    # IndexNSGFlat needs a KNN graph as a starting point. Faiss builds this
    # internally via `GK` (the K of the initial kNN graph) if you construct
    # it directly; expose it here so it's tunable.
    index = faiss.IndexNSGFlat(d, R, faiss_metric)
    index.nsg.search_L = L        # search width during construction/search
    index.GK = C                  # K of the initial KNN graph fed to NSG

    print(f"Building NSG index: N={n}, d={d}, R={R}, L={L}, GK(C)={C}, metric={metric}")
    t0 = time.time()
    index.add(vectors)
    t1 = time.time()
    print(f"Index built in {t1 - t0:.2f}s")

    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to SIFT .hdf5 file")
    parser.add_argument("--dataset-key", default="train",
                         help="hdf5 dataset key holding the base vectors (default: 'train')")
    parser.add_argument("--output", default="sift_nsg.index", help="Output faiss index path")
    parser.add_argument("--R", type=int, default=32, help="Max graph out-degree")
    parser.add_argument("--L", type=int, default=64, help="Search list width during construction")
    parser.add_argument("--C", type=int, default=100, help="K of the initial KNN graph (GK)")
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")
    parser.add_argument("--normalize", action="store_true", help="L2-normalise vectors (angular datasets)")
    args = parser.parse_args()

    vectors = load_vectors(args.input, args.dataset_key, args.normalize)
    index = build_nsg_index(vectors, R=args.R, L=args.L, C=args.C, metric=args.metric)

    faiss.write_index(index, args.output)
    print(f"Saved index to {args.output}")

    # Quick sanity check: search with the first few base vectors as queries.
    k = 5
    D, I = index.search(vectors[:3], k)
    print("Sanity check search results (should include the query itself at rank 0):")
    print("Indices:\n", I)
    print("Distances:\n", D)


if __name__ == "__main__":
    main()