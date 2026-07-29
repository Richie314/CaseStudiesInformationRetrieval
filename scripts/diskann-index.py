#!/usr/bin/env python3

import argparse
import os
import time

import diskannpy
import h5py
import numpy as np


def load_vectors(hdf5_path: str, dataset_key: str = "train") -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(
                f"Dataset key '{dataset_key}' not found in {hdf5_path}. "
                f"Available keys: {list(f.keys())}"
            )
        vectors = f[dataset_key][:]
    return np.ascontiguousarray(vectors, dtype=np.float32)


def build_diskann_index(
    vectors: np.ndarray,
    index_dir: str,
    R: int = 32,
    L: int = 64,
    alpha: float = 1.2,
    num_threads: int = 0,
    metric: str = "l2",
):
    
    os.makedirs(index_dir, exist_ok=True)
    n, d = vectors.shape
    print(f"Building DiskANN (Vamana) index: N={n}, d={d}, R={R}, L={L}, alpha={alpha}")

    t0 = time.time()
    diskannpy.build_memory_index(
        data=vectors,
        distance_metric=metric,          # "l2" or "cosine" or "mips"
        index_directory=index_dir,
        complexity=L,                     # Lbuild
        graph_degree=R,
        alpha=alpha,
        num_threads=num_threads,           # 0 == use all cores
        index_prefix="sift_diskann",
    )
    print(f"Index built and written to '{index_dir}' in {time.time() - t0:.2f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to SIFT .hdf5 file")
    parser.add_argument("--dataset-key", default="train")
    parser.add_argument("--index-dir", default="./data/sift_diskann_index")
    parser.add_argument("--R", type=int, default=32)
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--metric", choices=["l2", "cosine", "mips"], default="l2")
    parser.add_argument("--num-threads", type=int, default=0)
    args = parser.parse_args()

    vectors = load_vectors(args.input, args.dataset_key)
    build_diskann_index(
        vectors,
        index_dir=args.index_dir,
        R=args.R,
        L=args.L,
        alpha=args.alpha,
        num_threads=args.num_threads,
        metric=args.metric,
    )
    idx = diskannpy.StaticMemoryIndex(
        index_directory=args.index_dir,
        num_threads=args.num_threads,
        initial_search_complexity=args.L,
        index_prefix="sift_diskann",
    )

    # Test
    ids, dists = idx.search(vectors[0], k_neighbors=5, complexity=args.L)
    print("Sanity check (query = base vector 0):")
    print("IDs:", ids)
    print("Distances:", dists)


if __name__ == "__main__":
    main()