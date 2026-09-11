#!/usr/bin/env python3
"""Build a faiss HNSW index over an ann-benchmarks hdf5 file and export
its base layer (level 0) as a CSR graph for compression.

HNSW's base layer holds every node with max degree 2*M, making it the
natural counterpart of DiskANN's and NSG's flat R-bounded graphs; the
upper layers hold <2% of the edges and only accelerate entry-point
descent, so they are not exported.
"""

import argparse
import json
import time

import faiss
import h5py
import numpy as np


def load_vectors(hdf5_path: str, dataset_key: str = "train", normalize: bool = False) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(
                f"Dataset key '{dataset_key}' not found in {hdf5_path}. "
                f"Available keys: {list(f.keys())}"
            )
        vectors = f[dataset_key][:]
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if normalize:  # angular datasets: cosine ordering == inner-product ordering on unit vectors
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors


def build_hnsw(vectors: np.ndarray, M: int, ef_construction: int,
               metric: str) -> faiss.IndexHNSWFlat:
    n, d = vectors.shape
    faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
    index = faiss.IndexHNSWFlat(d, M, faiss_metric)
    index.hnsw.efConstruction = ef_construction
    print(f"Building HNSW index: N={n}, d={d}, M={M} (base degree <= {2 * M}), "
          f"efConstruction={ef_construction}, metric={metric}")
    t0 = time.time()
    index.add(vectors)
    print(f"Index built in {time.time() - t0:.2f}s")
    return index


def export_base_layer(index, prefix: str, sort_rows: bool = True):
    hnsw = index.hnsw
    N = index.ntotal
    offsets = faiss.vector_to_array(hnsw.offsets)          # (N+1,) int64
    neighbors = faiss.vector_to_array(hnsw.neighbors)      # flat, -1 padding
    cum = faiss.vector_to_array(hnsw.cum_nneighbor_per_level)
    level0_width = int(cum[1] - cum[0])                    # == 2*M

    row_ptr = np.zeros(N + 1, dtype=np.uint64)
    rows = []
    total = 0
    for i in range(N):
        row = neighbors[offsets[i]:offsets[i] + level0_width]
        valid = row[row >= 0].astype(np.uint32)
        if sort_rows:
            valid = np.sort(valid)
        rows.append(valid)
        total += len(valid)
        row_ptr[i + 1] = total
    flat = np.concatenate(rows) if rows else np.empty(0, dtype=np.uint32)

    row_ptr.tofile(f"{prefix}_offsets.bin")
    flat.tofile(f"{prefix}_neighbors.bin")
    meta = {
        "N": int(N),
        "nnz": int(total),
        "start": int(hnsw.entry_point),
        "max_level": int(hnsw.max_level),
        "level0_max_degree": level0_width,
        "sorted": sort_rows,
    }
    with open(f"{prefix}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"CSR built: nnz={total}, avg out-degree={total / N:.2f}, "
          f"entry point={meta['start']}")
    print(f"Wrote {prefix}_offsets.bin ({row_ptr.nbytes} bytes raw)")
    print(f"Wrote {prefix}_neighbors.bin ({flat.nbytes} bytes raw)")
    print(f"Wrote {prefix}_meta.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to SIFT .hdf5 file")
    parser.add_argument("--dataset-key", default="train")
    parser.add_argument("--prefix", default="./data/graph_hnsw",
                        help="Output CSR prefix")
    parser.add_argument("--M", type=int, default=16,
                        help="HNSW M; base-layer degree cap is 2*M")
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")
    parser.add_argument("--normalize", action="store_true", help="L2-normalise vectors (angular datasets)")
    parser.add_argument("--no-sort", action="store_true",
                        help="Keep original neighbor order")
    args = parser.parse_args()

    vectors = load_vectors(args.input, args.dataset_key, args.normalize)
    index = build_hnsw(vectors, args.M, args.ef_construction, args.metric)
    export_base_layer(index, args.prefix, sort_rows=not args.no_sort)

    D, I = index.search(vectors[:3], 5)
    print("Sanity check (first 3 base vectors, should self-match at rank 0):")
    print(I)


if __name__ == "__main__":
    main()
