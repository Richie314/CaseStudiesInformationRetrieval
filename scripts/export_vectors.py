#!/usr/bin/env python3
"""Dump an ann-benchmarks HDF5 file to DiskANN-style flat binaries for the
C++ search benchmark:

  <prefix>_base.fbin    uint32 N, uint32 d, then N*d float32 (train set)
  <prefix>_query.fbin   uint32 Q, uint32 d, then Q*d float32 (test set)
  <prefix>_gt.ibin      uint32 Q, uint32 K, then Q*K int32 true neighbor ids

For angular datasets the vectors are L2-normalised so that inner product
ordering equals cosine ordering (the ground truth in the file was computed
with angular distance, which is monotone in cosine).
"""

import argparse

import h5py
import numpy as np


def write_bin(path: str, arr: np.ndarray):
    n, d = arr.shape
    with open(path, "wb") as f:
        np.array([n, d], dtype="<u4").tofile(f)
        arr.tofile(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalise base and query vectors (angular datasets)")
    ap.add_argument("--gt-k", type=int, default=100)
    args = ap.parse_args()

    with h5py.File(args.input, "r") as f:
        base = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        query = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        gt = np.ascontiguousarray(f["neighbors"][:, : args.gt_k], dtype="<i4")
        print(f"keys={list(f.keys())} distance={f.attrs.get('distance')}")

    if args.normalize:
        base /= np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)

    write_bin(f"{args.prefix}_base.fbin", base.astype("<f4"))
    write_bin(f"{args.prefix}_query.fbin", query.astype("<f4"))
    write_bin(f"{args.prefix}_gt.ibin", gt)
    print(f"base {base.shape} query {query.shape} gt {gt.shape} -> {args.prefix}_*")


if __name__ == "__main__":
    main()
