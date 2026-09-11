#!/usr/bin/env python3
"""Build an in-memory DiskANN (Vamana) index directly from a .fbin file."""
import argparse, os, time
import numpy as np, diskannpy
ap = argparse.ArgumentParser()
ap.add_argument("--fbin", required=True); ap.add_argument("--index-dir", required=True)
ap.add_argument("--prefix", required=True); ap.add_argument("--R", type=int, default=32)
ap.add_argument("--L", type=int, default=64); ap.add_argument("--alpha", type=float, default=1.2)
ap.add_argument("--metric", default="l2"); ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()
os.makedirs(a.index_dir, exist_ok=True)
t0 = time.time()
diskannpy.build_memory_index(data=a.fbin, vector_dtype=np.float32, distance_metric=a.metric,
                             index_directory=a.index_dir, complexity=a.L, graph_degree=a.R,
                             alpha=a.alpha, num_threads=a.threads, index_prefix=a.prefix)
print(f"built {a.prefix} R={a.R} L={a.L} alpha={a.alpha} in {time.time()-t0:.0f}s", flush=True)
