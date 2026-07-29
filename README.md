# Case Studies in Information Retrieval — Compressing ANN Proximity Graphs with GEF

This project studies how well **Generalized Elias-Fano (GEF)** compresses the
adjacency lists of graph-based Approximate Nearest Neighbor (ANN) indexes.
A proximity graph is built in Python over the SIFT-1M dataset (DiskANN/Vamana,
or optionally NSG via faiss), exported as a raw CSR structure, and then
compressed in C++ with the **U-GEF** algorithm from the
[generalized-elias-fano](https://github.com/mxpucci/generalized-elias-fano)
library. The program measures the compression ratio and verifies that every
adjacency list reconstructs exactly.

Results for SIFT-1M are summarized in [REPORT.md](REPORT.md).

## Repository layout

| Path | Purpose |
|------|---------|
| [src/main.cpp](src/main.cpp) | Loads the CSR graph, compresses it with `gef::U_GEF`, prints ratios, verifies a full round-trip, serializes the compressed structures |
| [scripts/diskann-index.py](scripts/diskann-index.py) | Builds an in-memory DiskANN (Vamana) index over an ann-benchmarks HDF5 file |
| [scripts/export_diskann_graph.py](scripts/export_diskann_graph.py) | Parses the DiskANN graph file and writes it as raw CSR (`*_offsets.bin`, `*_neighbors.bin`) |
| [scripts/relabel_graph.py](scripts/relabel_graph.py) | Bandwidth-reducing id relabelings (Reverse Cuthill-McKee, BFS order, directed C-M) for better compression |
| [scripts/nsg-index.py](scripts/nsg-index.py) | Alternative graph builder: faiss NSG |
| [scripts/hnsw-index.py](scripts/hnsw-index.py) | Alternative graph builder: faiss HNSW, exports the base layer as CSR |
| [scripts/run_experiments.py](scripts/run_experiments.py) | Sweep driver: every relabeling × every graph → JSON + markdown results |
| [scripts/export_nsg_graph.py](scripts/export_nsg_graph.py) | CSR export for the faiss NSG graph |
| [external/gef](external/gef) | Git submodule: the GEF compression library (C++20) |
| `data/` | Datasets, indexes, and exported graphs (git-ignored) |

## The pipeline

```
SIFT-1M (hdf5) ──▶ diskann-index.py ──▶ Vamana graph ──▶ export_diskann_graph.py
                                                              │
                                          graph_offsets.bin (uint64 CSR row pointers)
                                          graph_neighbors.bin (uint32 neighbor ids)
                                                              │
                                                              ▼
                                        information_retrieval (C++, gef::U_GEF)
                                                              │
                                    compression ratios + graph_*.gef serialized output
```

Neighbor lists are sorted ascending per node during export (the default):
U-GEF is an Elias-Fano-style encoder and expects each compressed row to be
retrieved as written; sorted rows compress markedly better. Use
`--no-sort` only if you plan to switch the C++ side to `B_GEF`/`B_STAR_GEF`.

### CSR export format

- `<prefix>_offsets.bin` — `N+1` little-endian `uint64` values; node `i`'s
  neighbors live at positions `[offsets[i], offsets[i+1])`.
- `<prefix>_neighbors.bin` — `nnz` little-endian `uint32` neighbor ids.
- `<prefix>_meta.json` — graph metadata (entry point, max degree, …).

## Getting started

Clone with the submodule:

```bash
git clone --recurse-submodules <this-repo>
```

(or `git submodule update --init --recursive` in an existing checkout).

### 1. Python environment — building the graph

`diskannpy` ships wheels **only for Linux x86_64 and Windows**. On Linux the
native venv works:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r ./scripts/requirements.txt
```

On **macOS (Apple Silicon)** run the Python side in an x86_64 Linux container
instead (Rosetta makes this near-native speed):

```bash
brew install colima docker qemu lima-additional-guestagents
colima start --arch x86_64 --vm-type vz --vz-rosetta --cpu 8 --memory 8
docker run -d --name diskann-env --platform linux/amd64 \
  -v "$PWD":/work -w /work python:3.11-slim sleep infinity
docker exec diskann-env pip install diskannpy numpy h5py
```

### 2. Dataset, index, export

```bash
curl -L "http://ann-benchmarks.com/sift-128-euclidean.hdf5" -o "./data/sift-128-euclidean.hdf5"
```

Natively on Linux:

```bash
python3 ./scripts/diskann-index.py --input ./data/sift-128-euclidean.hdf5 --index-dir ./data/sift_diskann_index
python3 ./scripts/export_diskann_graph.py --index ./data/sift_diskann_index/sift_diskann --prefix ./data/graph
```

Via the macOS container (paths under `/work`):

```bash
docker exec diskann-env python /work/scripts/diskann-index.py --input /work/data/sift-128-euclidean.hdf5 --index-dir /work/data/sift_diskann_index
docker exec -w /work/data diskann-env python /work/scripts/export_diskann_graph.py --index /work/data/sift_diskann_index/sift_diskann --prefix graph
```

Defaults: `R=32` (max out-degree), `L=64` (build complexity), `alpha=1.2`, L2 metric.

### 3. Build and run the compressor (C++)

```bash
cmake -S . -B build
cmake --build build --target information_retrieval -- -j8
./build/information_retrieval ./data/graph
```

Pass `--approximate` as a second argument to use the approximate split-point
strategy (faster compression, potentially slightly worse ratio) instead of the
optimal one.

The program prints raw vs. compressed sizes and ratios for the offsets array,
the neighbors array, and their total; verifies that all `N` adjacency lists
decompress to exactly the original input; and writes `graph_offsets.gef` /
`graph_neighbors.gef` next to the input files.

#### macOS build notes

- On macOS configure with `-DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++`
  the first time: sdsl-lite's legacy CMake misdetects the default `c++`
  driver path and falls into its MSVC branch.
- [CMakeLists.txt](CMakeLists.txt) already defines
  `_LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS` (sdsl-lite uses `std::result_of`,
  removed from libc++ in C++20) and `_LIBCPP_ENABLE_EXPERIMENTAL`
  (gef uses `<experimental/simd>`) when building on Apple platforms.
- The gef submodule carries a one-line local patch replacing the GCC-only
  `-fvect-cost-model=dynamic` flag on Clang, plus sdsl-lite master has a
  member-name typo in `louds_tree.hpp` (`m_select1` → `m_bv_select1`) that
  strict AppleClang rejects; it is patched in the fetched sources under
  `build/_deps` after the first configure. Neither affects Linux/GCC builds.

## What main.cpp measures

`gef::U_GEF<T>` partitions the input (default partition size 32,000 elements)
and encodes each partition with a generalized Elias-Fano scheme that splits
each value into low/high bits at a chosen split point — chosen optimally per
partition (`OPTIMAL_SPLIT_POINT`, default) or heuristically
(`APPROXIMATE_SPLIT_POINT`, with `--approximate`). Compressed size is reported
by `size_in_bytes()` (including rank/select support structures), and random
access is exercised via `operator[]` (offsets) and `get_elements`
(contiguous neighbor ranges) during verification.
