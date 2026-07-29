# CLAUDE.md — project handoff

Case study for Prof. Ferragina's course: compress the adjacency structure
of graph-based ANN indexes (DiskANN/Vamana, NSG, HNSW) with the
**U-GEF** encoder from the `external/gef` submodule, measure compression
ratio (compressed/raw — Ferragina's convention, smaller is better) and
bits/edge, and study how node-id relabelings change the result.

Read [REPORT.md](REPORT.md) for current results and
[RESEARCH_LOG.md](RESEARCH_LOG.md) for experiment history and open
directions before starting new work.

## State of play

- Pipeline is fully working end-to-end; all results in REPORT.md come
  from `scripts/run_experiments.py` (20 graph×labeling combos, all
  verified with exact round-trip reconstruction).
- Best result so far: NSG + Reverse Cuthill-McKee, total ratio 0.4788.
- `data/` is gitignored: every dataset, index, and CSR export must be
  regenerated (commands below and in README.md).

## Building on Linux (you are not on macOS)

This project was developed on macOS, which needed several workarounds you
can mostly ignore:

- `cmake -S . -B build && cmake --build build --target information_retrieval -- -j$(nproc)`
  should work directly with GCC. No compiler override needed (the
  `-DCMAKE_CXX_COMPILER=/usr/bin/clang++` trick in the README is for
  macOS only, where sdsl's CMake misdetects the default compiler path).
- The `if(APPLE)` libc++ defines in CMakeLists.txt won't fire; libstdc++
  still ships `std::result_of` and gef's `<experimental/simd>` include is
  glibc/libstdc++-compatible. If `<experimental/simd>` is missing on your
  toolchain, configure gef with `-DGEF_DISABLE_SIMD=ON` instead.
- The gef submodule contains one **uncommitted local patch** on the Mac
  (Clang-only compiler-flag fix in its CMakeLists). A fresh
  `git submodule update --init --recursive` gives you pristine upstream,
  which builds fine under GCC — don't look for the patch.
- `diskannpy` has manylinux x86_64 wheels: `pip install -r scripts/requirements.txt`
  works natively — no Docker/Colima needed (that whole README section is
  macOS-only). On aarch64 Linux, diskannpy has no wheels; use an x86_64
  box or container.

## Pipeline (native Linux)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r scripts/requirements.txt scipy
curl -L "http://ann-benchmarks.com/sift-128-euclidean.hdf5" -o ./data/sift-128-euclidean.hdf5

# graphs (DiskANN ~5-25 min depending on machine; NSG ~10 min; HNSW ~3 min)
python scripts/diskann-index.py --input ./data/sift-128-euclidean.hdf5 --index-dir ./data/sift_diskann_index            # R=32 default; --R 16/64 with --index-dir ..._R16/_R64
python scripts/export_diskann_graph.py --index ./data/sift_diskann_index/sift_diskann --prefix ./data/graph
python scripts/nsg-index.py --input ./data/sift-128-euclidean.hdf5 --output ./data/sift_nsg.index
python scripts/export_nsg_graph.py --index ./data/sift_nsg.index --prefix ./data/graph_nsg
python scripts/hnsw-index.py --input ./data/sift-128-euclidean.hdf5 --prefix ./data/graph_hnsw

# the full experiment matrix
python scripts/run_experiments.py --binary ./build/information_retrieval \
    --out ./data/sweep_results.json \
    ./data/graph ./data/graph_R16 ./data/graph_R64 ./data/graph_nsg ./data/graph_hnsw
```

## Facts that will save you time

- CSR format: `<prefix>_offsets.bin` = N+1 little-endian uint64;
  `<prefix>_neighbors.bin` = nnz little-endian uint32, rows sorted
  ascending (U-GEF wants sorted; `--no-sort` exists for B_GEF variants).
  `<prefix>_meta.json` carries the entry point under `"start"`.
- The graphs are **directed** (72.7% of DiskANN edges reciprocated). RCM
  symmetrizes only to compute its permutation.
- DiskANN and faiss NSG independently pick the dataset medoid as entry
  point (node 123742 on SIFT-1M) — identical `start` values are correct,
  not a bug.
- faiss NSG adjacency rows contain duplicate neighbor ids and −1 padding;
  both are handled (export filters to [0,N), cm-dir dedupes). Remember
  this if you write new graph-traversal code.
- HNSW export is the base layer only (every node, degree ≤ 2M); upper
  layers are search accelerators, not part of the compressed structure.
- The order-oblivious bound Σ log₂C(N,degᵢ) is labeling-invariant. A
  structure-aware labeling can legitimately compress *below* it
  (NSG/HNSW + RCM do).
- `information_retrieval <prefix> [--approximate]` verifies exact
  reconstruction on every run; treat any verification failure as a data
  bug, not noise.

## Conventions

- **One logically separate change per commit** — never bundle (e.g. a fix
  and a doc update). Antonio insists on this.
- Commits on the Mac are authored as `Antonio Napolitano <anton@polit.no>`
  and GPG-signed with a smartcard-backed key. On another machine that key
  is unavailable: ask Antonio how to sign (or whether to skip `-S`)
  before committing.
- Charts in `docs/charts/` are hand-generated SVGs (light/dark aware via
  `prefers-color-scheme`), embedded in REPORT.md. Regenerate in the same
  style if numbers change; render and eyeball them before committing.
- REPORT.md carries the numbers; RESEARCH_LOG.md the narrative and the
  queue of future directions. Keep both current as experiments land.
