# Research log

Chronological record of the experiments in this case study: compressing
graph-based ANN indexes with Generalized Elias-Fano (U-GEF). Numbers in
[REPORT.md](REPORT.md); this file records what was tried, in what order,
what came out, and what to try next.

## 1. Toolchain bring-up (2026-07-29)

Goal: compile [src/main.cpp](src/main.cpp) (uses `gef::U_GEF`) on macOS.
Four independent obstacles, all fixed and documented in the README:

1. gef's CMake passes GCC-only `-fvect-cost-model=dynamic` to any compiler
   matching `Clang` — patched in the submodule (local, uncommitted there).
2. sdsl-lite's legacy CMake misdetects AppleClang via the compiler *path*
   and falls into its MSVC branch (`process.h` not found) — fixed by
   configuring with explicit `-DCMAKE_CXX_COMPILER=/usr/bin/clang++`.
3. libc++ removed `std::result_of` in C++20 (sdsl uses it) and gates
   `<experimental/simd>` (gef uses it) — two compile definitions added in
   the top-level CMakeLists, Apple-only.
4. sdsl-lite master has a member-name typo in `louds_tree.hpp` swap()
   (`m_select1` vs `m_bv_select1`) that strict AppleClang rejects —
   patched in the FetchContent copy under `build/_deps`.

## 2. Baseline: DiskANN R=32 on SIFT-1M

- diskannpy has no macOS wheels → built the index in an x86_64 Linux
  container (Colima + Rosetta). Build: R=32, L=64, α=1.2 → 1M nodes,
  27.8M directed edges, avg degree 27.8, built in 1490 s.
- Exported as CSR (uint64 offsets / uint32 neighbors, rows sorted).
- U-GEF (optimal split): **total ratio 0.5317, neighbors 17.95 bits/edge**;
  approximate split: 0.5385 at ~3× faster construction. Round-trip exact.
- Analysis: within-row gaps match uniform scattering (median ≈ N/degree),
  giving an order-oblivious bound of Σ log₂C(N,degᵢ) = **16.41 b/e**;
  U-GEF sits 9.4% above it. Conclusion: the labeling, not the codec, is
  where headroom lives.
- Also: hardened main.cpp validation; adopted Ferragina's ratio convention
  (compressed/raw) and bits/edge as metrics.

## 3. Bandwidth-reducing relabelings (Prof. Ferragina's suggestion)

Idea: permute rows/columns to cluster ones near the diagonal (bandwidth
minimization) so Elias-Fano sees smaller gaps.

- **Reverse Cuthill-McKee** (scipy, on symmetrized A+Aᵀ): 0.5253 —
  barely helps on DiskANN.
- **BFS order from the entry point** (medoid, node 123742): **0.5047**,
  neighbors 17.03 b/e (+3.8% over bound). 1.09M gaps collapse to exactly 1;
  BFS over a metric graph enumerates the space region-by-region.
- Reciprocity check: the graph is directed; 72.7% of edges reciprocated.

## 4. Sweep: R ∈ {16, 32, 64} × {DiskANN, NSG, HNSW} × 4 labelings

Added: directed C-M (out-edge BFS with C-M's by-degree tie-breaking, since
no standard directed C-M exists), faiss NSG and faiss HNSW (base layer)
builders, and a sweep driver. NSG/HNSW built natively (faiss-cpu has arm64
wheels); NSG and DiskANN independently picked the same medoid entry point.

Bugs found and fixed during the sweep:
- NSG export kept faiss's −1 padding as neighbor id 2³²−1 (int32 `row < N`
  passes for −1).
- cm-dir overflowed its BFS queue on duplicate neighbor ids — faiss NSG
  rows contain ~124k duplicates; deduped with `np.unique`.

Results (neighbors b/e; full table in REPORT.md):
- Denser graphs compress relatively better: DiskANN orig 18.71 → 17.43
  as R goes 16 → 64; α-pruning saturates at avg degree 38 even at R=64.
- **Winner depends on the construction**: BFS/cm-dir best on DiskANN
  (16.36 b/e at R=64); classic RCM best on NSG (16.43) and HNSW (16.55).
- **NSG+RCM and HNSW+RCM beat their order-oblivious bounds** (16.73/16.66)
  — the permutation demonstrably encodes metric structure.
- Best overall: **NSG + RCM, total ratio 0.4788**.
- Directed C-M ties plain BFS within 0.01 b/e everywhere: degree
  tie-breaking is inert on these graphs.

## Further directions

Orderings:
- **Recursive graph bisection** (Dhulipala et al., KDD'16) — the state of
  the art for compression-oriented reordering; strictly stronger than
  C-M-style level orderings, and the natural next candidate.
- **Vector-space orderings that skip the graph entirely**: Hilbert /
  space-filling curve order or k-means-cluster order over the original
  SIFT vectors. Would test whether metric locality alone explains the BFS
  gains — and works identically for every graph family.
- Understand *why* RCM wins on NSG/HNSW but not DiskANN: measure achieved
  bandwidth/profile per ordering; correlate with α-pruning aggressiveness.

Codecs:
- Try the library's other variants per component: `B_GEF`/`B_STAR_GEF` on
  *unsorted* rows (preserving DiskANN's distance-ordered neighbor lists,
  which search quality may prefer) vs U-GEF on sorted rows; `RLE_GEF`
  after relabeling (gap-1 runs are now common).
- Partition-size sweep (default 32,000) and its ratio/access trade-off.

Systems:
- Query-time cost: run beam search over the compressed graph
  (`get_elements` per hop) and measure latency vs the raw CSR — the
  memory halving is only free if decompression stays off the critical
  path.
- End-to-end index: combine graph compression with vector compression
  (PQ) for a full small-footprint index.

Generality:
- Repeat on non-SIFT datasets (GloVe/cosine, DEEP) and larger N to test
  whether the family-dependent labeling ranking is dataset-stable.
- Upstream the fixes: gef's Clang flag, sdsl's louds_tree typo, NSG
  export padding (this repo), and report the duplicate-neighbor quirk to
  faiss.
