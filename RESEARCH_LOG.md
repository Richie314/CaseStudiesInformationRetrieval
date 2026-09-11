# Research log

Chronological record of the experiments in this case study: compressing
graph-based ANN indexes with Generalized Elias-Fano (U-GEF), then
measuring what the compression costs at search time and what caching can
recover. Numbers in [REPORT.md](REPORT.md); this file records what was
tried, in what order, what came out, and what to try next. Literature
notes (four surveys with links) are in [docs/lit/](docs/lit/).

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

## 5. Literature survey and re-planning (2026-09-11)

Four survey passes (in `docs/lit/`): disk-resident graph ANN and its
caches; HNSW hierarchy, hubs and biased skip lists; Ferragina's work and
graph/inverted-index compression; the memory budget of in-memory indexes
with compressed vectors. What they changed in the plan:

- GEF (Pucci & Ferragina, under submission 2026) had never been applied to
  adjacency lists; nobody reports beam-search latency over EF/GEF-coded
  in-memory graphs (Severo, Ottaviano, Douze et al. 2025 compress HNSW/NSG
  lists with ANS to 13.6–17.7 bits/id but neither relabel nor time search).
  → the access-time study became the core.
- DiskANN ships two cache builders (BFS levels, sample-query frequency)
  that were never compared; the 2025–26 SSD papers (Gorgeous, GoVector,
  OctopusANN) report static caches plateauing. → equal-budget policy
  comparison with a held-out evaluation.
- Ferragina's own SASL (Ciriani, Ferragina, Luccio, Muthukrishnan, FOCS
  2002) is the precedent for "heights from access frequency"; the closest
  ANN prior art is HM-ANN (degree promotion) and a 2026 preprint
  (Mycelium-Index, traffic-driven levels, no random-level control).
  → two-tier store + biased navigation layers on one shared base graph.
- After LVQ/RaBitQ-style vector compression the uint32 adjacency is
  70–90% of an index (SVS Table 1); nobody prints that breakdown.

## 6. The search harness and the cost of compressed access

Built `src/search_bench.cpp` (DiskANN-style beam search with pluggable
stores, recall against ground truth, per-node access profiling, optional
navigation layers). Toolchain moved to podman (`ir-lab` image: python
3.11 + GCC 14; diskannpy needs a separate numpy<2 image because it
segfaults at import with numpy 2).

Measurement lessons that cost an afternoon:
- Concurrent single-thread runs on separate cores are ~35% slower than
  sequential ones (memory subsystem contention); every latency in the
  report is from sequential runs pinned to one core of an idle NUMA node.
- Even so, a repeated configuration moves 4–7% between batches; small
  deltas need interleaved repeats.
- Podman `:Z` mounts relabel the tree for one container and lock the
  others out; use `:z`. `podman run` has no stdin, so `python -` heredocs
  run nothing.
- A greedy-descent bug (mutating the current node inside a loop bounded by
  its own row) made the first hierarchy run scan the whole edge array per
  query; caught as a 15-minute run with 97% CPU and an empty log.

Results (SIFT-1M DiskANN R=32, single thread):
- Beam search over U-GEF is 37–45% slower at equal recall than over raw
  CSR; the hop count is unchanged, so it is decode cost ≈ 1 µs/hop.
- BFS ids speed up raw search by 10–15% (vector locality), RCM does not.
- The U-GEF partition size (1,024–128,000) changes space by 2 b/e and
  access time not at all: `rowbench` shows 810–960 ns per row for every
  partition against 60 ns raw — the cost is the rank/select positioning.
- Per-row codecs written as baselines (`src/row_codecs.hpp`): row EF
  18.13 b/e at 175 ns, row BIC 17.10 b/e at 258 ns, row packed gaps 19.79
  at 134 ns. Row EF equals U-GEF's space at 5× the speed; BIC beats it on
  both axes. Same picture on original ids and on GloVe.
- Plugged into the search: row EF searches at raw-CSR speed (SIFT L=64:
  170 vs 172 µs; GloVe 171 vs 171.5; 16 threads 20.8k vs 20.9k QPS) at
  18.1 b/e; BIC +2% single-thread, equal at 16 threads. U-GEF at 16
  threads is within 5% of raw (decode overlaps with memory stalls).
- Two bugs found by the two-tier row-EF store (a sub-CSR of cold rows
  whose ids span the whole graph): the codecs took the id universe from
  the number of rows ("id out of range"), then, after adding a `universe`
  argument, still looped over `universe` rows ("degree > 255"). Both
  fixed; the rowbench/search numbers of the full-graph codecs are
  unaffected (universe == rows there).

## 7. Access-frequency profiling and caching policies

- Profiles from 5,000 real queries are too sparse to rank nodes (0.35
  expansions/node); 200,000 sampled base vectors as queries converge (the
  vector-touch profile ranks within 3 points of an oracle built on the
  test queries) and need no query log.
- Skew is mild: Zipf slope −0.31 (SIFT) / −0.47 (GloVe); the entry region
  (BFS levels 0–3) gets 6.9% of expansions at L=64 and more at low L (top
  1% of nodes: 23.5% at L=10 vs 5.4% at L=128).
- Equal-budget hit rates on held-out queries: frequency 32% / 49% at
  10% / 20% of nodes on SIFT (BFS levels 13% / 20%, in-degree 21% / 34%);
  on GloVe 47% / 64% (BFS 29% / 44%). In-degree is a decent proxy on
  DiskANN, poor on NSG/HNSW.
- Two-tier store (hot rows raw, cold rows U-GEF): with three interleaved
  repeats, the 20% frequency tier is 204.8 ± 5.1 µs vs 230.3 ± 2.7 (U-GEF)
  and 165.2 ± 3.0 (raw) at L=64 — 38% of the overhead recovered, 3% above
  the additive model; the BFS-level tier (217.4 ± 1.6) lands on the model.
  A slot-map variant that keeps BFS ids (205.2 ± 1.6) rules out the
  relabeling as the cause of the residual (an earlier single-run
  "locality loss" reading was noise). Verdict: in memory the codec fix
  dominates (row EF 171.4 ± 2.5 at 18.1 b/e); on SSD the hit-rate gap
  between policies is the result.

## 8. Hierarchies, throughput, and the final picture

- Navigation layers (N/16, N/256; faiss-HNSW graphs on the subsets) over
  the same DiskANN base graph: recall and layer-0 hops identical to the
  flat medoid search (the hubs-paper result for HNSW reproduced on Vamana);
  frequency-chosen layers are the fastest of random / in-degree /
  frequency (−7% on raw, −4% on row EF at equal recall), random the
  slowest. Layers cost 3.8–4.3 b/e as implemented (two 4 MB id maps).
- 16 threads (one NUMA node with SMT): row EF and row BIC match the raw
  layouts (20.8–20.9k QPS at L=64 vs 20.6k raw / 20.9k packed), U-GEF is
  within 4–5% (19.7–19.8k), the two-tier stores are the slowest compressed
  variants (19.1–19.6k): in the bandwidth regime bytes matter, decode
  overlaps with stalls.
- Two-tier over row EF: 164–173 µs, indistinguishable from flat row EF
  and raw — nothing left to recover.
- Total: 10 experiment chains, ~110 timed runs (`results/`), every
  latency from a sequential pinned run; charts regenerated by
  `scripts/make_charts.py`.

What is publishable, in order of strength:
1. GEF applied to proximity graphs for the first time, with the
   diagnosis that the global-sequence partitioning is the wrong access
   unit (positioning ≈ 880 ns/row, partition-size-invariant) and the fix
   — row-aligned Elias-Fano/BIC blocks with a 40-bit header — that halves
   the graph at zero search cost on SIFT and GloVe, single- and
   multi-threaded, across DiskANN/NSG/HNSW. Ties to the GEF paper directly
   (a "row-partitioned GEF" is the natural extension).
2. The labeling study: BFS ids are a two-for-one (best DiskANN
   compression and 10–17% faster search); the compression-best labeling
   is dataset × construction dependent.
3. The caching study: measured access skew of beam search on three graph
   families and two datasets; an equal-budget, held-out comparison of
   DiskANN's two cache builders plus degree and random, with the
   frequency ranking from sampled base vectors (no log) at 2.5× / 1.6×
   the BFS-level hit rate; the additive cost model validated in memory
   and the honest conclusion about where a static tier pays (I/O).

## Further directions

Codecs and layout:
- **Row-partitioned GEF**: apply GEF's split-point optimisation per row
  (or per fixed-degree block of rows) with the 40-bit header — the space
  of B*-GEF/U-GEF with row-EF access time; a natural extension of the GEF
  paper for graphs. Compare with PEF-style partitioning by content.
- Interleave the row's Elias-Fano block with the node's compressed vector
  (LVQ/RaBitQ) so one hop touches one cache-line run (SymphonyQG-style).
- Recursive graph bisection (Dhulipala et al. KDD'16) and Gorder as
  labelings: does the compression-optimal order also minimise cache
  misses, or do the two objectives conflict? BP on proximity graphs has
  never been measured.

Caching:
- Simulate the SSD setting: charge each cold-tier access an I/O and sweep
  the budget 0.1–20% for BFS vs frequency vs in-degree on SIFT/GloVe/DEEP;
  add query-distribution shift (OOD queries) to test the stability of the
  frequency ranking.
- Static + dynamic split (IR's SDC): a small LFU tier for the
  query-specific phase-2 accesses.
- Frequency-aware *layout* without losing locality: keep BFS ids, mark hot
  rows with a bitmap or store the hot tier as a second CSR indexed by a
  rank over that bitmap (costs 1 bit/node instead of a relabeling).

Hierarchy:
- Biased navigation layers with a principled level rule
  h(v) = ⌊log_M(w_v / w_min)⌋ from measured weights, compared with random
  and degree promotion across intrinsic dimensions (d < 32 is where the
  hierarchy matters) and under Zipfian query workloads.

Generality:
- DEEP-96 (10M), GIST-960, larger N; hnswlib/DiskANN native search loops
  instead of our harness; PQ-compressed vectors so that the graph is the
  dominant memory component and its decode cost is measured against a
  cheaper distance.
