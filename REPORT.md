# Results — compressed proximity graphs for ANN search: space, access time, and caching

**TL;DR.** Part I (space, unchanged from the first report): U-GEF stores the
adjacency of every graph-based ANN index we tried in **≈48–53% of its raw
CSR size** (16.4–18.2 bits/edge), and bandwidth-reducing relabelings
(BFS from the entry point on DiskANN, Reverse Cuthill-McKee on NSG/HNSW)
are worth ~1 bit/edge; the same holds on GloVe-100 (18.5 → 18.1 b/e).
Part II (access time, new): searching *over* the compressed graph is not
free — U-GEF's `get_elements` costs **≈880 ns per neighbor row against
60 ns for raw CSR**, which makes beam search **37–45% slower** at equal
recall. The cost does not depend on the partition size: it is the
rank/select positioning inside a global sequence. **Re-partitioning the
same Elias-Fano code at row granularity** (one 5-byte header per node,
sequential decode, no rank/select) keeps the space (18.13 vs 18.18
bits/edge) and decodes a row in **175 ns** — and **beam search over that
store runs at raw-CSR speed** (within ±3% at every recall level, single
thread and 16 threads, SIFT and GloVe): the graph is halved for free.
Per-row binary interpolative coding reaches **17.10 bits/edge** for +2%
latency. Part III (caching, new): the per-node access frequency of beam
search is skewed but *mildly* (Zipf slope ≈ −0.3/−0.4): a frequency-ranked
hot set of **10% of the nodes serves 32% of held-out expansions on SIFT-1M
and 47% on GloVe-100**, **2.5×/1.6× the hit rate of DiskANN's BFS-level
policy** at the same budget, and the ranking can be obtained from sampled
base vectors instead of a query log. A two-tier store (hot rows raw, cold
rows U-GEF) follows the additive cost model — a 20% frequency tier
recovers 38% of the decode overhead (11% latency) for +3.6 bits/edge,
twice what DiskANN's BFS-level tier recovers — but in memory it is
dominated by the row-aligned codec, which leaves nothing to recover; the
caching budget pays when each cold access is an I/O (SSD-resident
graphs), where the hit-rate gap between policies is the whole cost model.
A frequency-biased navigation hierarchy (biased skip list over the same
base graph) beats random and in-degree layers, but by a few percent: in
128 dimensions the layers do not reduce the base-layer work. Also new: the BFS labeling that compresses best is
also the fastest to search (−10% latency on raw CSR), and the unhelpful
partition-size knob. Every number comes from a JSON file in `results/`.

## Setup

| | |
|---|---|
| Datasets | SIFT-1M (`sift-128-euclidean`, 1,000,000 × 128, L2) and GloVe-100 (`glove-100-angular`, 1,183,514 × 100, vectors L2-normalised, inner product); 10,000 test queries each with ann-benchmarks ground truth |
| Graphs | DiskANN/Vamana (diskannpy 0.7, R=32 L=64 α=1.2; also R=16/64 on SIFT), faiss NSG (R=32), faiss HNSW base layer (M=16, degree ≤ 32) |
| SIFT DiskANN R=32 | 27,808,597 directed edges, avg out-degree 27.8, medoid entry 123742 |
| GloVe DiskANN R=32 | 35,254,195 edges, avg out-degree 29.8, entry 879042, built in 91 s (16 threads) |
| Compressor | `gef::U_GEF` (partition 32,000, optimal split point) for neighbors; offsets either U-GEF<uint64> (Part I) or raw uint32 (4 B/node, Parts II–III) |
| Per-row codecs | [src/row_codecs.hpp](src/row_codecs.hpp): row Elias-Fano, row binary interpolative coding, row bit-packed gaps; 5-byte header per node (32-bit bit offset + 8-bit degree) |
| Search | [src/search_bench.cpp](src/search_bench.cpp): DiskANN-style greedy beam search (pool L, prefetch of neighbor vectors, AVX2 float32 distances), identical for every store; recall@10 against ground truth |
| Machine | AMD Threadripper 2950X (2 NUMA nodes × 8 cores, 32 MiB L3), 32 GiB DDR4; GCC 14.2 `-O3 -march=native`; everything runs in a podman container (python:3.11-slim) |
| Timing protocol | single thread pinned to core 0 of NUMA node 0 with the node otherwise idle, best wall time of 3 passes after a warm-up; 4 concurrent single-thread runs were ~35% slower, so all latency numbers are from *sequential* runs. Run-to-run variation of a repeated configuration is 2–4% (interleaved repeats in Part III) |

## Part I — Space

### Headline (SIFT-1M, DiskANN R=32)

![Bits per edge](docs/charts/bits-per-edge.svg)

![Index space](docs/charts/space.svg)

| Component | Raw | U-GEF | Ratio (compressed/raw) | Bits/edge |
|---|---:|---:|---:|---:|
| neighbors | 111,234,388 B | 62,402,690 B | 0.5610 (56.10%) | 17.95 |
| offsets | 8,000,008 B | 991,144 B | 0.1239 (12.39%) | 0.29 |
| **total** | **119,234,396 B** | **63,393,834 B** | **0.5317 (53.17%)** | **18.24** |

Round-trip verification passes on every run. The approximate split-point
strategy gives up 0.7% ratio for a ~3× faster build (1.2 s vs 3.5 s).

### Why ~53%: the labeling, not the codec

Elias-Fano codes exploit small gaps between consecutive sorted values, and
a DiskANN neighbor list is scattered almost uniformly over the id space
(median within-row gap ≈ 17.7k ≈ N/degree). The order-oblivious bound
Σᵢ log₂ C(N, degᵢ) = **16.41 bits/edge** caps any encoder that treats ids
as arbitrary; U-GEF's 17.95 is 1.094× that bound, so the headroom is in the
id assignment.

![Gap distribution](docs/charts/gap-dist.svg)

### Relabelings and graph families

Bandwidth-reducing permutations ([scripts/relabel_graph.py](scripts/relabel_graph.py)):
Reverse Cuthill-McKee on the symmetrized graph, BFS order from the entry
point, and a directed Cuthill-McKee (out-edge BFS with by-degree
tie-breaking). Neighbors bits/edge with U-GEF, all 20 combinations verified:

| Graph | avg deg | Bound b/e | orig | BFS | dir. C-M | RCM | best total ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| DiskANN R=16 | 15.7 | 17.19 | 18.71 | **17.78** | 17.79 | 18.32 | 0.5055 |
| DiskANN R=32 | 27.8 | 16.41 | 17.95 | 17.03 | **17.02** | 17.73 | 0.5046 |
| DiskANN R=64 | 38.0 | 15.92 | 17.43 | 16.47 | **16.36** | 16.92 | 0.4923 |
| NSG R=32 | 20.9 | 16.73 | 18.42 | 17.31 | 17.33 | **16.43** | **0.4788** |
| HNSW M=16 | 22.5 | 16.66 | 18.34 | 16.78 | 16.83 | **16.55** | 0.4846 |

![Ratio vs R](docs/charts/ratio-vs-R.svg)

![Bits/edge by graph and labeling](docs/charts/algo-labeling.svg)

BFS labeling collapses 1.09M gaps to exactly 1 on DiskANN (a BFS over a
metric graph enumerates the space region by region); RCM wins on NSG/HNSW
and lands *below* the order-oblivious bound there (the permutation encodes
metric structure). What no labeling removes is the entropy of the
deliberate long-range edges of a navigable small-world graph.

![Degree distribution](docs/charts/degree-dist.svg)

### GloVe-100 (angular): the same picture, smaller relabeling gains

| GloVe-100, DiskANN R=32 | orig | BFS | dir. C-M | RCM |
|---|---:|---:|---:|---:|
| U-GEF total bits/edge | 18.53 | **18.09** | 18.09 | 18.14 |
| total ratio | 0.5426 | **0.5298** | 0.5299 | 0.5312 |

Relabeling buys 0.44 bits/edge on GloVe against 0.93 on SIFT: word-vector
neighborhoods are less "regional" than SIFT patches, so a BFS visit order
turns fewer edges into small gaps. On the other GloVe families the
ranking of labelings flips relative to SIFT:

| GloVe-100 | edges | avg deg | orig | BFS | RCM |
|---|---:|---:|---:|---:|---:|
| faiss HNSW M=16 (base layer) | 30,059,873 | 25.4 | 18.94 | **18.28** | 18.41 |
| faiss NSG R=32 | 14,457,109 | 12.2 | 19.99 | **17.85** (ratio 0.479) | 19.18 |

(total bits/edge, U-GEF.) RCM won on NSG/HNSW for SIFT, BFS wins on GloVe
for every family: "which labeling" is a property of dataset × construction,
not of the construction alone, and BFS from the entry point is never far
from the best. faiss NSG on GloVe is very sparse (average out-degree 12),
which is why its bits/edge are the highest despite the best ratio.

## Part II — Access time: searching over the compressed graph

### Beam search over U-GEF costs 37–45% at equal recall

SIFT-1M, DiskANN R=32, single thread, 10,000 queries; latency interpolated
at fixed recall@10 from the L ∈ {10,…,256} sweep
([scripts/summarize_results.py](scripts/summarize_results.py)):

| labeling | store | bits/edge (incl. offsets) | µs @ recall 0.95 | µs @ 0.97 | µs @ 0.99 | QPS at L=64 (recall 0.9735) |
|---|---|---:|---:|---:|---:|---:|
| original ids | raw CSR | 34.30 | 140 | 179 | 313 | 5,314 |
| original ids | U-GEF | 19.10 | 194 | 258 | 434 | 3,706 |
| BFS | raw CSR | 34.30 | **122** | **160** | **267** | **6,094** |
| BFS | U-GEF | 18.18 | 187 | 230 | 402 | 4,166 |
| directed C-M | raw CSR | 34.30 | 130 | 167 | 288 | 5,921 |
| directed C-M | U-GEF | 18.17 | 177 | 235 | 396 | 4,142 |
| RCM | raw CSR | 34.30 | 140 | 179 | 310 | 5,332 |
| RCM | U-GEF | 18.84 | 182 | 236 | 425 | 4,109 |

Two findings:

- **The labeling that compresses best also searches fastest.** BFS ids cut
  raw-CSR latency by 10–15% (188 → 169 µs at L=64): the vectors are stored
  in id order, so a metric-locality labeling turns the neighbor-vector
  fetches of one hop into nearby pages (the effect Coleman et al.,
  NeurIPS 2022, obtain with Gorder). RCM gives nothing on DiskANN, matching
  its compression result.
- **U-GEF adds ≈ 1.0 µs per hop** ((240 − 169 µs) / 69.5 hops at L=64), a
  42% latency increase at recall 0.97 with BFS ids (45% with original
  ids). The hop count is unchanged (69.5), so this is pure decode cost.

The DiskANN in-memory layout (fixed stride: degree + R slots, 37.97
bits/edge with the padding) performs as raw CSR: 174.8 µs at L=64 with
BFS ids (raw CSR 168.8) and 184.6 with original ids (raw 188.1) — the
labeling effect carries over to the practical layout.

GloVe-100 (DiskANN R=32, angular; recall is lower at every L because the
dataset is harder — hops are unchanged at 75.8 for L=64):

| labeling | store | bits/edge | µs @ recall 0.80 | µs @ 0.85 | QPS at L=64 (recall 0.7701) | QPS at L=256 (0.8810) |
|---|---|---:|---:|---:|---:|---:|
| original ids | raw CSR | 34.15 | 279 | 474 | 4,899 | 1,494 |
| original ids | U-GEF | 19.33 | 385 | 652 | 3,644 | 1,088 |
| BFS | raw CSR | 34.15 | **233** | **407** | **5,842** | **1,787** |
| BFS | U-GEF | 18.90 | 343 | 567 | 3,981 | 1,283 |
| RCM | raw CSR | 34.15 | 251 | 419 | 5,435 | 1,702 |
| RCM | U-GEF | 18.94 | 352 | 610 | 4,026 | 1,203 |

Same shape as SIFT: BFS ids are 17% faster than original ids on the raw
store, and U-GEF costs +46% latency at equal recall (≈ 1.05 µs per hop).
On the GloVe HNSW base layer (flat search from its entry point, 77.6 hops
at L=64) raw/BFS runs at 172 µs and U-GEF at 237 µs (+38%), RCM ids are
7% slower than BFS on both stores. The faiss NSG built on GloVe is not
usable for search comparisons (recall@10 = 0.30 at L=64 and 0.43 at
L=256 with average out-degree 12): its compression numbers stand, its
timings are omitted.

### The cost is positioning, not partition size

Row-decoding microbenchmark ([src/rowbench.cpp](src/rowbench.cpp)): 2M
uniformly random rows, idle core, best of 3.

| store (SIFT-1M R=32, BFS ids) | bits/edge | ns/row |
|---|---:|---:|
| raw CSR (uint64 offsets) | 34.30 | 60 |
| U-GEF, partition 1,024 + raw32 offsets | 20.18 | 931 |
| U-GEF, partition 4,096 | 18.64 | 958 |
| U-GEF, partition 32,000 (default) | 18.18 | 879 |
| U-GEF, partition 128,000 | 18.14 | 879 |
| **row Elias-Fano** (5 B/node header) | **18.13** | **175** |
| **row interpolative (BIC)** | **17.10** | 258 |
| row bit-packed gaps | 19.79 | 134 |

The same table on original ids (U-GEF 19.10 b/e at 809 ns, row EF 18.36 at
176 ns, row BIC 17.87 at 258 ns) and on GloVe-100 (U-GEF 18.90 at 856 ns,
row EF 18.35 at 179 ns, row BIC 17.48 at 272 ns) tells the same story.
`get_elements(start, count)` on a global U-GEF sequence must first locate
`start` inside its partition through the exception bitvector (rank) and
the gap bitvector (select), then decode; those O(1) operations are 3–4
dependent cache misses, and shrinking the partition changes nothing
(the search-side sweep over partitions 1,024–128,000 confirms it: QPS is
flat within noise while space grows 2 bits/edge at 1,024). Aligning the
partition with the access unit — one Elias-Fano block per adjacency row,
addressed by a 40-bit header — keeps the space and removes the rank/select
entirely: **5× faster row access at equal space**. Binary interpolative
coding of the row is even smaller (it exploits the clustered runs the BFS
labeling creates) and still 3.4× faster than U-GEF. The extra header space
(1.44 bits/edge at degree 27.8) is already included in every figure.

### Beam search over row-aligned codes runs at raw speed

The same search, same graph (SIFT-1M R=32, BFS ids), with the per-row
codecs plugged in as the store (single thread; `rep` = an independent
repeat in a later batch, to show the run-to-run spread):

| store | bits/edge | µs @ recall 0.95 | µs @ 0.97 | µs @ 0.99 | mean µs at L=64 | QPS at L=64 |
|---|---:|---:|---:|---:|---:|---:|
| raw CSR | 34.30 | 132 | 166 | 277 | 171.9 | 6,074 |
| U-GEF (+ raw32 offsets) | 18.18 | 174 | 217 | 392 | 225.0 | 4,444 |
| **row Elias-Fano** | **18.13** | **131** | **164** | **284** | 170.4 | 5,866 |
| row Elias-Fano (rep) | 18.13 | 135 | 163 | 288 | 167.8 | 5,956 |
| **row interpolative (BIC)** | **17.10** | 135 | 169 | 295 | 174.8 | 5,724 |
| row bit-packed gaps | 19.79 | 129 | 161 | 280 | 166.6 | 6,032 |

With original ids the ranking is the same (row EF 191 µs @0.97, BIC 192,
packed 181, raw 179, U-GEF 258). On GloVe-100 at L=64: raw 171.5 µs,
row EF 171.1, row BIC 183.2, U-GEF 251.1. At 16 threads (SIFT, L=64):
packed raw 20,872 QPS, row EF 20,841, row BIC 20,938, U-GEF 19,812.

**A row-aligned Elias-Fano store halves the adjacency (18.1 bits/edge)
at no measurable search cost — single-thread latency is within the ±3%
run-to-run spread of the raw store at every recall level, and 16-thread
throughput is identical.** Interpolative coding buys a further 1 bit/edge
for +2% latency (single thread) and nothing at 16 threads. The decode is
hidden because a hop costs ≈ 2.4 µs of vector fetches and distances
(20.7 unvisited neighbors per hop, each a 512-byte random read) against
~110 ns of extra decode; U-GEF's ~820 ns extra is the one that shows.

![Latency vs space per store](docs/charts/pareto-latency-space.svg)

The same three stores on the other SIFT graphs (single thread, L=64):

| graph, labeling | recall@10 at L=64 | raw CSR µs (b/e) | row EF µs (b/e) | U-GEF µs (b/e) |
|---|---:|---:|---:|---:|
| NSG, BFS | 0.9736 | 149.4 (35.07) | 150.6 (18.93) | 207.1 (18.84) |
| NSG, RCM | 0.9736 | 151.5 (35.07) | 159.6 (18.86) | 207.9 (17.97) |
| HNSW base, BFS | 0.9693 | 154.3 (34.84) | 153.9 (18.67) | 218.9 (18.20) |
| HNSW base, RCM | 0.9693 | 163.4 (34.84) | 160.8 (18.62) | 219.6 (17.97) |
| DiskANN R=64, BFS | 0.9856 | 215.2 (33.69) | 223.8 (17.29) | 285.2 (17.31) |

Row EF stays within 0–5% of raw on every graph; U-GEF costs 33–42%. RCM
ids, which compress best on NSG/HNSW with U-GEF (−0.9 b/e), search 1–6%
slower than BFS ids on every store — a space/time trade the row codecs
do not need, since with per-row EF the two labelings are within 0.1 b/e.

## Part III — Caching: who is hot, and what a hot tier buys

### The access-frequency profile of beam search

`search_bench --profile` counts, per node, how many times its row is
fetched (expansions) and how many times its vector is read (touches). With
5,000 real queries at L=64 the profile is sparse (347k expansions over
250k distinct nodes, 0.35 per node); a converged profile uses **200,000
sampled base vectors as queries** — no query log is needed — giving 13.9M
expansions that touch 98% of the nodes. Ranking nodes by that profile and
measuring the share of expansions of 5,000 *held-out real queries* that
land in the top-h nodes ("hit rate") gives, on SIFT-1M DiskANN R=32:

![Hit rate vs budget by policy](docs/charts/hit-rate-policy.svg)

| hot set (nodes) | frequency | BFS levels (DiskANN) | in-degree (hubs) | out-degree | random | oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 1% (10,000) | **9.2%** | 5.3% | 6.4% | 5.7% | 1.0% | 14.4% |
| 5% (50,000) | **21.0%** | 8.7% | 13.7% | 9.8% | 4.9% | 38.8% |
| 10% (100,000) | **32.2%** | 12.8% | 20.9% | 14.3% | 9.8% | 56.0% |
| 20% (200,000) | **49.0%** | 20.0% | 33.6% | 24.7% | 19.5% | 84.8% |
| 30% (300,000) | **61.6%** | 28.4% | 44.8% | 35.9% | 29.5% | 100% |
| 50% (500,000) | **79.8%** | 43.5% | 63.7% | 61.9% | 49.4% | 100% |

("oracle" ranks nodes on the evaluation queries themselves and is an upper
bound that over-fits a 5k-query sample; the vector-touch profile, 282M
touches, shows frequency within 3 points of its oracle: 28.7% vs 31.4% at
10%, so the frequency ranking is essentially converged.)

- **Skew is real but mild.** Zipf slope of the rank–frequency curve is
  −0.31 (SIFT) / −0.47 (GloVe); the entry region (BFS levels 0–3, 21k
  nodes) receives only **6.9%** of expansions at L=64 — levels 4–6 take
  91%. This is why DiskANN's BFS-level cache plateaus (as Gorgeous 2025 and
  GoVector 2025 also observe on SSD): after the first few hops the search
  is inside a query-specific region.
- **Frequency beats every structural proxy at every budget**: 2.5× the BFS
  policy at 10% on SIFT, 1.6× on GloVe. In-degree (the hub proxy used by
  HM-ANN) is the best workload-free ranking on DiskANN but is much weaker
  on NSG and HNSW (at 1%: 3.5% and 2.3% vs 7.6–7.9% for frequency), whose
  pruning leaves fewer hubs.
- **Skew depends on the search depth.** With 5k training queries, the top
  1% of nodes serve 23.5% of expansions at L=10 (recall 0.77), 11.5% at
  L=32, 7.4% at L=64, 5.4% at L=128: fast, low-recall searches spend a
  larger share of their hops near the entry point.
- **GloVe is more skewed than SIFT**: frequency 1%/10%/20% → 15.9% / 47.0%
  / 63.8% (BFS levels 9.3 / 28.9 / 43.7; in-degree 9.9 / 33.2 / 48.9).
- **The family matters little**: NSG and HNSW base-layer search from the
  medoid give the same 7–8% (1%) and 22% (10%) frequency coverage as
  DiskANN with 5k training queries.

### A two-tier ("biased") store: hot rows raw, cold rows U-GEF

[scripts/make_hot_labeling.py](scripts/make_hot_labeling.py) relabels the
graph so the h hottest nodes get ids [0,h) (in decreasing frequency) and
the cold nodes keep their BFS order; the store keeps rows [0,h) as a raw
CSR and the cold rows in a U-GEF built over the cold rows only, so the tier
test is `u < h`. This is the biased-skip-list rule — heavy items get the
short path — applied to graph rows; under a space budget the optimal
static tier is exactly the top-h by access probability, since every raw
row costs the same extra space. The additive model
t(h) = t_raw + (1 − hit(h))·(t_gef − t_raw) predicts the latency from the
hit rates above (SIFT, L=64, recall 0.9735, single thread; t_raw = 162.5
µs, t_gef = 223.4 µs in this batch):

| policy | hot nodes | bits/edge | hit rate (held-out) | model µs | measured µs | vs raw |
|---|---:|---:|---:|---:|---:|---:|
| frequency | 1% | 18.39 | 9.2% | 218 | 229 | 1.41 |
| frequency | 5% | 19.14 | 21.0% | 211 | 218 | 1.34 |
| frequency | 10% | 20.04 | 32.2% | 204 | 213 | 1.31 |
| frequency | 20% | 21.76 | 49.0% | 194 | 205 | 1.26 |
| frequency | 30% | 23.43 | 61.6% | 186 | 208 | 1.28 |
| BFS levels | 2% | 18.61 | 6.7% | 219 | 227 | 1.40 |
| BFS levels | 10% | 19.96 | 12.8% | 216 | 221 | 1.36 |
| BFS levels | 20% | 21.54 | 20.0% | 211 | 215 | 1.32 |
| in-degree | 10% | 20.11 | 20.9% | 210 | 215 | 1.32 |
| in-degree | 20% | 21.95 | 33.6% | 203 | 210 | 1.29 |
| (all U-GEF) | 0 | 18.18 | — | 223 | 223 | 1.37 |
| (all raw) | 100% | 34.30 | — | 163 | 163 | 1.00 |

The measurements track the model's ordering (frequency > in-degree > BFS
at equal budget) and sit 5–20 µs above it; the small tiers (≤ 2%) are
within run-to-run noise of the all-compressed store. Three interleaved
repeats of the 20% tiers pin the effect down (mean ± sd of the per-run
mean latency at L=64; [scripts/errorbars.py](scripts/errorbars.py)):

| store | bits/edge | mean µs at L=64 | QPS | model µs |
|---|---:|---:|---:|---:|
| raw CSR (BFS ids) | 34.30 | 165.2 ± 3.0 | 6,128 ± 13 | — |
| row Elias-Fano | 18.13 | 171.4 ± 2.5 | 5,958 ± 39 | — |
| U-GEF | 18.18 | 230.3 ± 2.7 | 4,406 ± 73 | — |
| two-tier, BFS levels 20% | 21.54 | 217.4 ± 1.6 | 4,603 ± 39 | 217 |
| two-tier, frequency 20%, hot-first ids | 21.76 | 204.8 ± 5.1 | 4,902 ± 97 | 198 |
| two-tier, frequency 20%, BFS ids + slot map | 22.65 | 205.2 ± 1.6 | 4,948 ± 36 | 198 |

The BFS-level tier lands exactly on the model; the frequency tier
recovers **38% of the decode overhead** (vs 20% for BFS levels) and sits
3% above its prediction — the same whether the hot rows are made
contiguous by relabeling or addressed through a 4 MB slot map, so the
relabeling is not the cause (an earlier single raw-store run under the
hot-first labeling suggested a locality loss; it does not survive
repetition). Net, a 20% frequency tier over U-GEF buys 11% latency for
+3.6 bits/edge — a poor trade in memory, where the row-aligned codecs of
Part II remove the overhead at no space cost: the same tiers built over
row EF (hot 10–20% raw, cold rows row-EF) run at 164–173 µs, i.e. within
noise of both the flat row-EF store (170 µs) and the raw store (172 µs),
because there is no longer an overhead to recover.
On GloVe-100 the tier does better, because the profile is more skewed
(hit rates 47% / 64% at 10% / 20%) and the decode overhead is larger:

| GloVe-100, L=64 (recall 0.7701) | bits/edge | mean µs | µs @ recall 0.80 | vs raw |
|---|---:|---:|---:|---:|
| raw CSR (BFS ids) | 34.15 | 171.5 | 233 | 1.00 |
| U-GEF (BFS ids) | 18.90 | 251.1 | 343 | 1.46 |
| two-tier frequency 10% | 20.55 | 219.4 | 303 | 1.28 |
| two-tier frequency 20% | 22.11 | 225.1 | 306 | 1.31 |
| two-tier BFS levels 10% | 20.54 | 222.0 | 307 | 1.29 |

The 10% frequency tier recovers 40% of the decode overhead for +1.65
bits/edge; the 20% tier recovers no more (its labeling moves twice as many
vectors out of BFS order), and the BFS-level tier — which keeps BFS ids —
is within noise of the frequency tier despite a 29% vs 47% hit rate. The
locality of the id assignment is worth as much as the hit rate in memory.
The policy comparison itself is
the durable result: when a cold access is an SSD read rather than 800 ns
of decoding, hit rate is the whole cost model, and a frequency-ranked
adjacency tier built from sampled base vectors halves the misses of the
BFS-level cache DiskANN ships.

### Navigation layers as a way to spend the same budget

HNSW's upper layers are a skip-list-style cache of the entry region. We
build 2-level hierarchies (62,500 = N/16 and 3,906 = N/256 nodes, a
faiss-HNSW graph on each subset, greedy ef=1 descent,
[scripts/build_hierarchy.py](scripts/build_hierarchy.py)) over the *same*
DiskANN base graph, with the nested subsets chosen at random (HNSW's coin
flip), by in-degree (HM-ANN's hub promotion), or by measured expansion
frequency (the biased-skip-list rule: heavy items get the high levels),
and compare against the flat search from the medoid. SIFT-1M, BFS ids,
single thread:

| layer-0 entry | base store | recall@10 at L=64 | hops at L=64 | mean µs at L=64 | µs @ recall 0.97 |
|---|---|---:|---:|---:|---:|
| medoid (flat) | raw CSR | 0.9735 | 69.5 | 171.9 | 166 |
| random layers | raw CSR | 0.9734 | 72.8 | 169.7 | 165 |
| in-degree layers | raw CSR | 0.9731 | 72.8 | 165.9 | 161 |
| **frequency layers** | raw CSR | 0.9731 | 72.7 | **159.6** | **156** |
| medoid (flat) | row EF | 0.9735 | 69.5 | 172.0 | 166 |
| frequency layers | row EF | 0.9731 | 72.7 | 164.5 | 160 |
| medoid (flat) | U-GEF | 0.9735 | 69.5 | 225.0 | 217 |
| random layers | U-GEF | 0.9734 | 72.8 | 233.5 | 226 |
| frequency layers | U-GEF | 0.9731 | 72.7 | 227.1 | 221 |

(The hop count includes the ~3.2 descent steps; the layer-0 work is
unchanged.) In 128 dimensions the hierarchy does not reduce the base-layer
search at all — recall and layer-0 hops are identical to the flat search,
as Munyampirwa et al. (2024) found for HNSW — so its only effect is a
different entry point and the cache behaviour of the first hops. Within
that small effect the biased-skip-list ordering holds: frequency-chosen
layers are consistently the fastest (−7% latency on the raw store at
equal recall, −4% on row EF, and the only variant that does not lose on
U-GEF), in-degree second, random last. As implemented the layers cost
3.8–4.3 bits/edge (two 4 MB base-to-local id maps; a bitmap with rank
would make that ≈ 0.5). This is a null-to-small result for in-memory
search in high dimension, reported because it closes the question the
biased-skip-list analogy raises; the analogy's real payoff is the hot-tier
policy above.

### Multithreaded throughput: row codecs are free, U-GEF nearly so

16 threads on the 8 cores (+SMT) of one NUMA node, 10,000 queries, best of
3 passes (SIFT-1M R=32, BFS ids unless noted):

| store | bits/edge | QPS at L=32 (recall 0.93) | QPS at L=64 (0.9735) | QPS at L=128 (0.9918) | µs @ recall 0.97 |
|---|---:|---:|---:|---:|---:|
| raw CSR | 34.30 | 35,481 | 20,576 | 11,620 | 749 |
| packed / DiskANN layout | 37.97 | 33,395 | 20,872 | 11,775 | 744 |
| raw CSR, original ids | 34.30 | 30,688 | 17,983 | 10,165 | 855 |
| **row Elias-Fano** | **18.13** | **35,652** | **20,841** | **11,806** | **739** |
| **row interpolative (BIC)** | **17.10** | **35,952** | **20,938** | **11,874** | **736** |
| U-GEF (two runs) | 18.18 | 33,824 / 33,944 | 19,812 / 19,725 | 11,112 / 11,040 | 776 / 780 |
| two-tier frequency 20% + U-GEF | 21.76 | 32,357 | 19,136 | 10,813 | 804 |
| two-tier frequency 20% + row EF | 21.46 | 33,445 | 19,591 | 11,062 | 812 |

With all hardware threads busy the search is bound by memory latency and
bandwidth. The row codecs match the raw layouts at every operating point
(they move fewer bytes per hop), and U-GEF's ≈ 1 µs of decode per hop
now overlaps with the stalls of the sibling hyperthread: **the
compressed stores deliver the throughput of the raw layout at 48–53% of
its space, and U-GEF is within 4–5%.** The BFS labeling is worth 14% over
original ids here as well. The two-tier stores are the slowest compressed
variants in this regime: their extra bytes cost more than the decode they
save.

## Takeaways

- **Space**: U-GEF halves the adjacency of DiskANN/NSG/HNSW indexes on both
  datasets; BFS from the entry point is the robust labeling (best on
  DiskANN and on every GloVe family), RCM wins only on SIFT's NSG/HNSW;
  gains are dataset-dependent (0.9 b/e on SIFT, 0.4 on GloVe).
- **Access time**: a global U-GEF sequence is the wrong partitioning for
  row-oriented access — its O(1) positioning costs ≈ 880 ns per row, 5×
  what a row-aligned Elias-Fano block costs at identical space (18.1
  bits/edge), and per-row interpolative coding is better on both axes
  (17.1 bits/edge, 258 ns). Partition size is not the lever. This is the
  concrete design recommendation for applying GEF to graphs: partition at
  the row (or fixed-degree block) and index rows with a compact header.
- **Labeling is a two-for-one**: BFS ids give the best DiskANN compression
  and 10–15% faster raw search.
- **Caching**: expansion frequency is skewed mildly and predictably; a
  frequency-ranked hot set from sampled base vectors beats BFS levels 2.5×
  (SIFT) / 1.6× (GloVe) in hit rate at equal budget and is the right
  policy for SSD-resident graphs; in memory, a static hot tier over U-GEF
  buys 11% latency for +20% space and is dominated by fixing the codec.
- **Hierarchy**: navigation layers over the same base graph leave the
  layer-0 work unchanged in 128 dimensions; frequency-chosen layers are
  the best of random / in-degree / frequency, by a few percent.

## Reproducing

Pipeline in [README.md](README.md). Part II/III runs (inside the
container; see CLAUDE.md for the podman setup):

```bash
python scripts/export_vectors.py --input data/sift-128-euclidean.hdf5 --prefix data/sift
python scripts/relabel_graph.py --prefix data/graph --out-prefix data/graph_bfs --strategy bfs --start 123742   # also writes _perm.bin
./build/search_bench --graph data/graph_bfs --perm data/graph_bfs_perm.bin --base data/sift_base.fbin \
    --query data/sift_query.fbin --gt data/sift_gt.ibin --entry 123742 --backend gef --L 10,32,64,128,256 --json results/x.json
# access profile from sampled base vectors, policy comparison, hot-first labeling, two-tier store
python scripts/sample_base_queries.py --base data/sift_base.fbin --out-prefix data/sift_bsample200k --n 200000 --self-gt
./build/search_bench ... --query data/sift_bsample200k_query.fbin --gt data/sift_bsample200k_gt.ibin --backend raw --L 64 --threads 8 --profile data/prof200k
python scripts/hotness_analysis.py --graph data/graph_bfs --train-expand data/prof200k_expand.u32 --eval-expand data/prof_eval_expand.u32 --entry 0 --out results/hotness.json
python scripts/make_hot_labeling.py --prefix data/graph_bfs --perm data/graph_bfs_perm.bin --expand data/prof200k_expand.u32 --policy freq --hot-count 100000 --out-prefix data/graph_bfs_hotfreq100000
./build/search_bench --graph data/graph_bfs_hotfreq100000 --perm data/graph_bfs_hotfreq100000_perm.bin ... --backend hybrid --hot-count 100000
./build/rowbench data/graph_bfs          # codec microbenchmark
python scripts/summarize_results.py results/*.json ; python scripts/make_charts.py
```
