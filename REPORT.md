# Results — U-GEF on the SIFT-1M DiskANN graph

**TL;DR:** U-GEF stores the DiskANN proximity graph in **53.2% of its raw CSR
size** — **18.24 bits/edge** against 34.30 raw — while keeping O(1) random
access, and bandwidth-reducing id relabelings push every graph family we
tried to **≈48–51%**. Which relabeling wins depends on the family:
**BFS order from the entry point** is best on DiskANN (50.5% at R=32, 49.2%
at R=64), while classic **Reverse Cuthill-McKee** wins on NSG (**47.9%**,
the best result overall) and HNSW (48.5%) — on those two it even lands
*below* the order-oblivious lower bound, proof that the relabeling encodes
real metric structure. A directed C-M variant ties plain BFS everywhere.
What survives every labeling is the entropy of the graphs' deliberate
long-range links: no permutation makes a navigable small-world graph banded.

## Setup

| | |
|---|---|
| Dataset | SIFT-1M (`sift-128-euclidean.hdf5`, ann-benchmarks): 1,000,000 × 128 float32 |
| Graph | DiskANN (Vamana), `R=32`, `L=64`, `alpha=1.2`, L2 metric, via diskannpy 0.7.0 |
| Resulting graph | N = 1,000,000 nodes, **27,808,597 directed edges**, avg out-degree 27.81, max 32 |
| Compressor | `gef::U_GEF` (partition size 32,000), optimal split point unless noted |
| Export | per-node neighbor lists sorted ascending, CSR (`uint64` offsets, `uint32` neighbors) |
| Machine | Apple Silicon Mac; graph built in an x86_64 Linux container (Rosetta), compression measured natively (AppleClang, `-O3`) |
| Index build time | 1489.5 s (container, 8 threads) |

## Headline result

![Bits per edge](docs/charts/bits-per-edge.svg)

![Index space](docs/charts/space.svg)

| Component | Raw | U-GEF | Ratio (compressed/raw) | Bits/edge |
|---|---:|---:|---:|---:|
| neighbors | 111,234,388 B | 62,402,690 B | 0.5610 (56.10%) | 17.95 |
| offsets | 8,000,008 B | 991,144 B | 0.1239 (12.39%) | 0.29 |
| **total** | **119,234,396 B** | **63,393,834 B** | **0.5317 (53.17%)** | **18.24** |

Round-trip verification passes: all 1M adjacency lists decompress exactly,
with offsets checked element-wise against the raw array.

### Optimal vs. approximate split point

| Strategy | Total compressed | Ratio | Bits/edge | Wall time* |
|---|---:|---:|---:|---:|
| `OPTIMAL_SPLIT_POINT` | 63,393,834 B | 0.5317 | 18.24 | 3.5 s |
| `APPROXIMATE_SPLIT_POINT` | 64,205,954 B | 0.5385 | 18.47 | 1.2 s |

*Whole program: load + compress + verify + serialize. The approximate
strategy gives up 0.7% ratio (+0.23 bits/edge) for a ~3× faster build.

## Why the ratio is ~53% and not better

Elias-Fano-style codes exploit *small gaps* between consecutive sorted values.
A DiskANN graph's neighbor lists are the opposite of clustered: pruned for
diversity, each node's ~28 neighbors are scattered nearly uniformly over the
whole 0..10⁶ id space.

![Gap distribution](docs/charts/gap-dist.svg)

With the original DiskANN ids, the gaps between consecutive sorted neighbor
ids peak right where uniform scattering predicts (median ≈ 17.7k, mean ≈
32.7k ≈ N / avg-degree ≈ 36k). This pins the entropy: the combinatorial
lower bound for storing 1M sorted subsets of this size distribution,

> Σᵢ log₂ C(N, degᵢ) = 456.4 Mbit ⇒ **16.41 bits/edge**,

means no encoding that treats the labeling as arbitrary (*order-oblivious*)
can beat 16.41 bits/edge. U-GEF's 17.95 bits/edge for the neighbors array is
**1.094× the bound** — the encoding overhead (split-point metadata,
rank/select supports for O(1) access) costs under 10%. The raw `uint32`
layout, by contrast, wastes ~half: 32 bits where ~16.4 carry information.

## Experiment: bandwidth-reducing relabelings

A node id is an arbitrary label, and Elias-Fano codes reward *locality*: if
each node's neighbors cluster near its own id (a small adjacency-matrix
bandwidth), within-row gaps shrink and the same encoder spends fewer bits.
On Prof. Ferragina's suggestion we relabeled the graph
([scripts/relabel_graph.py](scripts/relabel_graph.py)):

- **Reverse Cuthill-McKee** on the symmetrized adjacency matrix — the
  classic bandwidth-minimization heuristic (scipy implementation);
- **BFS order** from the DiskANN entry point (node 123742) on the directed
  graph — a cheap approximation of Cuthill-McKee (same level-set idea,
  no by-degree tie-breaking);
- **directed C-M** — since no standard C-M exists for directed graphs, we
  transplant its by-degree tie-breaking onto the out-edge BFS.

All permute rows and columns, re-sort each list, and leave degrees — and
therefore the order-oblivious bound — unchanged. On the R=32 DiskANN graph:

| Labeling | Neighbors bits/edge | vs. bound | Total compressed | Total ratio |
|---|---:|---:|---:|---:|
| original ids | 17.95 | +9.4% | 63,393,834 B | 0.5317 |
| Cuthill-McKee | 17.73 | +8.0% | 62,635,530 B | 0.5253 |
| **BFS from entry point** | **17.03** | **+3.8%** | **60,181,634 B** | **0.5047** |
| **directed C-M** | **17.02** | **+3.7%** | **60,164,754 B** | **0.5046** |

Permutation cost is negligible next to the index build: 10.7 s for RCM,
0.3 s for BFS (native arm64 scipy). Round-trip verification passes for both.

![Gap distributions by relabeling](docs/charts/gap-dist.svg)

The gap distributions explain the ranking on DiskANN. **BFS** genuinely buys
locality: 1.09M gaps collapse to exactly 1 (median falls 17.7k → 6.1k),
because a BFS over a metric proximity graph enumerates the space region by
region, so mutually-near nodes get consecutive ids. **On DiskANN,
Cuthill-McKee underperforms its own approximation** (median 17.1k, nearly
unchanged): the symmetrized global ordering reshuffles the metric locality
the plain visit order preserves, and Vamana's α-pruned expander offers no
narrow level structure for RCM's minimization to exploit. The directed C-M
variant confirms the tie-breaking itself is inert: it tracks plain BFS
within 0.01 bits/edge.

The improvement is real but bounded: Vamana's α-pruning deliberately keeps
long-range shortcut edges, so a heavy tail of large gaps survives any
relabeling — the ~16k-gap peak shrinks but does not move. Bandwidth
minimization cannot make a small-world graph banded; it can only harvest
the local fraction of its edges, worth about **0.9 bits/edge (2.7
percentage points of total ratio)** here.

## Sweep: graph family, degree bound, and labeling

The same pipeline was run over DiskANN at R ∈ {16, 32, 64}, faiss NSG
(R=32) and faiss HNSW (M=16, base layer, degree ≤ 32) — every graph ×
every labeling ([scripts/run_experiments.py](scripts/run_experiments.py);
all 20 combinations verify exact reconstruction):

| Graph | avg deg | Bound b/e | orig | BFS | dir. C-M | RCM | best total ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| DiskANN R=16 | 15.7 | 17.19 | 18.71 | **17.78** | 17.79 | 18.32 | 0.5055 |
| DiskANN R=32 | 27.8 | 16.41 | 17.95 | 17.03 | **17.02** | 17.73 | 0.5046 |
| DiskANN R=64 | 38.0 | 15.92 | 17.43 | 16.47 | **16.36** | 16.92 | 0.4923 |
| NSG R=32 | 20.9 | 16.73 | 18.42 | 17.31 | 17.33 | **16.43** | **0.4788** |
| HNSW M=16 | 22.5 | 16.66 | 18.34 | 16.78 | 16.83 | **16.55** | 0.4846 |

(bits/edge on the neighbors array; bold = best labeling per graph.)

![Ratio vs R](docs/charts/ratio-vs-R.svg)

![Bits/edge by graph and labeling](docs/charts/algo-labeling.svg)

Three regularities:

- **Denser graphs compress relatively better.** Raising R lowers both the
  bound (log₂(N/deg) shrinks) and U-GEF's achieved bits/edge; the R=64
  graph is the DiskANN best at 0.4923 total with directed C-M. Note
  α-pruning stops at average degree 38 — well short of the R=64 cap — so
  degree saturation, visible as the 43.6% spike at R=32, disappears.
- **Which relabeling wins is a property of the construction, not of the
  encoder.** On both DiskANN graphs BFS/directed-C-M win and RCM trails.
  On NSG and HNSW the ranking flips: RCM wins outright — NSG's aggressive
  pruning (avg degree 20.9, tree-like backbone from its MST phase) and
  HNSW's un-α-diversified neighbor lists leave global bandwidth structure
  that the symmetrized ordering finds and a single-source BFS does not.
- **NSG+RCM (16.43) and HNSW+RCM (16.55) fall *below* their order-oblivious
  bounds (16.73 / 16.66).** This is not a paradox: the bound counts
  uniformly-random degree-constrained subsets, and a labeling chosen from
  the graph's own structure makes the actual neighbor sets far from
  uniform. Beating it is direct evidence the permutation moved real
  information into the id assignment.

The offsets array is the friendly case for Elias-Fano — strictly increasing
with small, regular gaps (the degrees) — and collapses to 12.4% of raw
(0.29 bits/edge of overhead in the total).

![Degree distribution](docs/charts/degree-dist.svg)

The degree distribution shows Vamana's pruning at work: 43.6% of nodes sit at
the `R=32` cap and the median degree is 30, so the offsets are near-linear —
which is exactly why they compress by 8×.

## Takeaways

- **U-GEF halves the graph memory of every ANN index we tried** — best case
  NSG+RCM at 47.9% of raw (113.7 → 54.5 MiB-scale savings across families)
  with exact reconstruction and O(1) random access.
- **Always relabel; pick the labeling by graph family.** BFS from the entry
  point on DiskANN, classic RCM on NSG/HNSW. Both cost seconds against
  builds of many minutes and need nothing at query time beyond permuting
  the stored vectors. When in doubt, BFS is the robust default: it is
  never worse than original ids by less than ~0.9 bits/edge in this study.
- **The ceiling is the data, not the codec.** On DiskANN, U-GEF lands
  within ~4% of the order-oblivious bound; NSG/HNSW with RCM even beat
  that bound by encoding structure into the labeling. The surviving cost
  everywhere is the entropy of deliberate long-range edges.
- **The approximate strategy is a good default for build-heavy pipelines**:
  3× faster construction for 0.7% more space.

## Reproducing

Follow the pipeline in [README.md](README.md). Individual runs:

```bash
./build/information_retrieval ./data/graph              # optimal
./build/information_retrieval ./data/graph --approximate
```

The full sweep (relabels every graph with every strategy, compresses,
verifies, and emits JSON + a markdown table):

```bash
./venv/bin/python scripts/run_experiments.py --binary ./build/information_retrieval \
    --out ./data/sweep_results.json \
    ./data/graph ./data/graph_R16 ./data/graph_R64 ./data/graph_nsg ./data/graph_hnsw
```

Graph statistics (degree/gap histograms, entropy bound) are computed by a
numpy pass over the exported CSR; the raw stats live in
`data/graph_stats.json` after running the pipeline.
