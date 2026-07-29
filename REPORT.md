# Results — U-GEF on the SIFT-1M DiskANN graph

**TL;DR:** U-GEF stores the DiskANN proximity graph in **53.2% of its raw CSR
size** — **18.24 bits/edge** against 34.30 raw — while keeping O(1) random
access, and it sits only **9.4% above the information-theoretic lower bound**
for this graph's neighbor lists. The adjacency data of a well-built ANN graph
is close to incompressible, and U-GEF captures most of what is there to take.

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

The gaps between consecutive sorted neighbor ids peak right where uniform
scattering predicts (median ≈ 17.7k, mean ≈ 32.7k ≈ N / avg-degree ≈ 36k).
This pins the entropy: the combinatorial lower bound for storing 1M sorted
subsets of this size distribution,

> Σᵢ log₂ C(N, degᵢ) = 456.4 Mbit ⇒ **16.41 bits/edge**,

means *no* encoding of these adjacency sets can beat 16.41 bits/edge. U-GEF's
17.95 bits/edge for the neighbors array is **1.094× the bound** — the encoding
overhead (split-point metadata, rank/select supports for O(1) access) costs
under 10%. The raw `uint32` layout, by contrast, wastes ~half: 32 bits where
~16.4 carry information.

The offsets array is the friendly case for Elias-Fano — strictly increasing
with small, regular gaps (the degrees) — and collapses to 12.4% of raw
(0.29 bits/edge of overhead in the total).

![Degree distribution](docs/charts/degree-dist.svg)

The degree distribution shows Vamana's pruning at work: 43.6% of nodes sit at
the `R=32` cap and the median degree is 30, so the offsets are near-linear —
which is exactly why they compress by 8×.

## Takeaways

- **U-GEF halves the graph memory of a DiskANN index** (113.7 MiB → 60.5 MiB
  for SIFT-1M) with exact reconstruction and O(1) random access — for
  in-memory serving, the graph side of the index effectively costs half.
- **The ceiling is the data, not the codec.** Within-list structure of an
  ANN proximity graph is nearly random; 16.41 bits/edge is the floor and
  U-GEF lands within 10% of it. Substantially better ratios would require
  changing what is stored (e.g. id re-labeling to induce locality), not the
  encoder.
- **The approximate strategy is a good default for build-heavy pipelines**:
  3× faster construction for 0.7% more space.

## Reproducing

Follow the pipeline in [README.md](README.md); the numbers above come from:

```bash
./build/information_retrieval ./data/graph              # optimal
./build/information_retrieval ./data/graph --approximate
```

Graph statistics (degree/gap histograms, entropy bound) are computed by a
numpy pass over the exported CSR; the raw stats live in
`data/graph_stats.json` after running the pipeline.
