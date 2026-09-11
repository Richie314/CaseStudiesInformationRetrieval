# Literature notes: Ferragina's group, graph/inverted-index compression, and graph reordering for ANN indexes

Compiled 2026-09-11. Sources verified via arXiv, DROPS/LIPIcs, ACM DL, OpenAlex, project sites and PDFs
(dblp.org is behind an Anubis bot-wall from this host; author lists were cross-checked on OpenAlex,
the authors' pages and the papers themselves). Where I could not verify something I say so.

---

## 1. Paolo Ferragina's relevant work (2018–2026)

Affiliation note: the 2026 GEF supplementary material lists Ferragina as "Sant'Anna School of Advanced
Studies and University of Pisa"; his homepage is https://pages.di.unipi.it/ferragina/ (lab: A³ Lab,
https://acube.di.unipi.it/topics/compressed-data-structures/; ERC/PRIN-style project site
"Multicriteria Learned Data Structures", https://learned.di.unipi.it/ — TLS cert expired at time of check).

### 1.1 Learned indexes and learned compressed structures
- **PGM-index.** P. Ferragina, G. Vinciguerra, "The PGM-index: a fully-dynamic compressed learned index with
  provable worst-case bounds", PVLDB 13(8):1162–1175, 2020. https://doi.org/10.14778/3389133.3389135 ,
  code/site http://pgm.di.unipi.it . Piecewise-linear ε-approximation (PLA) of the CDF of a sorted key set;
  O(log n) worst-case query with space proportional to the number of segments; compressed variant.
- **Why learned indexes work.** Ferragina, Lillo, Vinciguerra, "Why are learned indexes so effective?", ICML 2020,
  http://proceedings.mlr.press/v119/ferragina20a/ferragina20a.pdf — proves the number of PLA segments is
  Θ(n/ε²) under mild i.i.d.-gap assumptions (this is the Lillo paper; see correction under LeMonHash).
- **Survey.** Ferragina, Vinciguerra, "Learned data structures", in *Recent Trends in Learning From Data*,
  Studies in Computational Intelligence 896, Springer 2020, pp. 5–41. https://doi.org/10.1007/978-3-030-43883-8_2
  (PDF: https://learned.di.unipi.it/publication/learned-data-structures/learned-data-structures.pdf).
- **LA-vector (learned rank/select).** A. Boffa, P. Ferragina, G. Vinciguerra, "A 'learned' approach to quicken and
  compress rank/select dictionaries", ALENEX 2021; journal: "A learned approach to design compressed rank/select
  data structures", ACM TALG 18(3), 2022, https://dl.acm.org/doi/10.1145/3524060 . Idea: a sorted integer
  sequence is viewed as points (i, S[i]); a PLA with error ε replaces Elias-Fano's "high bits" by segments and
  stores ε-bounded corrections as fixed-width "low bits". Space bound in terms of number of segments; rank/select
  in O(log) or O(1) with a small index. This is the closest thing to a "learned Elias-Fano": I found **no paper
  titled "Learned Elias-Fano"**; the LA-vector is the EF-shaped learned structure.
- **Block-ε tree / repetition-aware.** Ferragina, Manzini, Vinciguerra, "Repetition- and linearity-aware
  rank/select dictionaries", ISAAC 2021, LIPIcs 212, 64:1–64:16,
  https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ISAAC.2021.64 ; journal version "Compressing and
  querying integer dictionaries under linearities and repetitions", IEEE Access 10:118831–118848, 2022. Combines
  LZ-style repetitions with PLA-linearities (the *block-ε tree*, code https://github.com/gvinciguerra/BlockEpsilonTree ;
  LZε https://github.com/gvinciguerra/LZEpsilon). Space bounded by a measure mixing LZ77 phrases and segments.
- **LeMonHash.** P. Ferragina, H.-P. Lehmann, P. Sanders, G. Vinciguerra, "Learned Monotone Minimal Perfect
  Hashing", ESA 2023, LIPIcs 274, 46, https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2023.46 ,
  arXiv:2304.11012. **Correction to the brief:** the coauthors are Lehmann and Sanders (KIT), not Lillo.
  PGM-based rank prediction + BuRR retrieval; 34% less space and ~16× faster queries than the next larger competitor.
- **Grafite.** M. Costa, P. Ferragina, G. Vinciguerra, "Grafite: taming adversarial queries with optimal range
  filters", SIGMOD 2024 (PACMMOD); arXiv 2023. Optimal-space range filter with theory + experiments.
- **NeaTS.** A. Guerra, G. Vinciguerra, A. Boffa, P. Ferragina, "Learned compression of nonlinear time series with
  random access", ICDE 2025, https://arxiv.org/abs/2412.16266 . Nonlinear-function approximation with random access.
- **PLA theory.** P. Ferragina, F. Lari, "Compressibility measures and succinct data structures for piecewise linear
  approximations", ISAAC 2025, LIPIcs 359, 31, https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ISAAC.2025.31
  — first lower bounds for storing PLAs in compression and indexing settings, plus structures matching them up to
  lower-order terms. Also Ferragina, Lari, "FL-RMQ: a learned approach to range minimum queries", CPM 2025, LIPIcs 331, 7,
  https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CPM.2025.7 .
- **Benchmarking integer indexes.** L. Bellomo, G. Cianci, L. de Rosa, P. Ferragina, M. Odorisio, "A comparative
  study of compressed, learned, and traditional indexing methods for integer data", SEA 2025, LIPIcs 338, 5,
  https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2025.5 — 12 datasets; finding: LA-vector and
  Elias-Fano compress best but lookups are 2–3× slower than SIMD B-trees; learned indexes minimise memory.

### 1.2 Generalized Elias-Fano (GEF) — Pucci & Ferragina, 2025/2026
- **Paper:** Michelangelo Pucci (ETH Zürich, mpucci@ethz.ch) and Paolo Ferragina, "Generalized Elias-Fano code for
  compressed indexing of arbitrary integer sequences" — project site says **"Under submission (2026)"**. No arXiv
  version found (searched arXiv/Google); Pucci's site https://michelangelopucci.com redirects to LinkedIn
  (not fetchable); I could not locate an MSc thesis on etd.adm.unipi.it. Treat the exact venue as unknown.
- **Site/code:** https://gef.di.unipi.it/ ; library https://github.com/mxpucci/generalized-elias-fano (C++20,
  header-only, Apache-2.0, 136 commits); experiments https://github.com/mxpucci/gef-experiments ; detailed tables
  https://gef.di.unipi.it/supplementary-material.pdf .
- **Idea:** EF requires monotone input. GEF encodes *arbitrary* (non-monotone) integer sequences by exploiting
  runs and gaps in the **high-order bits** while keeping fixed-width low bits, so `operator[]` stays O(1). Sequences
  are split into partitions (default 32,000 elements) and the best variant/parameters can be chosen per partition.
- **Variants (from README/site):**
  - **RLE-GEF** – runs of values sharing high-order bits; fastest random access ("speed-critical apps; clustered data").
  - **U-GEF** – *unidirectional*: mostly small positive gaps with occasional large ones — the site explicitly lists
    "graph adjacency lists; timestamps" as targets.
  - **B-GEF** – *bidirectional*: small gaps of either sign ("sensors, fluctuating logs").
  - **B\*-GEF** – "most compact for small gaps", claimed *near-optimal for Laplacian-distributed gaps*
    ("approaching the information-theoretic lower bound for Laplacian gaps").
  - `*_APPROXIMATE` variants trade a little space for faster encoding.
- **Claims:** O(1) random access; >1 GB/s decompression; up to 5 GB/s multithreaded compression; "up to 500×
  faster than Brotli/Xz/NeaTS"; "up to 42.83% more compact than high-throughput alternatives (ALP, LeCo, DAC)";
  "matches Xz/Brotli in space while orders of magnitude faster at access"; "consistently on the Pareto frontier".
- **What it is benchmarked on (supplementary PDF, Tables 1–4):** 16 real **time-series** datasets of 64-bit integers
  (IT/WD/AP/DP/DU NEON sensors; US/UK/GE stock ticks; ECG; LAT/LON GeoLife GPS; CT city temperature; BW/BT Basel
  weather; BM bird migration; BP Bitcoin), vs general-purpose (Brotli, LZ4, Snappy, Xz, Zstd) and special-purpose
  codecs (ALP, Chimp-family, LeCo, NeaTS, TSXor, DAC … — column headers partly unreadable in the PDF).
  Sample compression ratios (compressed/original): IT: GEF 0.10–0.12 vs Xz 0.13, Brotli 0.14; US: GEF 0.06–0.09
  vs Xz 0.09. Random-access throughput (Table 4) GEF variants ≈ 9–140 MB/s vs ≈0.05–1.6 MB/s for block compressors.
  **No graph adjacency lists, inverted lists or ANN indexes are in the benchmark**, despite U-GEF being pitched for
  adjacency lists — a direct opening for this project.

### 1.3 Strings, graphs, cache-oblivious
- **String dictionaries:** Boffa, Ferragina, Tosoni, Vinciguerra, "Compressed string dictionaries via data-aware
  subtrie compaction", SPIRE 2022; "CoCo-trie: data-aware compression and indexing of strings", Information Systems
  2023/24. Ferragina, Rotundo, Vinciguerra, "Engineering a textbook approach to index massive string dictionaries",
  SPIRE 2023, https://link.springer.com/chapter/10.1007/978-3-031-43980-3_16 ; journal "Two-level massive string
  dictionaries", Information Systems 2024. (I found **no** Ferragina–Manzini–Pibiri string-dictionary paper.)
- **Graphs:** Ferragina, Piccinno, Venturini, "Compressed indexes for string searching in labeled graphs", WWW 2015,
  pp. 322–332. **New (2026):** G. Carmona, P. Ferragina, G. Manzini, F. Tosoni, "Extended depth-first
  representations of k²-trees", arXiv:2607.28136, https://arxiv.org/abs/2607.28136 — locality-aware (depth-first,
  balanced-parentheses, subtree-compressed) layouts of k²-tree adjacency matrices, evaluated on web graphs and
  Wikidata for matrix ops; explicitly motivated by cache locality of graph compression formats. Also Tosoni, Bille,
  Brunacci, De Angelis, Ferragina, Manzini, "Toward greener matrix operations by lossless compressed formats",
  IEEE Access 2025.
- **Cache-oblivious:** Ferragina, Venturini, "Compressed cache-oblivious String B-tree", ESA 2013; ACM TALG 12(4):52,
  2016, https://dl.acm.org/doi/10.1145/2903141 ; Ferragina, Grossi, Gupta, Shah, Vitter, "On searching compressed
  string collections cache-obliviously", PODS 2008, https://dl.acm.org/doi/10.1145/1376916.1376943 ; the String
  B-tree (Ferragina & Grossi, JACM 1999).

### 1.4 What the group is doing now (2024–2026) and who is around
- People: Giorgio Vinciguerra is now Applied Scientist at Amazon (https://giorgiovinciguerra.net/; PVLDB 2026
  "Learned static function data structures" with Hermann, Lehmann, Walzer); Antonio Boffa is a postdoc at EPFL DIAS
  (https://www.boffa.top/). Current/recent students on Ferragina's papers: Filippo Lari (PLA theory, FL-RMQ),
  Mariagiovanna Rotundo (oblivious compressed structures, SECRYPT 2026), Francesco Tosoni (k²-trees, matrix
  compression, energy), Angelo Nardone (LLM-based lossless compression, 2026 arXiv), Mattia Odorisio (learned
  sorting, IEEE Access 2025), Lorenzo Bellomo, Andrea Guerra (NeaTS), Michelangelo Pucci (GEF, now ETH).
- Themes: (i) compressed integer/time-series sequences with random access (GEF, NeaTS, SEA'25 benchmark);
  (ii) theory of PLAs and learned structures (ISAAC'25, CPM'25); (iii) locality-aware graph/matrix layouts
  (k²-tree DFS, 2026); (iv) compression of source code / energy cost; (v) LLM-based compression.
  A paper "GEF-compressed proximity graphs + relabeling" fits themes (i) and (iii) squarely.

---

## 2. Graph and inverted-index compression

- **WebGraph.** P. Boldi, S. Vigna, "The WebGraph framework I: compression techniques", WWW 2004, pp. 595–602,
  https://dl.acm.org/doi/10.1145/988672.988752 (PDF https://vigna.di.unimi.it/ftp/papers/WebGraphI.pdf).
  Sorted successor lists; *reference/copy lists* (copy a nearby node's list via a bitmask), *intervalisation*
  (runs of consecutive ids), residual gaps with ζ codes. 3.08 bits/link on WebBase (118M nodes, 1G links),
  2.89 on the transpose. Locality + similarity of web graphs are what make this work, so **node numbering is the
  whole game**. (WebGraph also ships an Elias-Fano successor-list graph class, `EFGraph`, giving O(1)-ish
  successor access — unverified detail from memory.)
- **Shingle ordering / MLogA.** F. Chierichetti, R. Kumar, S. Lattanzi, M. Mitzenmacher, A. Panconesi, P. Raghavan,
  "On compressing social networks", KDD 2009, pp. 219–228, https://dl.acm.org/doi/10.1145/1557019.1557049 .
  Defines Minimum Logarithmic Arrangement (MLogA: min Σ log|π(u)−π(v)|) and MLogGapA (sum of log gaps in sorted
  adjacency lists) as the compression objectives; proves NP-hardness; proposes minhash/"shingle" ordering
  (order nodes by a fingerprint of their neighbourhood) so similar nodes get close ids. Social graphs compress
  worse than web graphs (web ≈ 3 bits/edge).
- **Layered Label Propagation (LLP).** P. Boldi, M. Rosa, M. Santini, S. Vigna, "Layered label propagation: a
  multiresolution coordinate-free ordering for compressing social networks", WWW 2011, pp. 587–596,
  https://dl.acm.org/doi/10.1145/1963405.1963488 . Runs label propagation at several resolutions and concatenates
  the clusterings into an ordering; the standard WebGraph ordering for social graphs.
- **Recursive graph bisection (BP).** L. Dhulipala, I. Kabiljo, B. Karrer, G. Ottaviano, S. Pupyrev, A. Shalita,
  "Compressing graphs and indexes with recursive graph bisection", KDD 2016, pp. 1535–1544,
  https://dl.acm.org/doi/10.1145/2939672.2939862 , arXiv https://arxiv.org/abs/1602.08820 .
  *Model:* bipartite MLogA (BiMLogA) over query vertices Q and data vertices D generalises MLogA and MLogGapA;
  cost = Σ_q Σ log(gaps) of q's sorted neighbour ids. *Algorithm:* recursively split D into two halves; iterate
  swap-based local search where each vertex's "move gain" is the change in the log-gap cost estimate
  (sum over incident query vertices of deg₁·log(n₁/deg₁)-style terms); recursion depth ~log n; initialization by
  Random/Natural/BFS/Minhash (BFS best on graphs, irrelevant on indexes); converges in ≤20 iterations
  (<1% vertices moved). Simple O(log n)-approximation argument for the bipartite objective. Parallel and
  distributed; run on FB graphs with 1B vertices.
  *Numbers (WebGraph BV bits/edge, their Table 2):* LiveJournal Natural 14.61 / BFS 14.69 / LLP 11.12 / **BP 10.73**;
  Twitter Natural 21.56 / BFS 17.99 / Minhash 14.76 / **BP 11.62**; web-Google Natural 20.08 / BFS 7.69 / LLP 5.13 /
  Multiscale 4.10 / BP 4.68; FB-NewOrleans Natural 14.64 / BFS 10.79 / LLP 8.54 / **BP 8.16**. Claimed 5–20%
  over the best alternative, ~50% over natural order; on inverted indexes 22%/15% gains (identical for PEF and BIC).
  *Reproduction:* J. Mackenzie, A. Mallia, M. Petri, J. S. Culpepper, T. Suel, "Compressing inverted indexes with
  recursive graph bisection: a reproducibility study", ECIR 2019, https://jmmackenzie.io/pdf/mm+19-ecir.pdf ,
  code https://github.com/pisa-engine/ecir19-bisection (PISA). PEF docid bits/posting: Gov2 Random 7.96 → URL 4.37
  → **BP 3.67**; ClueWeb09 7.69 → 6.12 → **5.49**; ClueWeb12 7.99 → 6.07 → **5.20**; CC-News 6.06 → 3.38 → 3.31.
  BP time: Gov2 28 min, ClueWeb09 90 min. Follow-up: Mackenzie et al., "Faster index reordering with bipartite graph
  partitioning", SIGIR 2021, https://dl.acm.org/doi/abs/10.1145/3404835.3462991 . A C++ implementation:
  https://github.com/mpetri/recursive_graph_bisection .
- **Gorder.** H. Wei, J. X. Yu, C. Lu, X. Lin, "Speedup graph processing by graph ordering", SIGMOD 2016,
  https://dl.acm.org/doi/10.1145/2882903.2915220 (PDF https://readingxtra.github.io/docs/cpu-graph/WeiSIGMOD2016.pdf).
  Objective: maximise Σ_{|π(u)−π(v)|<w} S(u,v) where S counts shared in-neighbours and direct edges (window w,
  e.g. 5, ≈ cache-line/prefetch reach); NP-hard; greedy priority-queue heuristic with a 1/2-approximation to the
  window objective; speedups >1 in 69/72 cases, up to >2× on PageRank-type workloads. Replication (ReScience 2021):
  https://github.com/lecfab/rescience-gorder . Note Gorder's objective is *locality within a window of positions*
  — exactly cache co-location — while BP's is *small log-gaps inside each adjacency list* — exactly compression.
- **Elias-Fano for indexes.** S. Vigna, "Quasi-succinct indices", WSDM 2013, https://dl.acm.org/doi/10.1145/2433396.2433409
  (PDF https://vigna.di.unimi.it/ftp/papers/QuasiSuccinctIndices.pdf): n sorted ints in [0,U): 2n + n⌈log(U/n)⌉ bits,
  O(1) access, skip via select on the high bits; applied to posting lists and (in WebGraph) adjacency lists.
- **Partitioned Elias-Fano.** G. Ottaviano, R. Venturini, "Partitioned Elias-Fano indexes", SIGIR 2014,
  https://dl.acm.org/doi/10.1145/2600428.2609615 (PDF http://groups.di.unipi.it/~ottavian/files/elias_fano_sigir14.pdf).
  Two-level: DP-optimal partition of the list into chunks each encoded as EF / bitmap / all-ones, chunk upper bounds
  in an EF "skip" list. Adapts to local density (dense clusters cost ≪ log(U/n) bits), keeps random access.
- **Survey.** G. E. Pibiri, R. Venturini, "Techniques for inverted index compression", ACM Computing Surveys 53(6),
  2020, https://dl.acm.org/doi/10.1145/3415148 , arXiv https://arxiv.org/abs/1908.10598 . Table 11 (docids, bits/int and
  ns/int, sequential decoding): Gov2 — VByte 8.81 (0.96 ns), Opt-VByte 3.89 (0.73), BIC 2.94 (5.06), δ 3.74, Rice 4.08,
  **PEF 3.12 (0.76 ns)**, Opt-PFor 3.63 (1.38), Simple16 4.19, QMX 5.12, Roaring 6.63 (0.50); ClueWeb09 — BIC 4.43,
  PEF 4.99, Opt-PFor 5.46, VByte 9.20. Takeaway: PEF ≈ BIC in space but 6–7× faster to decode, and it supports
  random access/skipping, which the others don't natively.
- **Lucene HNSW.** Lucene's on-disk HNSW graph stores each node's neighbour list as `vint` count + delta-encoded
  neighbour ordinals (`Lucene99HnswVectorsFormat`,
  https://lucene.apache.org/core/9_9_1/core/org/apache/lucene/codecs/lucene99/Lucene99HnswVectorsFormat.html) —
  i.e. industry already sorts+gap-codes ANN adjacency, but with byte-aligned codes and no relabeling.

---

## 3. Reordering and compression of ANN proximity graphs

- **Coleman, Segarra, Smola, Shrivastava, "Graph reordering for cache-efficient near neighbor search", NeurIPS 2022**
  (arXiv:2104.03221, https://arxiv.org/abs/2104.03221 ; proceedings
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/fb44a668c2d4bc984e9d6ca261262cbb-Abstract-Conference.html).
  Formulates beam search over HNSW as cache-hit maximisation (relates Gorder's window objective and RCM's bandwidth
  objective to MLA/MLogA). Orderings tried on **HNSW (hnswlib layer-0)**: **Gorder, RCM (Reverse Cuthill–McKee),
  Degree sort (in/out), Hub sort, Hub cluster, DBG (degree-based grouping)** — *BP/recursive bisection was NOT
  tested*, and no compression was measured. Datasets: GIST-1M, SIFT 10M–100M, DEEP 10M–100M, MNIST; k_c∈{4..96};
  R100@100. Results: 10–40% query-time speedup (≈10% at N<1M, up to 40% at 100M and high recall); Gorder and RCM
  consistently help, degree/hub heuristics do not ("pathological degree distribution"). SIFT100M cachegrind:
  L3 miss 13.56% (original) → 8.32% (Gorder) → 8.91% (RCM); P99 latency −30% (Gorder), −17% (RCM). Gorder cost
  ≈ 10× cheaper than HNSW construction. They explicitly suggest combining reordering with quantised indices.
- **PLASMA (GPU).** Y. Oguri, M. Nishimura, Y. Matsui, "On the effectiveness of graph reordering for accelerating
  ANN search on GPU" / "PLASMA: a layout-aware benchmark…", VLDB 2026 VecDB workshop, arXiv:2508.15436,
  https://arxiv.org/abs/2508.15436 . Degree sort, Hub sort, Gorder (w=5), RCM on CAGRA, NSG, Vamana, NN-Descent;
  12 datasets; up to 80% (typically 10–30%) QPS gain on A100, <15% on RTX 6000 Ada. No compression.
- **DiskANN++.** J. Ni et al., "DiskANN++: efficient page-based search over isomorphic mapped graph index using
  query-sensitivity entry vertex", arXiv:2310.00402, https://arxiv.org/abs/2310.00402 . "Isomorphic mapping" = a
  bijective relabeling of Vamana ids produced by a light *pack–merge* heuristic so that a node's neighbours land in
  the same SSD page (a "page compactness" metric), plus page-level search. 1.5–2.2× QPS over DiskANN on 8 datasets.
  Their Table V compares against **parallelGorder** (the parallel Gorder used in OOD-DiskANN, arXiv:2211.12850,
  https://arxiv.org/pdf/2211.12850) on sift/deep/turing-100M (R=32): parallelGorder needs >70 GB RAM and ~2650 s
  (speedup 1.84–2.15×) vs pack-merge 5–6 GB, 123–222 s (1.69–2.02×); random order 1.02–1.08×. **No adjacency
  compression**; all lists remain fixed-R uint32.
- **Starling.** M. Wang et al., "Starling: an I/O-efficient disk-resident graph index framework…", SIGMOD 2024,
  https://arxiv.org/abs/2401.02116 — block-shuffling reorder of the on-disk graph + in-memory navigation graph;
  33M 128-d vectors in 2 GB RAM/10 GB disk, <1 ms latency. No compression.
- **Gorgeous.** P. Yin et al., arXiv:2508.15290, https://arxiv.org/abs/2508.15290 — profiling shows the *graph
  structure is accessed more often than vectors*; caches adjacency lists in RAM; +60% QPS, −35% latency. No compression.
- **VeloANN.** W. Zhao et al., "Optimizing SSD-resident graph indexing for high-throughput vector search",
  PVLDB 14(1)/2026, arXiv:2602.22805, https://arxiv.org/html/2602.22805 — sorts adjacency lists and compresses them
  with **Partitioned Elias-Fano** (Ottaviano–Venturini), co-places "affine" records on disk; up to 5.8× throughput
  over disk baselines, 0.92× of in-memory throughput at 10% of the memory (SIFT1M, GIST1M, Wiki-35M, Image-100M,
  Text-100M). Bits/edge after PEF not reported in the abstract-level extraction.
- **DecoupleVS.** Y. Ren, J. Zhang, Y. Ren, R. Yang, D. Wu, P. P. C. Lee (CUHK/ByteDance), "Decoupling vector
  data and index storage for space efficiency", arXiv:2604.09173v2 (May 2026), https://arxiv.org/html/2604.09173v2 .
  Sorts each DiskANN neighbour list and encodes it with **plain Elias-Fano**: −48.6% neighbour-list size at R=96
  (vs −31.0% with general-purpose compressors); vectors XOR-delta+Huffman (−23.8…−46.4%); total index −58.7% on
  their 109M set, −33.7…−42.6% at billion scale (SIFT1B, SPACEV1B); 2.39× throughput on SIFT100M@98.8% recall and
  −54.6% latency on SIFT1B vs DiskANN (I/O-bound regime, so smaller pages ⇒ faster). Observes that once vectors are
  quantised, **neighbour lists become the dominant memory component**. **No graph-wide relabeling** (only in-list sort).
- **Lossless compression of vector IDs (Meta/FAIR).** D. Severo, G. Ottaviano, M. Muckley, K. Ullrich, M. Douze,
  arXiv:2501.10479 (Jan 2025), https://arxiv.org/abs/2501.10479 , code in Faiss. Compresses IVF id lists and
  **HNSW/NSG edge lists** losslessly with ANS "random order coding" (exploits that neighbour *sets* are unordered)
  and, offline, REC/Zuckerli. Graphs (SIFT1M, Deep1M, FB-ssnpp1M; HNSW/NSG degree 16–256): 13.6–17.7 bits/id
  offline vs 20 (⌈log N⌉) and 32 raw, i.e. up to 2.31× over compact ids; online (per-node stream, random access)
  best ≈50% for NSG256. IVF ids: 9.4–12.5 bits vs 20. **No node relabeling** (they argue relabeling gains
  log(K!) only matter when K≈N). Search-time impact measured only for IVF (≤19% slowdown); *graph search latency
  over compressed lists is not reported*. Notably Ottaviano (BP, PEF author) is a coauthor and still did not
  try BP-style relabeling.
- **Intel SVS / LVQ / LeanVec.** C. Aguerrebere, I. Bhati, M. Hildebrand, M. Tepper, T. Willke, "Similarity search in
  the blink of an eye with compressed indices", VLDB 2023, https://arxiv.org/abs/2304.04759 ; library
  https://github.com/intel/ScalableVectorSearch , docs https://intel.github.io/ScalableVectorSearch/advanced/lowermem.html .
  Their Table 1 (graph + vectors): deep-96-1B, R=32: FP 477 GiB → LVQ-4 168 GiB; R=64: 596 → 287; R=128: 834 → 525.
  Since FP vectors are ≈358 GiB, the **uncompressed uint32 graph is ≈119 GiB at R=32 (≈71% of the LVQ-4 index) and
  ≈477 GiB at R=128 (≈91%)**. SVS docs offer only "use smaller R" for graph memory — **no adjacency compression**.
  LeanVec adds dimensionality reduction (Intel blog https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Dimensionality-Reduction-for-Scalable-Vector-Search/post/1681469).
- **SymphonyQG.** Y. Gou, J. Gao, Y. Xu, C. Long, SIGMOD 2025, https://arxiv.org/abs/2411.12229 — the opposite
  trade: *replicates* neighbours' RaBitQ codes next to neighbour ids (bigger nodes, sequential access), 2× QPS.
- **Zoom.** M. Zhang, Y. He (Microsoft), "Zoom: SSD-based vector search for optimizing accuracy, latency and memory",
  arXiv:1809.04067 (2018), https://arxiv.org/abs/1809.04067 — PQ preview in RAM, full vectors on SSD; graph not compressed.
- **HVS** (K. Lu et al., PVLDB 15(2), 2022, https://www.vldb.org/pvldb/vol15/p246-lu.pdf), **ELPIS** (I. Azizi,
  K. Echihabi, T. Palpanas, PVLDB 16(6), 2023, https://dl.acm.org/doi/10.14778/3583140.3583166), **CAGRA** (H. Ootomo
  et al., ICDE 2024, https://arxiv.org/abs/2308.15136): none compress adjacency; CAGRA uses fixed-degree uint32
  arrays (e.g. 64×4 B = 256 B/node) for coalesced GPU access.
- **Other 2025–26 disk systems** (no compression, layout only): AlayaLaser (SIGMOD 2026, arXiv:2602.23342 —
  argues on-disk graph search is compute-bound), GoVector (arXiv:2508.15694), BatANN (arXiv:2512.09331).

---

## 4. Gap analysis

**(a) Has anyone compressed ANN proximity graphs with EF/GEF codes and measured search latency over the compressed graph?**
Partially, and only in the disk/SSD setting: DecoupleVS (plain EF on sorted DiskANN lists, 2026) and VeloANN (PEF,
2026) both report end-to-end QPS/latency, but their speedups come from fewer/shorter I/Os, not from decoding cost,
and neither reports bits/edge under different labelings. Severo et al. (2025) compress HNSW/NSG lists in memory
(ANS/REC, 13.6–17.7 bits/id) but do *not* report graph search latency and explicitly avoid relabeling. Lucene
gap-codes HNSW lists with vints with no evaluation. **Nobody has (i) used EF/PEF/GEF for an in-memory HNSW/Vamana/NSG
index on CPU, (ii) reported bits/edge as a function of the node labeling, and (iii) measured the QPS/recall Pareto
curve of beam search decoding those lists on the fly.** GEF itself has never been evaluated on adjacency lists, even
though U-GEF is advertised for them. A natural framing: after LVQ/RaBitQ-style vector compression the adjacency is
70–90% of the index (SVS Table 1; DecoupleVS), so 32→~10 bits/edge is the biggest remaining lever.

**(b) Has anyone combined reordering-for-compression (BP/LLP) with reordering-for-cache (Gorder/RCM) on ANN graphs?**
No. Coleman et al. test Gorder/RCM/degree heuristics but not BP and measure no compression; BP papers test web/social
graphs and inverted indexes, never k-NN/proximity graphs; DiskANN++/OOD-DiskANN use parallel Gorder for SSD pages
without compressing. Open questions that would make a paper: does Gorder's window-locality already yield small
log-gaps (i.e., is one relabeling enough), or do the two objectives conflict? What does BP do to the cache-miss
rate of beam search? Can a BiMLogA-style objective be augmented with a window term to get both? Proximity graphs
(k-regular-ish, no hubs, geometric locality) have a very different gap distribution from power-law web graphs, so
the known BP/LLP numbers do not transfer and need measuring (SIFT/DEEP/GIST/MS-Turing/T2I at 1M–100M).

**(c) What does Ferragina's group value in a paper?**
The recurring template (PGM-index, LA-vector, LeMonHash, Grafite, PLA ISAAC'25, GEF): a clean combinatorial object
with a *provable space bound in terms of a data-dependent measure* (segments, LZ phrases, gap entropy, Laplacian
gaps) plus *worst-case query time*, then an *engineered C++ implementation* compared on the space/time Pareto frontier
against many baselines on standard public datasets, with code released (pgm.di.unipi.it, gef.di.unipi.it, GitHub).
Venues: PVLDB/SIGMOD/ICDE for systems-flavoured results; ESA/ISAAC/CPM/SEA and TALG/Information Systems for
algorithmic ones. For this project that means: (1) a space bound for EF/GEF-coded adjacency as a function of the
labeling's log-gap cost (tie to BiMLogA and to B\*-GEF's Laplacian-gap optimality); (2) an O(1)/O(R) access guarantee
so beam search is unchanged asymptotically; (3) experiments on SIFT1M/100M, DEEP, GIST, MS-Turing, T2I with
hnswlib/DiskANN/NSG builds, reporting bits/edge, index size, QPS@recall and cache misses for {natural, BFS, RCM,
Gorder, BP, combined} × {uint32, vint, EF, PEF, U-GEF, B\*-GEF}; (4) reordering cost vs build cost, as Coleman did.
