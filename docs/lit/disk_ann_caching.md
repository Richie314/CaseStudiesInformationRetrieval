# Disk-resident / memory-constrained graph ANN indexes and their caching strategies — literature notes

Project: EF-compressed adjacency + caching for graph ANN (supervisor: P. Ferragina, Univ. of Pisa).
Compiled 2026-09-11 from arXiv / ACM DL / USENIX / NeurIPS / GitHub sources (all links inline).
Where I could not verify a claim from the primary text, it is flagged **[unverified]**.

---

## 0. Taxonomy of what is actually cached in disk-graph ANN systems

| Policy family | What is cached | Systems |
|---|---|---|
| **Static, entry-region (BFS/SSSP hop distance from medoid)** | full vector + neighbor list of nodes within C hops | DiskANN `cache_bfs_levels`; CXL-ANNS "relationship-aware graph cache"; GoVector static tier; Gorgeous (1–2 hops around nav-index entry points, adjacency lists only); OctopusANN "SSSP cache" |
| **Static, frequency-profiled (sample queries → visit counts → top-N)** | same as above | DiskANN `generate_cache_list_from_sample_queries`; Proxima "hot nodes" (top 3 %); DQF Hot Index (in-memory, retrieval frequency) |
| **Degree / hub promotion** | high-degree nodes placed in fast memory / upper layer | HM-ANN; (hub-highway paper explains why hubs are hot) |
| **In-memory navigation subgraph instead of a cache** | a sampled sub-graph (0.5–10 % of vectors) used only to pick entry points | Starling, PipeANN, Gorgeous, PageANN, BAMG, OctopusANN "MemGraph", d-HNSW |
| **Dynamic page/node caches (LRU / LFU / FIFO)** | recently-read 4 KB pages or records | GoVector dynamic tier (LFU best); XN-Graph (FIFO); AiSAQ (per-thread LRU for PQ pages); VeloANN record-level buffer pool; pgvector buffer cache (Turbocharging); NAVIS entrance-graph-aware edgelist cache |
| **Hybrid static + dynamic** | static hot region + dynamic region | GoVector (2:8 split), DGAI, "Experimental Evaluation" hybrid baseline |
| **Learned / workload-aware** | query-group → cluster prefetch; hotness-ranked partitions | CALL, OrchANN, (survey lists "adaptive/predictive caching" as open) |
| **Relabeling / reordering for locality (not a cache, but changes what a cache line/page holds)** | — | Gorder/RCM/Porder (Coleman et al.), DiskANN++ isomorphic mapping, Starling block shuffling, GoVector k-means layout, Proxima hotness reordering, PLASMA (GPU) |

---

## 1. Paper-by-paper notes

### 1.1 DiskANN — Subramanya, Devvrit, Simhadri, Krishnaswamy, Kadekodi. NeurIPS 2019.
Link: https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html (PDF: https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf); MSR page: https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/
- **Key ideas.** Vamana graph (RNG-style pruning with α-relaxation, high degree R); full-precision vector + out-neighbor list stored together on SSD; PQ codes (e.g., 32 B/point) in DRAM for routing; BeamSearch reads the neighborhoods of W frontier nodes per round; implicit re-ranking with the full-precision vectors that piggy-back on the sector reads.
- **Page layout (paper §3.5).** "the neighborhood of a vertex (4 ∗ 128 bytes long for degree 128 graphs) and full-precision coordinates can be stored on the same disk sector"; reading a 4 KB-aligned address costs the same as 512 B. In the C++ code (`src/pq_flash_index.cpp`, tag 0.5.0) `max_node_len = disk_bytes_per_point + (R+1)*4`, `nnodes_per_sector = 4096 / max_node_len`, `NODE_SECTOR_NO(id) = id / nnodes_per_sector + 1` — i.e. node IDs are assigned round-robin to sectors (no locality).
- **Beam search & I/O.** W = 2, 4, 8 gives the best latency/throughput balance; "if W is too large, say 16 or more, then both compute and SSD bandwidth could be wasted"; at that setting the SSD load factor is 30–40 % and each thread spends 40–50 % of query time in I/O. Hops are the rounds of disk reads; Fig. 2(c) counts hops *assuming the first 3 BFS levels around the medoid are cached in DRAM* (for HNSW: all upper levels cached).
- **Caching policy (paper §3.4, verbatim).** "we cache the data associated with a subset of vertices in DRAM, either based on a known query distribution, or simply by caching all vertices that are C = 3 or 4 hops from the starting point s. Since the number of nodes in the index graph at distance C grows exponentially with C, larger values of C incur excessively large memory footprint." A cached node holds its full-precision vector + neighbor IDs (Starling confirms: "each hot vertex comprises vector data and the neighbor IDs").
- **Code reality (C++ era, tag 0.5.0 `pq_flash_index.cpp`; the repo https://github.com/microsoft/DiskANN was since rewritten in Rust, crates `diskann-disk` etc.).** Two builders of the cache list exist: (i) `cache_bfs_levels(num_nodes_to_cache, node_list, shuffle)` — BFS level-by-level from the medoid(s), hard-capped: "Do not cache more than 10% of the nodes in the index"; (ii) `generate_cache_list_from_sample_queries(sample_bin, l_search, beamwidth, num_nodes_to_cache, ...)` — runs `cached_beam_search` on a sample query file with `count_visited_nodes = true`, fills `node_visit_counter[i]`, sorts descending and keeps the top `num_nodes_to_cache`. So DiskANN ships **both a BFS-level and a frequency-based static cache, but the paper never compares them**.
- **Numbers.** SIFT1B on a 64 GB-RAM workstation: > 5000 QPS, < 3 ms mean latency, 95 %+ 1-recall@1; DEEP1B similar (< 5 ms). Memory ≈ PQ codes (32 B × 1 B = 32 GB) + cache. Datasets: SIFT1M, GIST1M, DEEP1M, SIFT1B, DEEP1B.

### 1.2 FreshDiskANN — Singh, Subramanya, Krishnaswamy, Simhadri. arXiv 2105.09613 (2021).
Link: https://arxiv.org/abs/2105.09613
- Memory-resident temporary (fresh) index + SSD-resident long-term index, StreamingMerge to fold updates; >95 % 5-recall@5 on 1 B points with thousands of concurrent inserts/deletes/searches; 5–10× cheaper freshness than rebuilds. **Caching:** inherits DiskANN's static node cache; no new policy. (Also: Xu et al. "In-place updates of a graph index" arXiv 2502.13826 continues this line.)

### 1.3 OOD-DiskANN — Jaiswal, Krishnaswamy, Garg, Simhadri, Agrawal. arXiv 2211.12850 (2022).
Link: https://arxiv.org/abs/2211.12850
- Uses a 1 %-of-index-size sample of out-of-distribution queries at build time; up to 40 % latency improvement for OOD (e.g., text→image) queries; OOD queries otherwise cost an order of magnitude more latency. **Relevance to caching:** shows the *visited region* depends on the query distribution, so a frequency cache profiled on in-distribution samples may be mis-targeted for OOD workloads. No caching change.

### 1.4 DiskANN++ — Ni, Xu, Wang, Li, Yao, Xiao, Zhang. arXiv 2310.00402 (2023).
Link: https://arxiv.org/abs/2310.00402 ; PDF https://arxiv.org/pdf/2310.00402
- **Key ideas.** (1) *Query-sensitive entry vertex*: offline k-means centroids (N_cluster) as entry candidates; online, pick the candidate nearest the query — shortens the "NN-approaching" phase (proved via MSNET monotonicity). (2) *Isomorphic mapping* of Vamana onto SSD pages: a bijection on node IDs that **keeps DiskANN's arithmetic ID→page addressing** (page capacity b, node i in page ⌈i/b⌉) but changes which nodes share a page. Pack–merge: *star packing* puts a vertex and its b−1 nearest neighbors (by PQ distance) in one temporary page; *merging* uses First-Fit-Decreasing bin packing; new IDs are assigned in page order. They define *page compactness* γ = λ₂(G[page]) / diam(G[page]) and prove γ > 0.5 for star-derived pages; Table I: original round-robin layout γ ≈ 0 (0.000004 / 0 / 0) vs 0.658 / 0.560 / 0.547 on sift100M / deep100M / turing100M (R = 32). (3) *Pagesearch*: a per-query "page heap" (operators Cache/Update/Check2ret/Pop) that keeps every fetched 4 KB page, computes full-vector distances for **all** vertices in the page during the CPU stall of outstanding I/O ("asynchronous page expansion"), and uses them as extra candidates.
- **Caching.** A naive page cache on top of beamsearch ("cachedBeamsearch") gives the *same* number of logical I/O requests with only 10–20 % hits, because "most of the cached SSD pages are unused for the node expansion of beamsearch" — caching without active page expansion is nearly useless. Pagesearch cuts SSD I/O by ≈ 50 % (Fig. 4). No BFS/frequency hot-node cache is discussed; DiskANN++ compares against reordering baselines (parallel Gorder) and notes Gorder needs the whole graph + reverse index in memory.
- **Numbers (ablation Table, A = entry vertex, B = mapping, C = pagesearch).** sift100M @ recall@100 = 97 %: baseline 447.5 mean I/Os, 57.6 hops → ABC 278.7 I/Os, 36.6 hops, QPS ×1.79; deep100M 557.9 → 291.8 I/Os (×2.02); turing100M 1244 → 694 (×1.95). B alone changes nothing (beamsearch cannot exploit page co-location); BC gives most of the gain. Overall 1.5–2.2× QPS vs DiskANN. Memory budgets tested at 1–25 % of dataset size. Datasets: sift100M, deep 10M–1B, turing100M, msong, crawl, glove-100, gist, plus a 100M commercial image set.

### 1.5 Starling — Wang, Xu, Yi, Wu, Peng, Ke, Gao, Xu, Guo, Xie. SIGMOD 2024 (PACMMOD 2(1)), DOI 10.1145/3639269; arXiv 2401.02116.
Links: https://dl.acm.org/doi/10.1145/3639269 ; https://arxiv.org/abs/2401.02116 ; code https://github.com/zilliztech/starling
- **Key ideas.** Per "data segment" (2 GB RAM + 10 GB disk holds 33 M 128-d vectors). (1) *Block shuffling*: reorder the disk graph to maximize the overlap ratio OR(G) (neighbors co-located in the same 4 KB block); NP-hard, heuristics BNP / BNF (neighbor-frequency) / BNS. (2) *In-memory navigation graph*: a **uniform random sample** (< 10 % of vertices, sized to the memory limit) indexed with the same graph algorithm, used to get query-close entry points. (3) *Block search*: compute distances for every vertex in a fetched block, prune, pipeline I/O with compute.
- **Caching.** No hot-node cache; the nav graph replaces it. On DiskANN: "DiskANN necessitates the generation of hot vertices which involves the sampling of a large pool of queries and executing a slow disk-based graph search to tally vertex visit frequency" (index-build cost T_hot > T_shuffling + T_memory_graph); memory C_graph + C_mapping < C_hot. On SSNPP "most query results are near the centroid ... DiskANN's cache policy loads the data near the centroid, covering most query results" — the one case where the BFS cache nearly matches Starling.
- **Numbers.** DiskANN: disk I/O = up to 92.5 % of query time and up to 94 % of vertices in a loaded block wasted. Starling: vertex utilization 0.0625 → 0.3438 (BIGANN), path length 362 → 182 hops, up to 43.9× throughput and 98 % lower latency vs DiskANN at the same accuracy; > 0.9 AP/recall@10 under 1 ms. Datasets: BIGANN 33M, DEEP 11M, SSNPP 16M, Text2Image 5M (segment-sized).

### 1.6 LM-DiskANN — Pan, Sun, Yu. IEEE BigData 2023, DOI 10.1109/BigData59044.2023.10386517.
Links: https://ieeexplore.ieee.org/document/10386517 ; https://par.nsf.gov/biblio/10539353 ; Julia re-implementation (JOSS 2025) https://joss.theoj.org/papers/10.21105/joss.08199
- Each on-disk node stores its full vector + neighbor IDs + **the PQ codes of all its neighbors**, so routing needs no in-memory PQ table (SIFT1M example: (1 + 128 + 70 + 70×8)×4 = 3036 B per node with R = 70, 8-B PQ). Memory ≈ ¼ of Vamana/DiskANN at a similar recall–latency curve; supports dynamic insert/delete. **Caching:** none beyond the OS; no hot-node policy. Datasets: SIFT1M, GIST1M, DEEP1M.

### 1.7 AiSAQ — Tatsuno, Miyashita, Ikeda, Ishiyama, Sumiyoshi (Kioxia). arXiv 2404.06004 (2024).
Links: https://arxiv.org/abs/2404.06004 ; https://github.com/kioxia-jp/aisaq-diskann
- DiskANN fork that moves the PQ codes to SSD too (optionally inline in each node's page, "some or all PQ vectors can be stored as part of the index node"), giving ≈ 10 MB DRAM at billion scale and sub-ms index load. **Caching:** a "static PQ vectors cache" populated at load time and a "dynamic page-level read cache of PQ vectors, managed per-thread using LRU eviction"; keeps DiskANN's `num_nodes_to_cache`. No policy comparison.

### 1.8 PipeANN — Guo, Lu. USENIX OSDI 2025; extended in ACM TOS (DOI 10.1145/3793926); OdinANN follow-up at FAST 2026.
Links: https://www.usenix.org/conference/osdi25/presentation/guo ; https://github.com/thustorage/PipeANN
- Aligns best-first search with the SSD: fully pipelined async I/O (poll-based), dynamic pipeline width (small in the approach phase, larger in the converge phase), and an **in-memory graph over sampled entry points** (max degree 32; 2.4 GB for SIFT1B, 3.1 GB for SPACEV1B) for entry-point optimization. Memory < 40 GB for 1 B vectors (32 GB PQ @ 32 B/vector + < 4 GB in-memory index) vs DiskANN ≈ 32 GB. < 1 ms latency and ≈ 20 K QPS at 1 B (top-10, 90 % recall); 1.14–2.02× the latency of an in-memory index with > 10× less memory. **Caching:** README still exposes `num_nodes_to_cache`; no new policy. Datasets: SIFT1B, SPACEV1B, DEEP100M, SIFT/SPACEV100M.

### 1.9 SPANN / SPFresh / FusionANNS / SmartANNS (non-graph-on-disk baselines; brief)
- SPANN — Chen et al., NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/299dc35e747eb77177d9cea10a802da2-Abstract.html — IVF-style: centroids (SPTAG graph) in memory, posting lists on disk; 2× faster than DiskANN at 90 % recall with the same memory. Caching = the centroid index; no node cache.
- SPFresh — Xu et al., SOSP 2023: https://dl.acm.org/doi/10.1145/3600006.3613166 — LIRE in-place rebalancing; 1 % of DRAM and < 10 % cores vs rebuild-based baselines.
- FusionANNS — Tian et al., USENIX FAST 2025 (arXiv 2409.16576): https://www.usenix.org/conference/fast25/presentation/tian-bing — CPU/GPU cooperative filtering + heuristic re-ranking + I/O deduplication; 9.4–13.1× QPS vs SPANN.
- SmartANNS — Tian et al., USENIX ATC 2024 (+ ACM TOS 2025, DOI 10.1145/3736589): https://www.usenix.org/conference/atc24/presentation/tian — hierarchical indices on SmartSSDs, learned shard pruning; 10.7× vs CSDANNS. None of these studies a node-level cache policy.

### 1.10 CXL-ANNS — Jang, Choi, Bae, Lee, Kwon, Jung. USENIX ATC 2023.
Links: https://www.usenix.org/conference/atc23/presentation/jang ; PDF https://www.usenix.org/system/files/atc23-jang.pdf
- Graph + vectors in a CXL memory pool; **relationship-aware graph caching**: "the pool manager considers how many edge hops ... exist from the fixed entry-node to each node"; per-node hop counts via SSSP/BFS, nodes sorted ascending by hop count and allocated to local DRAM "from the top (having the smallest hop count) ... as many as it can" (budget = free physical pages). Justification with measured skew: over 1 M kNN queries "the nodes most frequently accessed ... reside in the 2∼3 edge hops" (Fig. 9b). Uncached nodes are prefetched one hop ahead; near-data distance computation in the CXL endpoints. 111.1× QPS and 93.3 % lower latency vs SOTA billion-scale platforms; ablation attributes multi-fold QPS gains (9.4× / 20.3× reported in §6) to caching + prefetching. Datasets: six billion-point sets incl. BigANN, Yandex-DEEP, Yandex-Text2Image.

### 1.11 HM-ANN — Ren, Zhang, Li. NeurIPS 2020.
Link: https://proceedings.neurips.cc/paper/2020/hash/788d986905533aba051261497ecffcbb-Abstract.html
- HNSW-like hierarchy on DRAM + Optane PMM: bottom layer L0 in slow memory; upper layers built by **high-degree promotion** ("The hub nodes of the graph at L0 are those nodes with a large number of connections ... Most of the shortest paths between nodes flow through hubs"), promotion rate 1/M per level, upper-layer size set by the available fast memory; ≈ 2 GB of fast memory reserved as a software-managed cache for *dynamic migration* (prefetch of L0 neighbors). 95 % top-1 recall on BIGANN/DEEP1B; 2×–5.8× lower latency vs NUMA/hardware-cached baselines; 46 % higher recall than L&C / IMI+OPQ at equal latency. This is the clearest **degree-based** hot-set policy in the literature.

### 1.12 Gorgeous — Yin et al. (9 authors). arXiv 2508.15290 (Aug 2025).
Link: https://arxiv.org/abs/2508.15290
- Profiling insight: "the structure of the proximity graph index is accessed more frequently than the vectors themselves". DiskANN/Starling caches plateau: "after including a small number of frequently visited vectors (e.g., those close to the entry node), the remaining vectors are accessed more uniformly; the cache size is small compared to the entire dataset, and thus the IO reduction is not obvious" (Fig. 1/6). **Graph-prioritized cache**: cache *only adjacency lists*; worked example S_v = 1536 B, S_a = 200 B: a cache of 10 % of the dataset size holds adjacency lists of **88 % of the nodes** vs 10 % for a coupled vector+adjacency cache — "almost avoiding disk read during graph traversal". Which lists: nodes within 1–2 hops of the entry points of a small (0.5 %) sampled navigation index, validated on sampled queries; a footnote says popular nodes could be selected if visit probabilities differ. **Graph-replicated disk block**: each block stores the vector plus its neighbors' adjacency lists (≤ R+1 replication, ≈ 10 % extra disk). Results: +78 % / +60 % throughput and −41 % / −35 % latency vs DiskANN / Starling (avg), larger at high dimension; datasets Sift100M, Deep100M, Text2Image100M, Wiki, Laion-I2I (768-d), Laion-T2I; memory 10–20 % of dataset.

### 1.13 GoVector — Zhou, Lin, Gong, Yu, Fan, Zhang, Yu. Journal of Software 37(3), 2026; arXiv 2508.15694 (Aug 2025).
Link: https://arxiv.org/abs/2508.15694 (HTML https://arxiv.org/html/2508.15694)
- **Measured access pattern.** Phase 1 (approach) hits the static entry-region cache; phase 2 (converge) visits an annulus around the query with strong spatial locality and low reuse across queries. Static cache hit rate "shows a clear power-law decay" along the search: 19 % (SIFT) / 63 % (GIST) in phase 1 → 4 % / 9 % in phase 2.
- **Policy = hybrid.** Static tier: "entry points and several of their multi-hop neighbors according to a preset capacity" (DiskANN-style; baseline DiskANN cache = 1 % of index file). Dynamic tier: pages of nodes accessed in phase 2 *plus their disk-adjacent pages* (similarity-aware batch loading), replacement FIFO/Random/**LFU (best)**; optimal static:dynamic = 2:8. Layout: k-means clusters placed on the same/adjacent pages (distance-based, unlike Starling's topology-based shuffle).
- **Numbers.** I/O −46 % avg (−57 % max), QPS ×1.73 (×2.25), latency −42 % (−55 %) vs SOTA at 90 % recall; GoVector-Hybrid ×2.61–4.59 QPS vs DiskANN, ×1.10–3.97 vs Starling. Datasets (1 M scale): SIFT, Text2Img, DEEP, Word2Vec, MSONG, GIST (128–960 d).

### 1.14 PageANN — Kang, Jiang, Yang, Liu, Li. arXiv 2509.25487 (2025).
Link: https://arxiv.org/abs/2509.25487 — page-node graph aligning logical nodes with SSD pages, storing representative vectors + topology, lightweight in-memory index; 1.85–10.83× throughput, 51.7–91.9 % lower latency vs disk-based SOTA. Caching: static (per the 2026 evaluation paper's Table 1). Details of the memory manager not in the abstract **[unverified]**.

### 1.15 BAMG — Li, Huang, Choi, Xu. arXiv 2509.03226 (2025, rev. 2026).
Link: https://arxiv.org/abs/2509.03226 — Block-aware monotonic RNG guaranteeing I/O-monotonic paths; decoupled vector/graph storage; multi-layer in-memory navigation graphs; block-first search. No frequency cache.

### 1.16 XN-Graph — Zhang, Wang, Zhao, Xiao. SIGIR 2025, "Highly Efficient Disk-based Nearest Neighbor Search on Extended Neighborhood Graph".
Link: https://cmmlab.xmu.edu.cn/pubs/sigir25.pdf — extended-neighborhood graph + IMF search replacing beam search (no page reorganization needed); loaded points kept in a **FIFO cache** (Alg. line "B ← ∅; FIFO cache for the loaded points"); vs DiskANN/SPANN/Starling on SIFT100M/1B, DEEP100M, LAION100M.

### 1.17 DGAI — Lou et al. arXiv 2510.25401 (Oct 2025, rev. 2026).
Links: https://arxiv.org/abs/2510.25401 ; https://github.com/iDC-NEU/DGAI — decoupled vectors vs topology for cheap updates (8.17×/8.16× insert/delete), similarity-aware *dynamic* layout that turns read amplification into prefetch, hierarchical-PQ two-stage query; classified as **hybrid cache** by the 2026 evaluation paper.

### 1.18 VeloANN — Zhao et al. arXiv 2602.22805 (Feb 2026).
Link: https://arxiv.org/abs/2602.22805 — "hierarchical compression and affinity-based placement" to co-locate related vectors per page, a **record-level buffer pool** that pins neighbor groups, coroutine runtime, beam-aware search prioritizing cached data; 5.8× throughput vs disk systems, 0.92× in-memory throughput at 10 % memory. Closest existing work to "compressed + cached", but compression is of *vectors*, not adjacency.

### 1.19 NAVIS — Song, Jang, Shin, Park, Ryoo, Park, Lee. arXiv 2605.11523 (May 2026).
Link: https://arxiv.org/abs/2605.11523 — concurrent search/update; observes that with a fresh entrance graph "queries repeatedly visit the vertices near entry points, creating a concentrated hot set", but low-reuse edgelists from deep traversal evict them under **LRU** (hit rate collapses); proposes an **entrance-graph-aware edgelist cache** (caches edge lists, not vectors; capacity concentrated near refreshed entry points) and selective vector reads; 2.74× insert throughput, 1.37× search throughput, −25.26 % latency.

### 1.20 OctopusANN / design-space exploration — Li, Gong, Yang, Wang, Wu. PVLDB 2026; arXiv 2602.21514.
Link: https://arxiv.org/abs/2602.21514
- Page-level cost model: **page reads/query = O(R̄·H / (OR(G)·n_p))** (R̄ avg out-degree, H hops, OR overlap ratio, n_p records/page; with PQ → O(H/(OR·n_p))). Dimensions: memory layout {PQ, cache management, MemGraph}, disk layout {page shuffle, all-in-storage}, search {dynamic width, pipeline, page search}.
- **Cache finding:** SSSP/BFS-hop caching "delivers especially strong gains on SPACEV" but only ≈ 9 % I/O reduction on SIFT; "SSSP-based caching is sensitive to graph quality" and is "rarely adopted in recent systems compared with MemGraph" — so it is **excluded from the combination study**; no frequency/LRU/degree ablation. MemGraph + dynamic width are the strongest single factors; page shuffle + page search only work together. OctopusANN: +4.1–37.9 % QPS vs Starling, +87.5–149.5 % vs DiskANN at R@10 = 90 %. Datasets: SIFT/DEEP/SPACEV 100M, GIST1M; memory 3.8–4.7 GB.

### 1.21 Disk-Resident Graph ANN Search: An Experimental Evaluation — Chen, Qu, Song, Lu, Li, Jiang, Zhou, Xu, Zhou, Wu. arXiv 2603.01779 (Mar 2026).
Link: https://arxiv.org/abs/2603.01779 (HTML https://arxiv.org/html/2603.01779)
- Compares DiskANN, FreshDiskANN, AiSAQ, Starling, Gorgeous, PageANN, PipeANN/OdinANN, DGAI along storage strategy / disk layout / **cache management** / query execution / updates. Cache taxonomy: *static* (hot data = "hub vectors that play a critical role in graph navigation, or ... answer vectors corresponding to popular or recurring query topics"; used by DiskANN, Starling, Gorgeous, PageANN), *dynamic* (LRU/LFU; XN-Graph), *hybrid* (DGAI).
- **Only published equal-budget comparison** (static budget 1 % of dataset): hit rate of dynamic/hybrid vs static graph-prioritized = 2.75× / 2.60× on SIFT1M, up to 4.69× vs static hot-data on Deep, but on GIST the static graph-prioritized cache has 2.22× the hit rate of dynamic (which falls to 15.82 %). Recommendation: "dynamic and hybrid caching perform best in low dimensions, whereas static (graph-prioritized) is more effective in high dimensions". Other findings: I/O utilization of all layouts ≤ 15 % (peak 12.55 %); smaller pages win when layouts are optimized. No absolute I/O-per-query per policy, no frequency-static vs BFS-static vs degree comparison. Datasets: GloVe, SIFT1M, SIFT100M, Deep, Tiny, MSong, GIST, OpenAI (100–3072 d).

### 1.22 Turbocharging Vector Databases using Modern SSDs — Shim, Oh, Roh, Do, Lee. PVLDB 18 (2025), DOI 10.14778/3749646.3749724.
Link: https://www.vldb.org/pvldb/vol18/p4710-do.pdf
- pgvector HNSW on SSD with a buffer cache = 50 % of index size (GloVe 1M, 200-d). **Per-layer access/hit table (Table 3):** layer 3: 80 nodes, 40.66 visits/query, 99.98 % hit; layer 2: 1,785 nodes, 47.86, 95.26 %; layer 1: 41,405 nodes, 48.94, 70.21 %; **layer 0: 1,000,000 nodes, 1823.57 visits, 57.49 % hit**; overall hit 59.24 %, SSD utilization 1.98 %. Fixes: io_uring parallel neighbor reads, spatially-aware insertion reordering, locality-preserving colocation → 3.23× hit ratio, cache-miss penalty −80.5 %. Shows that in a hierarchical index the upper layers are effectively a free BFS-level cache while the base layer has weak temporal locality under LRU.

### 1.23 Storage-Based ANN Search: performance, cost and I/O characteristics — Ren, Doekemeijer, Apparao, Trivedi. IISWC 2025.
Link: https://atlarge-research.com/pdfs/2025-iiswc-vectordb.pdf — Milvus DiskANN issues 4 KiB random reads, beats in-memory IVF by up to 3.2×, never saturates the SSD (≤ 1.7 GiB/s); 22 observations, but **no cache-policy or access-skew analysis**.

### 1.24 "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'" — Munyampirwa, Lakshman, Coleman. arXiv 2412.01940 (Dec 2024, rev. Jul 2025).
Link: https://arxiv.org/abs/2412.01940 (HTML https://arxiv.org/html/2412.01940)
- Flat NSW ≈ HNSW in latency/recall with less memory on high-d data; **hub-highway hypothesis**: a "well-connected, frequently traversed 'highway' of hub nodes" does the job of the hierarchy. Evidence: k-occurrence distribution becomes right-skewed as d grows (ℓ2; much less for cosine); hubs preferentially connect to hubs (Mann-Whitney/t-tests, top-1 % and top-5 %); traversal traces binned per 30 visited nodes show hubs concentrated early — "high percentage of hub nodes visited in the first 5–10 % of the search steps" (GIST). Datasets: GIST, GloVe, NYTimes, DEEP10M, SpaceV 10–100M, BigANN100M, MSMARCO (384-d), synthetic IID normals up to 1536-d. This is the best current evidence that **degree (in-degree / k-occurrence) predicts visit frequency**, and thus that a degree-based cache is a proxy for a frequency cache.

### 1.25 Graph Reordering for Cache-Efficient Near Neighbor Search — Coleman, Segarra, Shrivastava, Smola. NeurIPS 2022; arXiv 2104.03221.
Links: https://arxiv.org/abs/2104.03221 ; https://papers.neurips.cc/paper_files/paper/2022/file/fb44a668c2d4bc984e9d6ca261262cbb-Paper-Conference.pdf
- In-memory (CPU cache) relabeling: Gorder, RCM, DegSort, HubSort, HubCluster, DBG, Corder, and **Porder (profile order)**: "some parts of the search index graph are visited more frequently than others"; edge weights = number of traversals observed on 1 K profiling queries, plugged into a weighted Gorder objective. Objective-based methods (Gorder/RCM/Porder) give 10–40 % speedups (larger at high recall and on large N), P99 −17 % (RCM) / −30 % (Gorder); **Porder improves over Gorder by a further 5–10 %**; lightweight degree/hub orderings underperform except on SIFT10M. Datasets: SIFT 10–100M, DEEP 10–100M, GIST1M, MNIST. Frequency-aware relabeling therefore exists, but only for cache lines in RAM, not for SSD pages or a hot/cold split.

### 1.26 Proxima — Xu, Chen, Hsu, Kang, Zhou, Pinge, Yu, Rosing. IEEE Trans. Computers 2026; arXiv 2312.04257 (Dec 2023).
Link: https://arxiv.org/abs/2312.04257
- Near-storage (3D-NAND) graph ANN accelerator. **Hotness-based reordering**: "The hotter (more frequent) vertices have smaller indices"; "the vertices' visiting frequency is based on the graph search trace from the randomly sampled base data"; the hottest nodes (smallest IDs) are *replicated* as "hot nodes" whose neighbor IDs are stored together with the neighbors' PQ codes so a hot vertex is expanded in one shot. Adding 1 % hot nodes cuts latency 2.2× on 100M datasets; 3 % ≈ 3×; plateau beyond 3 % (default 3 %). This is the only work found that **relabels by visit frequency** (gap (c)) — but in a hardware accelerator, with replication rather than a hot/cold cache, and without adjacency compression.

### 1.27 PLASMA — Oguri, Nishimura, Matsui. VecDB@VLDB 2026; arXiv 2508.15436.
Link: https://arxiv.org/abs/2508.15436 — unified GPU framework isolating topology from memory layout; vertex reordering gives up to 80 % (typically 10–30 %) QPS on GPU. Confirms reordering value on another memory hierarchy.

### 1.28 DQF / "Hot Index" — Zhu, Zhao, Li, Zheng, Qiu, Zhang, Ge. arXiv 2508.07218 (2025, v4 Jun 2026).
Link: https://arxiv.org/abs/2508.07218 — in-memory dual index for skewed, drifting query workloads: a Hot Index of frequently *retrieved* nodes (frequency-weighted construction) + Full Index, hit-rate-triggered promotion/demotion; 2.2–6.9× at 95 % recall, scales to 100M. The workload-driven analogue of a frequency cache, but in RAM and over result frequency, not traversal frequency.

### 1.29 CALL — Jeong, Cho, Park, Kim, Park. arXiv 2509.18670 (Sep 2025).
Link: https://arxiv.org/abs/2509.18670 — disk-based vector DB; groups queries by embedding similarity and caches/prefetches *clusters* shared by a query group (group-aware prefetching); P99 latency −33 %, higher hit ratio. Cluster-level, not node-level.

### 1.30 OrchANN — Chen et al. (12 authors). arXiv 2512.22838 (Dec 2025).
Link: https://arxiv.org/abs/2512.22838 — out-of-core search under *query* skew ("query hotness"); non-replicated SSD partitions + compact memory-resident graph abstraction for boundary reachability; prunes low-value clusters; up to 17.2× QPS, 25× lower latency vs out-of-core baselines.

### 1.31 Disaggregated-memory HNSW (brief)
- SHINE — Widmoser, Kocher, Augsten, arXiv 2507.17647 (2025): https://arxiv.org/abs/2507.17647 — unpartitioned HNSW over RDMA with compute-node caches "logically combined"; policy details not in abstract **[unverified]**.
- d-HNSW — Liu, Ang, Qian, HotStorage 2025, arXiv 2505.11783: https://arxiv.org/abs/2505.11783 — caches a *representative* sampled index on compute nodes; up to 117× latency vs naive; SIFT1M.

### 1.32 Onyx — Rathee, Watson, Zhao, Suh, Popa. arXiv 2604.20401 (Apr 2026).
Link: https://arxiv.org/abs/2604.20401 — oblivious (ORAM) ANN on SSD; inverts the usual design (minimize bandwidth at the ANN layer, access count at the ORAM layer); 1.7–9.9× lower cost. Not a caching paper; listed because it appeared in the searches.

### 1.33 Survey / tutorial — Song, Zhou, Jensen, Xu. "Vector Search for the Future: From Memory-Resident, Static, Heterogeneous Storage, to Cloud-Native Architectures", arXiv 2601.01937 (2026).
Link: https://arxiv.org/abs/2601.01937 — catalogs entry graphs + compact vectors in memory with SSD-resident neighbors (DiskANN, Starling, PageANN); lists **"adaptive and predictive caching"** and tier-aware index co-design as open problems; does not treat graph compression.

### 1.34 ParlayANN / "Scaling Graph-Based ANNS Algorithms to Billion-Size Datasets: A Comparative Analysis" — Manohar, Shen, Blelloch, Dhulipala, Gu, Simhadri, Sun. PPoPP 2024; arXiv 2305.04359.
Links: https://arxiv.org/abs/2305.04359 ; https://github.com/cmuparlay/ParlayANN — in-memory, deterministic parallel builds of DiskANN/HNSW/HCNNG at 1 B scale. **I could not find any paper titled "Scaling graph-based ANNS with SSD"**; if the intended reference is this one, note it is in-memory and has no caching component.

### 1.35 Industrial compressed adjacency (for gap (a))
- Apache Lucene `Lucene99HnswVectorsFormat`: on-disk HNSW neighbor lists are stored per node as a vint count followed by **delta-encoded vint neighbor ordinals**, node offsets via `DirectMonotonicWriter`: https://lucene.apache.org/core/9_10_0/core/org/apache/lucene/codecs/lucene99/Lucene99HnswVectorsFormat.html — uniform compression, no hot/cold distinction, no Elias-Fano. Partitioned Elias-Fano background: Ottaviano & Venturini, SIGIR 2014 (https://en.wikipedia.org/wiki/Partitioned_Elias%E2%80%93Fano_indexes).

---

## 2. What is known about node-access skew during beam search (collected evidence)

1. **Hop-distance skew.** CXL-ANNS: over 1 M queries the most-visited nodes lie within 2–3 hops of the entry node; DiskANN assumes the first 3–4 BFS levels are the hot set; Gorgeous: after "a small number of frequently visited vectors (e.g., those close to the entry node)" accesses become near-uniform; OctopusANN: hop-based caching gives ≈ 9 % I/O reduction on SIFT (i.e., the entry region is a small share of total I/O) but much more on SPACEV.
2. **Phase structure.** GoVector: static-cache hit rate decays as a power law along the search; phase-2 accesses are spatially concentrated around the query but have little cross-query reuse (hence dynamic LFU + adjacent-page prefetch).
3. **Hierarchy as skew.** Turbocharging: HNSW upper layers (≈ 4.3 % of nodes) get ≈ 137 of ≈ 1960 visits/query at 95–100 % hit; layer 0 gets 1824 visits/query at 57 % hit with a 50 % buffer.
4. **Hubness.** Hub-highway paper: k-occurrence skew grows with dimension (ℓ2), hubs are visited disproportionately early; HM-ANN exploits exactly this with degree promotion.
5. **Frequency saturation.** Proxima: replicating the top 1 % / 3 % most-visited vertices yields 2.2× / 3× latency reduction, flat beyond 3 %; DiskANN's BFS builder is capped at 10 % of nodes.
6. **Distribution dependence.** OOD-DiskANN and the 2026 evaluation (static wins on GIST, dynamic wins on SIFT/Deep) show the skew — and thus the right cache — depends on dimension and on the query distribution.

---

## 3. Gap analysis

### (a) Frequency-aware caching combined with compressed graph storage (hot nodes raw / cold nodes compressed)
- **Not done.** No paper stores adjacency lists with Elias-Fano (or any integer code) *and* keeps a hot subset uncompressed. Existing "compression" in disk-graph ANN is of **vectors** (PQ in memory: DiskANN; PQ inline per node: LM-DiskANN, AiSAQ, Proxima; hierarchical vector compression: VeloANN). Adjacency lists are fixed-width 4-byte IDs padded to R (DiskANN sector format) in every system surveyed; only Lucene uses delta-vint, uniformly.
- Gorgeous gives the strongest motivation: adjacency lists are the hot object, and decoupling them from vectors lets a 10 %-of-dataset budget cover 88 % of all lists. Compressing the cold lists (EF/delta after relabeling) would push in-RAM coverage toward 100 % of the topology at the same budget, leaving only vector reads on SSD — nobody has measured this trade-off (decode cost vs I/O saved), nor the interaction with page search / block search that needs whole-page neighbor sets.
- Open sub-questions: (i) compression ratio of Vamana/HNSW adjacency under different ID assignments (random vs BFS vs Gorder vs frequency vs DiskANN++ isomorphic mapping); (ii) whether decoding EF lists in the beam-search inner loop is hidden by SSD latency (PipeANN-style pipelining would make it free); (iii) hybrid formats where hot nodes keep raw lists + full vectors and cold nodes keep EF lists + PQ codes only (an LM-DiskANN/AiSAQ-style inline layout).

### (b) Comparing caching policies (BFS-level vs frequency vs degree vs learned) at equal memory budgets
- **Only fragments exist.** DiskANN ships both a BFS builder and a sample-query frequency builder but never compares them. OctopusANN compares SSSP caching only with MemGraph (and drops it). The 2026 evaluation compares static "graph-prioritized" vs LRU/LFU-dynamic vs hybrid at a single 1 % budget and reports hit rates only. GoVector compares FIFO/Random/LFU for its dynamic tier and a fixed static tier. Gorgeous compares node cache vs nav-index vs adjacency-only cache. Starling only reports that the frequency profiling is expensive to build. Degree/hub-based caches (HM-ANN style) and learned/workload caches (DQF, CALL) have never been benchmarked against BFS or frequency caches on the same disk index, and no study sweeps the budget (0.1 %–10 %) reporting I/Os/query, hit rate and recall together.
- Missing controls: query-distribution shift (ID vs OOD queries, Text2Image), dimensionality (the 2026 evaluation shows the winner flips between SIFT and GIST), cache granularity (node vs 4 KB page vs adjacency-only), and interaction with an in-memory navigation graph (which itself consumes the budget and changes which nodes are hot, as NAVIS notes).

### (c) Relabeling nodes by access frequency for cache locality
- **Partially done, never for a disk-resident hot/cold cache.** Porder (Coleman et al.) weights Gorder's objective with profiled edge traversal counts, for CPU cache lines in RAM (+5–10 % over Gorder). Proxima assigns the smallest IDs to the most-visited vertices, but inside a 3D-NAND accelerator and paired with hot-node replication. DiskANN++'s isomorphic mapping and Starling's block shuffling are purely topological (PQ-nearest neighbors / overlap ratio); GoVector's layout is k-means-based; none uses visit counts.
- Nobody has evaluated: (i) frequency-ordered IDs so that the hot set is a contiguous ID prefix (trivial O(1) "is-hot" test, contiguous hot pages, sequential warm-up), (ii) page packing driven by *co-visitation* (a Porder-like weighted BNP/star packing) rather than graph adjacency, (iii) the effect of frequency relabeling on the size of EF/delta-coded neighbor lists (hot hubs get small IDs → small gaps in every list that points to them), which ties (a) and (c) together, or (iv) the stability of a frequency ordering under updates (FreshDiskANN/NAVIS show the hot set moves with the entry graph).

---

## 4. Verification notes
- All citations above were resolved to an arXiv/ACM/USENIX/NeurIPS page or a downloaded PDF during this review. Items marked **[unverified]** are details I could not read in the primary text (PageANN memory manager; SHINE cache policy).
- "Scaling graph-based ANNS with SSD": no paper with this title was found; closest match is ParlayANN (§1.34), which is in-memory.
- "SmartANNS" and "CXL-ANNS" are hardware-centric; their caching notes come from the ATC'24/ATC'23 papers.
- DiskANN code facts come from tag 0.5.0 of microsoft/DiskANN (`src/pq_flash_index.cpp`); the current `main` is a Rust rewrite and the C++ paths no longer exist there.
