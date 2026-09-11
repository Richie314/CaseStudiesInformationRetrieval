# HNSW hierarchy, hubs, biased skip lists, and workload-aware graph indexes — literature notes

Prepared 2026-09-11 for the Ferragina-supervised project on graph-based ANN indexes.
Verification policy: every entry below was checked against at least one primary page (arXiv abstract/HTML/PDF, ACM DL, Springer, NeurIPS/ICML proceedings, or the Semantic Scholar API). Items I could **not** verify are collected in §8 and marked "UNVERIFIED" inline.

---

## 1. The core graph indexes

**HNSW — Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs".** arXiv 1603.09320 (2016), IEEE TPAMI 42(4):824–836 (2020, online 2018). <https://arxiv.org/abs/1603.09320>
- Multi-layer proximity graph; the maximum layer of each element "is selected randomly with an exponentially decaying probability distribution": `l = floor(-ln(unif(0,1)) * mL)`, `mL = 1/ln(M)`, so a node reaches level ≥ l with probability ≈ M^-l (a 1/M "promotion rate", exactly as in Pugh's skip list with p = 1/M).
- The paper states the skip-list analogy explicitly ("similarity of the algorithm to the skip list structure allows straightforward balanced distributed implementation") and argues the hierarchy gives "logarithmic complexity scaling" by starting the search in the sparse top layers (scale separation of link lengths), then greedy descent, then ef-beam search in layer 0.
- Measured: recall vs. queries/s and vs. distance computations on SIFT/GloVe/DEEP-type data against NSW, FLANN, Annoy, FAISS-IVF; growth of the number of hops with N (near-logarithmic); robustness to low-dimensional data where plain NSW degraded to polynomial scaling.

**NSG — Fu, Xiang, Wang, Cai, "Fast approximate nearest neighbor search with the navigating spreading-out graph".** PVLDB 12(5):461–474 (2019); arXiv 1707.00143. <https://www.vldb.org/pvldb/vol12/p461-fu.pdf> · <https://arxiv.org/abs/1707.00143> · code <https://github.com/ZJULearning/nsg>
- Defines the Monotonic RNG (MRNG) and approximates it: start from a kNN graph, use a single *navigating node* (medoid) as fixed entry point, run greedy search from it to collect candidates, prune with an RNG-style rule under a max out-degree, and add a DFS tree from the navigating node to guarantee connectivity. Flat (no hierarchy).
- Measured: recall vs. QPS on SIFT1M/GIST1M and Taobao e-commerce data; index size and construction time vs. HNSW/FANNG/DPG; deployed at Alibaba at billion scale.

**SSG / NSSG — Fu, Wang, Cai, "High dimensional similarity search with satellite system graph: efficiency, scalability, and unindexed query compatibility".** IEEE TPAMI 44(8), 2022; arXiv 1907.06146. <https://arxiv.org/abs/1907.06146>
- Fixes NSG's lack of guarantees for unindexed queries and its excessive sparsity: out-edges spread evenly in angle (a family of MSNETs); NSSG is the practical variant with a sparsity hyperparameter α. Measured on SIFT/GIST/Deep/Crawl-type sets; better speed at high recall and cheaper construction than NSG.

**Vamana / DiskANN — Subramanya, Devvrit, Kadekodi, Krishnaswamy, Simhadri, "DiskANN: fast accurate billion-point nearest neighbor search on a single node".** NeurIPS 2019. <https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf>
- Vamana: random initial graph, medoid entry point, beam search, and the **α-RNG pruning** rule (drop candidate p′ if α·d(p*, p′) ≤ d(p, p′), α ≥ 1; two passes with α = 1 then α ≈ 1.2) that keeps a few long edges and cuts hop counts. Graph + full vectors on SSD, PQ codes in DRAM, re-rank with full precision fetched together with neighbor lists.
- **Relevant detail for this project (§3.4 of the paper):** "we cache the data associated with a subset of vertices in DRAM, *either based on a known query distribution*, or simply by caching all vertices that are C = 3 or 4 hops from the starting point" — i.e. access-frequency-driven caching is already a (barely analysed) knob in DiskANN.
- Measured: recall@k vs. latency/IOPS on SIFT1B/DEEP1B on one 64 GB machine; comparisons to IVF-PQ/FAISS and HNSW on in-memory subsets.

---

## 2. Hubs vs. hierarchy

**Lin & Zhao, "Graph based nearest neighbor search: promises and failures".** arXiv 1904.02077 (2019). <https://arxiv.org/abs/1904.02077>
- Early evidence that "hierarchical structure could not achieve much better logarithmic complexity scaling" in high dimensions; a flat, diversified kNN graph matches HNSW; advantage of the hierarchy vanishes around d ≈ 8–32 (see the hubs paper's restatement below).

**Munyampirwa, Lakshman, Coleman, "Down with the hierarchy: the 'H' in HNSW stands for 'Hubs'".** arXiv 2412.01940 (Dec 2024, v3 Jul 2025); oral at the ICML 2025 Workshop on Vector Databases; also in *Advances in Information Retrieval* (ECIR 2026), DOI 10.1007/978-3-032-21324-2_3. <https://arxiv.org/abs/2412.01940> · <https://openreview.net/forum?id=OJwITuuU3h> · <https://icml.cc/virtual/2025/48994> · <https://dl.acm.org/doi/10.1007/978-3-032-21324-2_3>
- Method: build the full HNSW with hnswlib, *extract its base layer* as a flat index (FlatNav) so the base graph is identical, and compare. Result: in high d the flat graph has "essentially identical" latency/recall; hierarchy helps only for d < 32 (synthetic IID normal d ∈ {4 … 1536}; GloVe, GIST, NYTimes, BigANN 100M, SpaceV 100M, Yandex DEEP 100M, MSMARCO 384-d). Peak build memory drops 38–39 % (BigANN 183→113 GB; DEEP 100→60.7 GB).
- **Hub Highway Hypothesis** and its measurements — this is the closest existing measurement to what a biased design needs: (i) hubness via the k-occurrence distribution N_k and its skewness S_Nk (Radovanović); (ii) the **node access distribution P_m(x_i)** = number of times node x_i is visited by beam search over 10 000 queries (Fig. 7, Fig. 12): "a clear skew indicating that a small subset of nodes are accessed orders of magnitude more frequently than others", increasingly right-skewed with d for ℓ2, much weaker for cosine (anti-hub property); (iii) **temporal concentration**: binning each query's visit sequence (bins of 30) shows hubs dominate "the first 5–10 % of the search steps", GIST highest, GloVe lowest.
- They stop at the diagnosis: no index that *uses* the visit distribution is proposed; they only suggest that pruning and search design "should prioritize hubs".
- Follow-ups found citing it (Semantic Scholar, Sept 2026): LEANN, VIBE benchmark, MIRAGE-ANNS (PACMMOD 2025), "Dynamically detect and fix hardness" (PACMMOD 2025), FaTRQ (DATE 2026), Hope (NVMSA 2026), a SIGIR 2026 replicability study. None re-assigns levels.

**Radovanović, Nanopoulos, Ivanović, "Hubs in space: popular nearest neighbors in high-dimensional data".** JMLR 11:2487–2531 (2010). <https://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf>
- Origin of the k-occurrence / hubness formalism; hubs and anti-hubs as a consequence of distance concentration.

**Xu, Dai, Li, Wang, Yue, "Through the lens of hubness: a revisit on graph-based approximate nearest neighbor search [Experiments & Analysis]".** Proc. ACM Manag. Data (SIGMOD) 2026, DOI 10.1145/3802120. <https://dl.acm.org/doi/10.1145/3802120>
- Hubness explains the long-tail latency of graph ANNS: over-central hubs plus isolated anti-hubs break the greedy heuristic; reinterprets HNSW/NSG/Vamana/… as a taxonomy of hubness-mitigation strategies; proposes hubness-aware pruning; experiments to 100M vectors. Complementary to the hubs paper (hubs as a liability for outliers rather than an asset).

**Wang, Wu, et al., "LEANN: a low-storage vector index".** arXiv 2506.08276 (2025). <https://arxiv.org/abs/2506.08276>
- "High-degree preserving pruning": rank nodes by degree, keep the top β ≈ 2 % at full degree M, cap everyone else at m = M/5; half the edges with search cost comparable to the unpruned graph (random pruning needs up to 1.8× more recomputation). Evidence that hub identity (by degree) is the structural asset worth protecting.

**Coleman, Segarra, Smola, Shrivastava, "Graph reordering for cache-efficient near neighbor search".** NeurIPS 2022; arXiv 2104.03221. <https://arxiv.org/abs/2104.03221>
- Formalises beam search as a cache-hit maximisation problem and evaluates Gorder, RCM, degree sort, **hub sort / hub cluster** (hubs = above-average degree), DBG on HNSW/Vamana-type graphs. All orderings are *structural* (degree/adjacency); none is driven by a measured query-visit profile.

---

## 3. Non-random level / "highway" assignment — what exists

**HM-ANN — Ren, Zhang, Li, "HM-ANN: efficient billion-point nearest neighbor search on heterogeneous memory".** NeurIPS 2020. <https://proceedings.neurips.cc/paper/2020/hash/788d986905533aba051261497ecffcbb-Abstract.html>
- Generalises HNSW construction into *top-down insertion* (build L0 in slow memory/PMem) plus **bottom-up promotion**: "promotes elements with the highest degree in L0 into L1", then promotes high-degree nodes upward with rate 1/M ("the similar promotion rate setting is used in HNSW and typical skip list"); L1 gets degree 2M and is sized to the available DRAM. Justification is explicitly the hub argument ("most of the shortest paths between nodes flow through hubs").
- Measured: at billion scale 2× / 5.8× faster than HNSW / NSG at equal recall; **high-degree promotion vs. random promotion (same number of promoted nodes): 1.8×, 4.3×, 3.9× faster to reach 95 %, 99 %, 99.5 % recall**; index 5–13 % larger. This is the strongest published evidence that a *non-random* level rule beats the skip-list coin, but the weight is degree, not access frequency, and the setting is tiered memory.

**H3NSW — Wu, Wang, Xu, "H3NSW: a hybrid hierarchical graph index with hub-first construction and density-aware layering".** ICIC 2026 (Springer LNCS), DOI 10.1007/978-981-92-3432-5_26. <https://link.springer.com/chapter/10.1007/978-981-92-3432-5_26>
- Abstract only (paywalled): three layers (sampling / navigation / data); hubs "identified based on their neighborhood radii" form the navigation layer; the data layer is refined by an "LSG" algorithm. Geometry-driven, not workload-driven. Details and numbers not verified.

**Elliott & Clark, "The impacts of data, ordering, and intrinsic dimensionality on recall in HNSW".** ICTIR 2024, DOI 10.1145/3664190.3672512; arXiv 2405.17813. <https://arxiv.org/abs/2405.17813> · <https://dl.acm.org/doi/10.1145/3664190.3672512>
- Insertion order informed by pointwise LID "can shift recall by up to 12 percentage points" (hnswlib +2.6 pp, FAISS +6.2 pp for descending-LID insertion). Levels stay random; the lever is that early-inserted points become hubs by preferential attachment.

**"Dual-branch HNSW approach with skip bridges and LID-driven optimization".** arXiv 2501.13992 (2025). <https://arxiv.org/abs/2501.13992>
- LID-based insertion/promotion, a second branch entered from the opposite direction, and "skip bridges" that bypass layers; claims +18 % (NLP) / +30 % (CV) recall and −20 % build time; ablation says LID-based insertion is the main contributor. Preprint; claims not independently reproduced.

**Pakhunov, "Mycelium-Index: a streaming ANN index with myelial edge decay, traffic-driven reinforcement, and adaptive living hierarchy".** arXiv 2604.11274 (Apr 2026, single-author preprint). <https://arxiv.org/abs/2604.11274>
- **The closest thing to a "biased HNSW" found.** Replaces random levels by a per-node `query_use_count` "incremented exclusively during real search queries"; Level 1 = top 2 % of nodes by count (k = 16 edges), Level 2 = top 0.1 % (k = 32); refreshed every 10 000 / 50 000 insertions; search descends L2 → L1 → base.
- Measured only on SIFT-1M: ablation +0.164 recall@10 over a base graph with random entry; recall 0.962 vs. hnswlib (M = 16) 0.965 at ef = 192 with 5.2× less RAM; in a 100 %-turnover streaming benchmark, refreshing the hierarchy each cycle is the single biggest fix (0.122 → 0.738 recall@5), attributed to stale entry points.
- Gaps: no comparison against a *random-level* hierarchy of the same size on the same base graph; no analysis of the visit-count distribution; no level rule derived from weights (fixed 2 % / 0.1 % cutoffs); no theory; one dataset; not peer-reviewed.

**Baranchuk, Persiyanov, Sinitsin, Babenko, "Learning to route in similarity graphs".** ICML 2019, PMLR 97:475–484. <https://proceedings.mlr.press/v97/baranchuk19a.html>
- Learned per-vertex representations guide greedy routing out of local minima; a learned *routing policy*, not a learned hierarchy.

**Oguri & Matsui, "Theoretical and empirical analysis of adaptive entry point selection for graph-based ANNS".** arXiv 2402.04713 (2024). <https://arxiv.org/abs/2402.04713>
- b-monotonic paths / B-MSNET; proves adaptive (k-means-candidate) entry points have a better upper bound than a fixed central entry under general conditions; measured accuracy/speed/memory incl. OOD queries.

**Ruan et al., "Empowering graph-based ANNS with adaptive awareness capabilities" (GATE).** KDD 2025; arXiv 2506.15986. <https://arxiv.org/abs/2506.15986>
- Extracts a small set of **hub nodes as candidate entry points**, trains a contrastive two-tower model (graph structure + query features) to pick one per query, and builds a navigation graph on the hubs; 1.2–2.0× speed-up over SOTA graphs. Query-aware entry, static hubs.

UNVERIFIED: a search-engine summary described a "FastHNSW+Degree" variant that picks upper-layer nodes bottom-up by total out-degree (better on Crawl, worse on Deep1M, +2.8 % / +8.0 % build time). I checked arXiv 2410.01231 (Yang et al., PVLDB 2025), 2502.18113 and 2310.20419 (Ono & Matsui) and found no such section; treat as unsourced until located.

---

## 4. Query-workload-aware graph indexes and access-frequency use

**RoarGraph — Chen, Zhang, He, Jing, Wang.** PVLDB 17(11), 2024; arXiv 2408.08933. <https://arxiv.org/abs/2408.08933> · <https://dl.acm.org/doi/10.14778/3681954.3681959>
- Builds the graph "under the guidance of query distribution": a bipartite query–base graph projected onto base vectors, for cross-modal/OOD workloads; up to 3.56× faster at 90 % recall. Workload enters at *construction*, not as level bias. **OOD-DiskANN** (Jaiswal et al., arXiv 2211.12850, <https://arxiv.org/abs/2211.12850>) is the earlier query-sample-guided Vamana build; **Hua et al., "Dynamically detect and fix hardness"** (PACMMOD 3(6), 2025; arXiv 2510.22316, <https://arxiv.org/abs/2510.22316>) extends RoarGraph.

**CleANN — Zhang, Wei, Engels, Shun.** arXiv 2507.19802 (2025). <https://arxiv.org/abs/2507.19802>
- "Workload-aware linking of diverse search-tree descendants to combat distribution shift" + query-adaptive neighborhood consolidation for fully dynamic workloads; 7–1200× throughput on million-scale dynamic benchmarks (56 hyper-threads).

**Zhang & Miller, "Distribution-aware exploration for adaptive HNSW search".** SIGMOD 2026; arXiv 2512.06636. <https://arxiv.org/abs/2512.06636>
- Per-query `ef` from a statistical model of the query/database similarity distribution; index untouched; up to 4× lower latency than learned early-termination, 50× / 100× cheaper offline.

**Zhu et al., "Accelerating high-dimensional nearest neighbor search with dynamic query preference".** arXiv 2508.07218 (2025, v2 2026). <https://arxiv.org/abs/2508.07218>
- A **Hot Index** (compact graph over "frequently accessed nodes", from online frequency counters of nodes appearing in results) beside the Full Index, shared priority queue, early termination; periodic promotion/demotion; 2.2–6.9× at 95 % recall, 100M scale. This is workload-biased *two-tier* indexing — conceptually a 2-level biased skip list whose weight is "how often is x an answer", but without the level-rule/analysis.

**OrchANN — Chen et al., "Hierarchical orchestration for skewed out-of-core vector search".** arXiv 2512.22838 (2025/26). <https://arxiv.org/abs/2512.22838>
- Exploits *query hotness over clusters* (memory-resident abstraction over hot partitions, pruning cold ones); up to 17.2× QPS.

**Disk/tier caching using visit frequency.** DiskANN caches by "known query distribution" or BFS depth (above). **Starling** (Wang et al., SIGMOD 2024; arXiv 2401.02116, <https://arxiv.org/abs/2401.02116>) instead builds its in-memory navigation graph from a *random sample* used only for entry points. **GoVector** (arXiv 2508.15694, <https://arxiv.org/abs/2508.15694>) keeps a static cache of entry points and "frequently accessed neighbors" plus a dynamic locality cache (−46 % I/O). **Huang et al., ICDE 2026** (arXiv 2605.10090, <https://arxiv.org/abs/2605.10090>) report "high access locality in vector search in our online services" and exploit it for CCD-aware thread placement. UNVERIFIED: a survey snippet says PageANN (arXiv 2509.25487) has a sample-query warm-up to pick cached nodes; the abstract does not say so.

---

## 5. Biased skip lists and self-adjusting search structures

**Bagchi, Buchsbaum, Goodrich, "Biased skip lists".** Algorithmica 42(1):31–48 (2005), DOI 10.1007/s00453-004-1138-6. <https://link.springer.com/article/10.1007/s00453-004-1138-6> · <https://dl.acm.org/doi/10.1007/s00453-004-1138-6>
- Each item i has weight w_i, W = Σ w_i. **Level rule:** rank r_i = ⌊log_a w_i⌋ and the item's height satisfies h_i ≥ r_i (heavy items are forced into high levels). *(a,b)-biased skip list* (deterministic): invariants (1) at most b consecutive items of height i, (2) for every node x and every level i between rank(x) and height(x), at least a nodes of height i−1 separate x from the next node of height ≥ i — the biased analogue of Munro–Papadakis–Sedgewick's deterministic skip list. *Randomized* version: height = rank + geometric random extra. **Bound:** access time O(1 + log(W/w_i)) worst-case (deterministic) or expected (randomized): the search descends only the ≈ log_a W − log_a w_i levels above the item's rank, O(1) horizontal steps per level. Supports insert/delete/reweight/join/split with corresponding weighted bounds. (Conference version usually cited as ISAAC 2002 — not verified here.) Level-rule text confirmed via the Vadrevu–Xing–Aref survey (below).

**Ergun, Sahinalp, Sharp, Sinha, "Biased skip lists for highly skewed access patterns".** ALENEX 2001, LNCS 2153:216–229, DOI 10.1007/3-540-44808-X_18. <https://link.springer.com/chapter/10.1007/3-540-44808-X_18> · PDF via <https://homes.luddy.indiana.edu/fergun/publications.html>
- Motivated by IP lookup/packet classification where access probability decays geometrically with recency. Rank r(k) = number of distinct keys accessed since the last access to k (the *working-set number*); keys sorted by rank are cut into classes C_1, C_2, … with |C_i| = 2^{i−1}; height is determined by class plus randomness; after an access the key moves to rank 1 (move-to-front). Expected search O(log r(k)) — a working-set bound. Experiments vs. plain skip list and a binary trie on 32/48/64-bit keys: BSL wins under strong bias on 64-bit keys, not always vs. the trie on shorter keys.

**Vadrevu, Xing, Aref, "The ubiquitous skiplist: a survey of what cannot be skipped about the skiplist and its applications in data systems".** ACM Computing Surveys 2025, DOI 10.1145/3736754; arXiv 2403.04582; tutorial arXiv 2304.09983. <https://arxiv.org/abs/2403.04582>
- Section on deterministic/adaptive/biased skiplists: the two papers above, the SASL (Ciriani et al.), splay-list; a clean source for the level rules quoted here.

**Bent, Sleator, Tarjan, "Biased search trees".** SIAM J. Comput. 14(3):545–568 (1985). <https://dx.doi.org/10.1137/0214041>
- The tree-shaped origin of the O(log(W/w_i)) weighted access bound. **Sleator & Tarjan, "Self-adjusting binary search trees"**, JACM 32(3):652–686 (1985) <https://dl.acm.org/doi/10.1145/3828.3835>: static-optimality and **working-set** theorems for splay trees (no weights known in advance).

**Bose, Douïeb, Langerman, "Dynamic optimality for skip lists and B-trees".** SODA 2008, pp. 1106–1114. <https://dl.acm.org/doi/10.5555/1347082.1347203>
- Self-adjusting skip lists/B-trees with working-set-type guarantees; shows skip-list height is the right knob for adaptivity.

**Ciriani, Ferragina, Luccio, Muthukrishnan, "Static optimality theorem for external memory string access".** FOCS 2002 (DOI 10.1109/SFCS.2002.1181945); extended as "Self-adjusting data structures for external memory string access". <http://eprints.adm.unipi.it/2070/>
- The SASL: a self-adjusting skip list whose node heights are promoted/demoted by access pattern (bands of exponentially growing size), achieving static optimality (entropy bound) in the I/O model — the supervisor's own precedent for "heights from access frequency" and for an I/O-cost version of the argument.

**Aksenov, Alistarh, Drozdova, Mohtashami, "The splay-list: a distribution-adaptive concurrent skip-list".** DISC 2020 (LIPIcs 179:3); Distributed Computing 2023; arXiv 2008.01009. <https://arxiv.org/abs/2008.01009> · <https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.DISC.2020.3>
- "The height of each element adapts dynamically to its access rate: popular elements move up, rarely-accessed elements decrease in height"; order-optimal amortized bounds for a subset of operations; concurrent implementation beats classic skip lists under skew.

**Avin, Salem, Schmid, "Working set theorems for routing in self-adjusting skip list networks".** IEEE (2020 copyright notice; INFOCOM-style venue not confirmed from the PDF). <https://eprints.cs.univie.ac.at/6322/2/self-adj-skip-list-univie-eprints.pdf>
- Skip lists as *network topologies*: routing cost between two nodes bounded by the log of their working-set number; SASL2 randomized/sequential, then distributed and deterministic variants. The nearest theoretical analogue to "a navigable graph whose hierarchy adapts to who is being searched for".

---

## 6. IR static caching background (hot/cold tiering by log frequency)

- **Baeza-Yates, Gionis, Junqueira, Murdock, Plachouras, Silvestri, "The impact of caching on search engines".** SIGIR 2007, pp. 183–190. <https://dl.acm.org/doi/10.1145/1277741.1277775> — one-year query log; static vs. dynamic caching; caching posting lists gives higher hit rates than caching results; a new static posting-list cache selection algorithm (frequency/size trade-off).
- **Fagni, Perego, Silvestri, Orlando, "Boosting the performance of web search engines: caching and prefetching query results by exploiting historical usage data".** ACM TOIS 24(1), 2006. <https://dl.acm.org/doi/10.1145/1125857.1125859> — **SDC**: a static read-only cache filled from the most frequent historical queries plus a dynamic part; the direct ancestor of "bias the structure by measured frequency, keep a dynamic part for drift".
- **Baeza-Yates & Saint-Jean, "A three level search engine index based in query log distribution".** SPIRE 2003, LNCS 2857. <https://link.springer.com/chapter/10.1007/978-3-540-39984-1_5> — index tiers (memory / disk / cold) chosen from the query-log term distribution.
- **Zhang, Long, Suel, "Performance of compressed inverted list caching in search engines".** WWW 2008. <https://dl.acm.org/doi/10.1145/1367497.1367550> — list caching vs. compression trade-offs under real logs.

Take-away for the design: IR settled long ago that (a) log-derived static tiers beat pure LRU when the distribution is stable, (b) a small dynamic tier absorbs drift, (c) the right cache unit is the *shared intermediate object* (posting list), not the final answer — the ANN analogue of the posting list is the hub node traversed by many queries, not the returned neighbour.

---

## 7. Gap analysis

**Q1. Is a "biased HNSW" (levels from measured access frequency, biased-skip-list style) novel?**
Largely yes, with three near neighbours that must be cited and differentiated:
1. **HM-ANN (2020)** — non-random promotion, but by *degree*, motivated by tiered memory; it does show random promotion loses 1.8–4.3× at equal upper-layer size, which is the best existing argument that the coin flip is not sacred.
2. **Mycelium-Index (2026 preprint)** — promotion by online query traffic with fixed 2 % / 0.1 % cutoffs; SIFT-1M only, no random-level control, no distributional analysis, no theory.
3. **Hot Index (Zhu et al. 2025)** — a two-tier "hot graph" from result frequencies, i.e. a 2-level biased structure without the skip-list framing.
Nobody has (i) stated the level rule as h_i = ⌊log_M(w_i / w_min)⌋ (+ geometric tail) with w_i a measured visit or hit frequency, (ii) connected it to the O(1 + log(W/w_i)) / static-optimality bounds (Bagchi et al.; Ciriani–Ferragina et al.), or (iii) tested it against the random hierarchy *and* against a flat graph across intrinsic dimensions.

**Q2. Has anyone measured the node-visit frequency distribution of beam search and used it?**
- *Measured:* Munyampirwa et al. (P_m(x_i) on HNSW beam search; skew grows with d for ℓ2; MSMARCO long tail "orders of magnitude"; hubs concentrated in the first 5–10 % of steps). Nothing comparable for Vamana or NSG was found, and no fitted tail model (Zipf exponent, top-k coverage) was reported.
- *Used:* only for caching/tiering — DiskANN's "known query distribution" cache, GoVector's static hot-neighbor cache, OrchANN's cluster hotness — and, in a crude form, Mycelium's cutoffs. Not for layer assignment with a principled rule, not for memory layout (Coleman et al. order by structure only), not for pruning (LEANN uses degree).

**Two conceptual pitfalls to address in the paper.**
- *Which weight?* Three candidates: k-occurrence/in-degree (static, data-only; ≈ HM-ANN/LEANN), *visit* frequency under a workload (dynamic; what the hubs paper measured), and *hit* frequency (how often x is a returned neighbour; the biased-skip-list "target" semantics, Hot Index). Visit frequency is partly self-fulfilling (promoting a node raises its visits), so the design should either use hit frequency or iterate visit-based promotion to a fixed point and report stability.
- *Where can it pay?* The hubs paper shows the hierarchy itself buys little in high d on in-memory hardware, so a biased hierarchy should be pitched where the hierarchy matters: low/moderate intrinsic dimension (d < 32, filtered or structured embeddings), tiered memory (HM-ANN/DiskANN: upper layers pinned in DRAM, so promotion = what to cache), skewed workloads (Zipfian query mixes), and streaming with drift (Mycelium's refresh finding; SDC's static+dynamic split).

**Q3. Cleanest experiment.**
1. *Measurement study (contribution on its own):* hnswlib, FlatNav, Vamana (DiskANN), NSG on SIFT1M, GIST1M, DEEP10M, MSMARCO/OpenAI-1536, plus synthetic d ∈ {4…1024}; 100k queries at fixed ef; log per-node visit counts and hit counts. Report log-log rank plots, fitted Zipf exponent, Gini, top-1 %/2 %/5 % visit coverage, Spearman correlation with degree and k-occurrence, and stability across disjoint query halves (train/test split of the log).
2. *Controlled intervention on one shared base graph* (the hubs-paper protocol: build with hnswlib, keep layer 0 fixed): hierarchies (a) random (HNSW), (b) degree-biased (HM-ANN), (c) access-biased with h_i = ⌊log_M(c·w_i)⌋ using train-half weights, (d) flat. Equal upper-layer node budget and memory. Metrics: recall vs. QPS, hops and distance computations to first entry into the true kNN neighbourhood, p99 latency, and on a DRAM+SSD or DRAM+PMem setup the fraction of visits served from the fast tier.
3. *Skew and drift knobs:* Zipfian cluster-sampled queries with exponent ∈ {0, 0.6, 1.0, 1.2}; evaluate (c) on the test half and on a shifted distribution; theory predicts gain ∝ entropy gap between uniform and measured weights — check it.
4. *Same weights, other levers:* visit-ordered memory layout vs. Coleman's hub sort; visit-based DiskANN cache list vs. BFS-hop cache; hit-frequency-aware pruning vs. LEANN's degree rule.

---

## 8. Unverified / partially verified items

- "FastHNSW+Degree" degree-selected upper layers (Crawl/Deep1M, +2.8 %/+8.0 % build time): no source located.
- PageANN sample-query warm-up cache: not in the abstract; MARGO not checked.
- H3NSW mechanism and results: abstract only (paywalled).
- Bagchi et al. conference version at ISAAC 2002: commonly cited, not confirmed from a primary page.
- Avin–Salem–Schmid venue: PDF carries © 2020 IEEE; conference name not confirmed.
- Dual-branch LID HNSW recall claims (+18 %/+30 %): preprint, not reproduced.
- OOD-DiskANN details taken from the arXiv listing only.
