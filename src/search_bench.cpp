// search_bench: greedy beam search (DiskANN/Vamana style) over a proximity
// graph stored in one of several adjacency representations, so that the
// *access-time* cost of graph compression can be measured end to end:
//
//   raw      CSR: uint64 offsets + uint32 neighbors (the exported format)
//   packed   fixed-stride rows, (degree, R slots) per node — the layout used
//            by DiskANN's in-memory index and hnswlib's level-0 links
//   gef      neighbors in gef::U_GEF, offsets either raw uint32 or U_GEF
//   hybrid   "biased" two-tier store: a hot set of rows kept raw, the cold
//            majority in U_GEF (hot set = ids [0,h) after a hot-first
//            relabeling, or an explicit id list)
//
// The search itself is identical across backends: one neighbor-row fetch per
// hop, one distance per unvisited neighbor, a sorted candidate pool of size L.
// Per-node expansion/touch counts can be dumped to derive access-frequency
// profiles for the caching experiments.

#include "gef/gef.hpp"
#include "row_codecs.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// ----------------------------------------------------------------- I/O ----

template <typename T>
std::vector<T> read_raw(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Failed to open " + path);
    const std::streamsize bytes = ifs.tellg();
    if (bytes % static_cast<std::streamsize>(sizeof(T)) != 0)
        throw std::runtime_error(path + ": size not a multiple of element size");
    std::vector<T> v(static_cast<size_t>(bytes) / sizeof(T));
    ifs.seekg(0);
    ifs.read(reinterpret_cast<char*>(v.data()), bytes);
    if (!ifs) throw std::runtime_error("Short read on " + path);
    return v;
}

// DiskANN-style .fbin/.ibin: uint32 npts, uint32 dim, then row-major payload.
template <typename T>
std::vector<T> read_bin(const std::string& path, uint32_t& n, uint32_t& d) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open " + path);
    ifs.read(reinterpret_cast<char*>(&n), 4);
    ifs.read(reinterpret_cast<char*>(&d), 4);
    std::vector<T> v(static_cast<size_t>(n) * d);
    ifs.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    if (!ifs) throw std::runtime_error("Short read on " + path);
    return v;
}

// ------------------------------------------------------------ distances ----

enum class Metric { L2, IP };

__attribute__((target("avx2,fma")))
float l2_avx2(const float* a, const float* b, uint32_t d) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= d; i += 16) {
        __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        acc0 = _mm256_fmadd_ps(x0, x0, acc0);
        acc1 = _mm256_fmadd_ps(x1, x1, acc1);
    }
    for (; i + 8 <= d; i += 8) {
        __m256 x0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(x0, x0, acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc0);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < d; ++i) { float x = a[i] - b[i]; s += x * x; }
    return s;
}

__attribute__((target("avx2,fma")))
float ip_avx2(const float* a, const float* b, uint32_t d) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= d; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= d; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc0 = _mm256_add_ps(acc0, acc1);
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc0);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < d; ++i) s += a[i] * b[i];
    return -s;  // smaller is better everywhere
}

struct Vectors {
    std::vector<float> data;  // n * dim, row-major
    uint32_t n = 0, dim = 0;
    const float* row(uint32_t i) const { return data.data() + static_cast<size_t>(i) * dim; }
};

// -------------------------------------------------------- graph stores ----

struct Row {
    const uint32_t* ptr;
    uint32_t len;
};

struct RawCSR {
    std::vector<uint64_t> off;
    std::vector<uint32_t> nbr;
    static constexpr const char* name() { return "raw"; }
    size_t bytes() const { return off.size() * 8 + nbr.size() * 4; }
    Row row(uint32_t u, std::vector<uint32_t>&) const {
        const uint64_t s = off[u];
        return {nbr.data() + s, static_cast<uint32_t>(off[u + 1] - s)};
    }
};

struct Packed {
    uint32_t stride = 0;  // (R + 1) uint32 per node
    std::vector<uint32_t> data;
    static constexpr const char* name() { return "packed"; }
    size_t bytes() const { return data.size() * 4; }
    Row row(uint32_t u, std::vector<uint32_t>&) const {
        const uint32_t* p = data.data() + static_cast<size_t>(u) * stride;
        return {p + 1, p[0]};
    }
};

template <size_t P>
struct GefNbr {
    gef::U_GEF<uint32_t, P> enc;
    size_t bytes() const { return enc.size_in_bytes(); }
    void build(const std::vector<uint32_t>& v, gef::SplitPointStrategy s) {
        enc = gef::U_GEF<uint32_t, P>(v, s);
    }
    size_t get(size_t start, size_t count, std::vector<uint32_t>& buf) const {
        return enc.get_elements(start, count, buf);
    }
};

// Offsets either raw uint32 (4 B/node) or U_GEF<uint64_t>.
struct OffRaw32 {
    std::vector<uint32_t> v;
    static constexpr const char* name() { return "raw32"; }
    size_t bytes() const { return v.size() * 4; }
    void build(const std::vector<uint64_t>& o) { v.assign(o.begin(), o.end()); }
    uint64_t operator[](size_t i) const { return v[i]; }
};
struct OffGef {
    gef::U_GEF<uint64_t> enc;
    static constexpr const char* name() { return "gef"; }
    size_t bytes() const { return enc.size_in_bytes(); }
    void build(const std::vector<uint64_t>& o) { enc = gef::U_GEF<uint64_t>(o, gef::OPTIMAL_SPLIT_POINT); }
    uint64_t operator[](size_t i) const { return enc[i]; }
};

template <class Nbr, class Off>
struct GefGraph {
    Nbr nbr;
    Off off;
    static constexpr const char* name() { return "gef"; }
    size_t bytes() const { return nbr.bytes() + off.bytes(); }
    Row row(uint32_t u, std::vector<uint32_t>& buf) const {
        const uint64_t s = off[u];
        const uint64_t e = off[u + 1];
        const size_t w = nbr.get(s, e - s, buf);
        return {buf.data(), static_cast<uint32_t>(w)};
    }
};

// Two-tier store. Hot rows raw (contiguous CSR restricted to the hot set),
// cold rows in U_GEF over the concatenation of the cold rows only. With a
// hot-first labeling the tier test is `u < h` and needs no map; otherwise
// a per-node slot array maps ids into the two tiers (charged to space).
template <class Nbr, class Off>
struct Hybrid {
    uint32_t h = 0;
    bool contiguous = true;
    std::vector<uint32_t> slot;      // only when !contiguous: hot idx | 0x80000000, or cold idx
    std::vector<uint32_t> hot_off;   // h + 1
    std::vector<uint32_t> hot_nbr;
    Nbr cold_nbr;
    Off cold_off;
    static constexpr const char* name() { return "hybrid"; }
    size_t bytes() const {
        return hot_off.size() * 4 + hot_nbr.size() * 4 + cold_nbr.bytes() + cold_off.bytes() +
               (contiguous ? 0 : slot.size() * 4);
    }
    Row row(uint32_t u, std::vector<uint32_t>& buf) const {
        uint32_t idx;
        bool hot;
        if (contiguous) { hot = u < h; idx = hot ? u : u - h; }
        else { const uint32_t s = slot[u]; hot = s & 0x80000000u; idx = s & 0x7fffffffu; }
        if (hot) {
            const uint32_t s = hot_off[idx];
            return {hot_nbr.data() + s, hot_off[idx + 1] - s};
        }
        const uint64_t s = cold_off[idx];
        const uint64_t e = cold_off[idx + 1];
        const size_t w = cold_nbr.get(s, e - s, buf);
        return {buf.data(), static_cast<uint32_t>(w)};
    }
};

// Per-row codecs (src/row_codecs.hpp): each row decoded from its own header,
// no global rank/select supports. Baselines for the U_GEF global encoding.
template <class C>
struct RowCodecGraph {
    C codec;
    RowCodecGraph(const std::vector<uint64_t>& off, const std::vector<uint32_t>& nbr) : codec(off, nbr) {}
    static constexpr const char* name() { return C::name(); }
    size_t bytes() const { return codec.bytes(); }
    Row row(uint32_t u, std::vector<uint32_t>& buf) const {
        const rowcodec::Row r = codec.row(u, buf);
        return {r.ptr, r.len};
    }
};

// Two-tier store with a per-row codec for the cold tier (hot-first labeling).
template <class C>
struct HybridRow {
    uint32_t h = 0;
    std::vector<uint32_t> hot_off, hot_nbr;
    std::unique_ptr<C> cold;
    static constexpr const char* name() { return "hybrid-row"; }
    size_t bytes() const { return hot_off.size() * 4 + hot_nbr.size() * 4 + cold->bytes(); }
    Row row(uint32_t u, std::vector<uint32_t>& buf) const {
        if (u < h) {
            const uint32_t s = hot_off[u];
            return {hot_nbr.data() + s, hot_off[u + 1] - s};
        }
        const rowcodec::Row r = cold->row(u - h, buf);
        return {r.ptr, r.len};
    }
};

// ------------------------------------------------------ navigation layers ----

// Optional HNSW-style upper layers over the base graph: nested node sets with
// a small navigable graph each (in layer-local ids). Search greedily descends
// from the top layer's entry to find the base-layer entry point, replacing the
// fixed medoid. Layers are stored raw; their bytes (incl. the per-layer
// base-id -> local-id maps) are charged to the index.
struct Layer {
    std::vector<uint32_t> nodes;      // local -> base id
    std::vector<uint32_t> local;      // base id -> local id (UINT32_MAX if absent)
    std::vector<uint64_t> off;
    std::vector<uint32_t> nbr;
};

struct Hierarchy {
    std::vector<Layer> layers;   // top first
    uint32_t entry_local = 0;    // in layers[0]
    size_t bytes() const {
        size_t b = 0;
        for (auto& l : layers) b += l.nodes.size() * 4 + l.local.size() * 4 + l.off.size() * 8 + l.nbr.size() * 4;
        return b;
    }
};

// Minimal JSON field extraction (the file is produced by build_hierarchy.py).
std::string json_str(const std::string& s, const std::string& key, size_t from, size_t& pos) {
    pos = s.find("\"" + key + "\"", from);
    if (pos == std::string::npos) throw std::runtime_error("hier json: missing " + key);
    size_t q1 = s.find('"', s.find(':', pos) + 1);
    size_t q2 = s.find('"', q1 + 1);
    return s.substr(q1 + 1, q2 - q1 - 1);
}
long json_num(const std::string& s, const std::string& key, size_t from, size_t& pos) {
    pos = s.find("\"" + key + "\"", from);
    if (pos == std::string::npos) throw std::runtime_error("hier json: missing " + key);
    return std::stol(s.substr(s.find(':', pos) + 1));
}

Hierarchy load_hierarchy(const std::string& path, uint32_t N) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Failed to open " + path);
    std::string s((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    Hierarchy h;
    size_t pos = 0, p;
    size_t lv = s.find("\"levels\"");
    size_t cur = lv;
    while (true) {
        size_t nx = s.find("\"nodes\"", cur);
        if (nx == std::string::npos) break;
        Layer L;
        L.nodes = read_raw<uint32_t>(json_str(s, "nodes", cur, p));
        L.off = read_raw<uint64_t>(json_str(s, "offsets", cur, p));
        L.nbr = read_raw<uint32_t>(json_str(s, "neighbors", cur, p));
        if (L.off.size() != L.nodes.size() + 1) throw std::runtime_error("layer offsets/nodes mismatch");
        L.local.assign(N, UINT32_MAX);
        for (uint32_t i = 0; i < L.nodes.size(); ++i) L.local[L.nodes[i]] = i;
        h.layers.push_back(std::move(L));
        cur = p + 1;
    }
    h.entry_local = static_cast<uint32_t>(json_num(s, "entry", 0, pos));
    (void)pos;
    return h;
}

// --------------------------------------------------------------- search ----

struct Cand {
    float d;
    uint32_t id;
    bool expanded;
};

struct QueryStats {
    uint32_t hops = 0;      // rows fetched (node expansions)
    uint32_t dists = 0;     // distance computations
    double micros = 0;
};

struct SearchCtx {
    std::vector<uint32_t> epoch;    // visited marks, per thread
    uint32_t cur_epoch = 0;
    std::vector<Cand> pool;
    std::vector<uint32_t> rowbuf;
    std::vector<uint32_t> expand_cnt;   // optional profiling
    std::vector<uint32_t> touch_cnt;
    bool profile = false;
    explicit SearchCtx(size_t n, size_t maxdeg, bool prof)
        : epoch(n, 0), rowbuf(maxdeg + 1), profile(prof) {
        if (prof) { expand_cnt.assign(n, 0); touch_cnt.assign(n, 0); }
    }
};

template <class G>
QueryStats beam_search(const G& g, const Vectors& base, const float* q, Metric metric,
                       uint32_t entry, uint32_t L, uint32_t k, SearchCtx& ctx,
                       std::vector<uint32_t>& out_ids, const Hierarchy* hier = nullptr) {
    QueryStats st;
    auto dist = [&](uint32_t id) -> float {
        ++st.dists;
        const float* v = base.row(id);
        return metric == Metric::L2 ? l2_avx2(q, v, base.dim) : ip_avx2(q, v, base.dim);
    };
    if (hier && !hier->layers.empty()) {
        // greedy descent through the navigation layers (ef = 1)
        uint32_t cur = hier->layers[0].nodes[hier->entry_local];
        float best = dist(cur);
        for (const Layer& Lr : hier->layers) {
            uint32_t loc = Lr.local[cur];
            bool improved = true;
            while (improved) {
                improved = false;
                ++st.hops;
                const uint64_t e0 = Lr.off[loc], e1 = Lr.off[loc + 1];   // fixed row bounds
                uint32_t next = loc;
                for (uint64_t e = e0; e < e1; ++e) {
                    const uint32_t v = Lr.nodes[Lr.nbr[e]];
                    const float d = dist(v);
                    if (d < best) { best = d; next = Lr.nbr[e]; cur = v; improved = true; }
                }
                loc = next;
            }
        }
        entry = cur;
    }
    auto& pool = ctx.pool;
    pool.clear();
    if (++ctx.cur_epoch == 0) { std::fill(ctx.epoch.begin(), ctx.epoch.end(), 0); ctx.cur_epoch = 1; }
    const uint32_t ep = ctx.cur_epoch;

    auto insert = [&](float d, uint32_t id) {
        if (pool.size() == L && d >= pool.back().d) return;
        auto it = std::lower_bound(pool.begin(), pool.end(), d,
                                   [](const Cand& c, float x) { return c.d < x; });
        pool.insert(it, Cand{d, id, false});
        if (pool.size() > L) pool.pop_back();
    };

    ctx.epoch[entry] = ep;
    insert(dist(entry), entry);
    if (ctx.profile) ++ctx.touch_cnt[entry];

    size_t cur = 0;
    while (cur < pool.size()) {
        if (pool[cur].expanded) { ++cur; continue; }
        pool[cur].expanded = true;
        const uint32_t u = pool[cur].id;
        ++st.hops;
        if (ctx.profile) ++ctx.expand_cnt[u];
        const Row r = g.row(u, ctx.rowbuf);
        // Prefetch the vectors of the unvisited neighbors, as hnswlib/DiskANN do.
        for (uint32_t j = 0; j < r.len; ++j) {
            const uint32_t v = r.ptr[j];
            if (ctx.epoch[v] != ep) _mm_prefetch(reinterpret_cast<const char*>(base.row(v)), _MM_HINT_T0);
        }
        for (uint32_t j = 0; j < r.len; ++j) {
            const uint32_t v = r.ptr[j];
            if (ctx.epoch[v] == ep) continue;
            ctx.epoch[v] = ep;
            if (ctx.profile) ++ctx.touch_cnt[v];
            insert(dist(v), v);
        }
        // restart from the first unexpanded candidate (DiskANN semantics)
        cur = 0;
        while (cur < pool.size() && pool[cur].expanded) ++cur;
    }
    out_ids.clear();
    for (size_t i = 0; i < pool.size() && i < k; ++i) out_ids.push_back(pool[i].id);
    return st;
}

// --------------------------------------------------------------- driver ----

struct Args {
    std::string graph, perm, base, query, gt, backend = "raw", offsets = "raw32";
    std::string hot_ids, profile, json, hier;
    std::string metric = "l2";
    std::vector<uint32_t> Ls{64};
    uint32_t k = 10, threads = 1, repeat = 3, nq = 0, qoff = 0, hot_count = 0, partition = 32000;
    int64_t entry = -1;
    bool approximate = false;
};

std::vector<uint32_t> parse_list(const std::string& s) {
    std::vector<uint32_t> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(static_cast<uint32_t>(std::stoul(tok)));
    return v;
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
            return argv[++i];
        };
        if (s == "--graph") a.graph = next();
        else if (s == "--perm") a.perm = next();
        else if (s == "--base") a.base = next();
        else if (s == "--query") a.query = next();
        else if (s == "--gt") a.gt = next();
        else if (s == "--backend") a.backend = next();
        else if (s == "--offsets") a.offsets = next();
        else if (s == "--metric") a.metric = next();
        else if (s == "--L") a.Ls = parse_list(next());
        else if (s == "--k") a.k = std::stoul(next());
        else if (s == "--threads") a.threads = std::stoul(next());
        else if (s == "--repeat") a.repeat = std::stoul(next());
        else if (s == "--nq") a.nq = std::stoul(next());
        else if (s == "--query-offset") a.qoff = std::stoul(next());
        else if (s == "--entry") a.entry = std::stoll(next());
        else if (s == "--hot-count") a.hot_count = std::stoul(next());
        else if (s == "--hot-ids") a.hot_ids = next();
        else if (s == "--partition") a.partition = std::stoul(next());
        else if (s == "--profile") a.profile = next();
        else if (s == "--hier") a.hier = next();
        else if (s == "--json") a.json = next();
        else if (s == "--approximate") a.approximate = true;
        else throw std::runtime_error("unknown argument " + s);
    }
    if (a.graph.empty() || a.base.empty() || a.query.empty() || a.gt.empty() || a.entry < 0)
        throw std::runtime_error("required: --graph --base --query --gt --entry");
    return a;
}

struct Dataset {
    std::vector<uint64_t> off;
    std::vector<uint32_t> nbr;
    std::vector<uint32_t> perm;   // new -> old (identity if none)
    std::vector<uint32_t> inv;    // old -> new
    Vectors base, query;
    std::vector<int32_t> gt;      // nq_total * gtk, original ids
    uint32_t gtk = 0, nq_total = 0;
    uint32_t entry = 0;           // in the graph's id space
    uint32_t maxdeg = 0;
    Metric metric = Metric::L2;
    Hierarchy hier;              // empty unless --hier
};

Dataset load(const Args& a) {
    Dataset d;
    d.off = read_raw<uint64_t>(a.graph + "_offsets.bin");
    d.nbr = read_raw<uint32_t>(a.graph + "_neighbors.bin");
    if (d.off.empty() || d.off.back() != d.nbr.size()) throw std::runtime_error("offsets/neighbors mismatch");
    const uint32_t N = static_cast<uint32_t>(d.off.size() - 1);
    for (uint32_t i = 0; i < N; ++i) d.maxdeg = std::max<uint32_t>(d.maxdeg, static_cast<uint32_t>(d.off[i + 1] - d.off[i]));

    d.perm.resize(N);
    if (a.perm.empty()) std::iota(d.perm.begin(), d.perm.end(), 0u);
    else {
        d.perm = read_raw<uint32_t>(a.perm);
        if (d.perm.size() != N) throw std::runtime_error("perm size != N");
    }
    d.inv.resize(N);
    for (uint32_t i = 0; i < N; ++i) d.inv[d.perm[i]] = i;
    d.entry = d.inv[static_cast<uint32_t>(a.entry)];

    uint32_t n, dim;
    std::vector<float> raw = read_bin<float>(a.base, n, dim);
    if (n != N) throw std::runtime_error("base vector count != N");
    d.base.n = n; d.base.dim = dim;
    d.base.data.resize(raw.size());
    for (uint32_t i = 0; i < N; ++i)   // vectors follow the labeling
        std::memcpy(d.base.data.data() + static_cast<size_t>(i) * dim,
                    raw.data() + static_cast<size_t>(d.perm[i]) * dim, dim * sizeof(float));
    raw.clear(); raw.shrink_to_fit();

    d.query.data = read_bin<float>(a.query, d.query.n, d.query.dim);
    if (d.query.dim != dim) throw std::runtime_error("query dim != base dim");
    d.gt = read_bin<int32_t>(a.gt, d.nq_total, d.gtk);
    if (d.nq_total != d.query.n) throw std::runtime_error("gt rows != queries");
    d.metric = a.metric == "ip" ? Metric::IP : Metric::L2;
    if (!a.hier.empty()) {
        d.hier = load_hierarchy(a.hier, N);
        std::printf("hierarchy: %zu layers (", d.hier.layers.size());
        for (auto& l : d.hier.layers) std::printf("%zu ", l.nodes.size());
        std::printf(") %zu B\n", d.hier.bytes());
    }
    return d;
}

struct RunResult {
    uint32_t L;
    double recall, qps, mean_us, p50_us, p99_us, hops, dists;
};

template <class G>
std::vector<RunResult> run_backend(const G& g, const Dataset& d, const Args& a, size_t graph_bytes) {
    const uint32_t N = d.base.n;
    const uint32_t q0 = a.qoff;
    const uint32_t nq = a.nq ? std::min<uint32_t>(a.nq, d.nq_total - q0) : d.nq_total - q0;
    const uint32_t T = std::max<uint32_t>(1, a.threads);
    std::vector<std::unique_ptr<SearchCtx>> ctxs;
    for (uint32_t t = 0; t < T; ++t)
        ctxs.emplace_back(std::make_unique<SearchCtx>(N, d.maxdeg, !a.profile.empty()));

    std::vector<RunResult> results;
    std::printf("backend=%s bytes=%zu (%.2f bits/edge incl. offsets)%s\n", G::name(), graph_bytes,
                graph_bytes * 8.0 / d.nbr.size(), d.hier.layers.empty() ? "" : " + hierarchy");
    for (uint32_t L : a.Ls) {
        const uint32_t k = std::min(a.k, L);
        std::vector<std::vector<uint32_t>> res(nq);
        std::vector<double> lat(nq);
        std::vector<QueryStats> qst(nq);
        double best_wall = 1e300;
        for (uint32_t rep = 0; rep < a.repeat + 1; ++rep) {   // rep 0 = warm-up
            const bool prof = (rep == a.repeat) && !a.profile.empty();
            for (auto& c : ctxs) c->profile = prof;   // profile only on the last pass
            auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 16) num_threads(T)
            for (uint32_t i = 0; i < nq; ++i) {
#ifdef _OPENMP
                SearchCtx& ctx = *ctxs[omp_get_thread_num()];
#else
                SearchCtx& ctx = *ctxs[0];
#endif
                auto s0 = std::chrono::steady_clock::now();
                QueryStats st = beam_search(g, d.base, d.query.row(q0 + i), d.metric, d.entry, L, k, ctx, res[i],
                                            d.hier.layers.empty() ? nullptr : &d.hier);
                auto s1 = std::chrono::steady_clock::now();
                st.micros = std::chrono::duration<double, std::micro>(s1 - s0).count();
                lat[i] = st.micros;
                qst[i] = st;
            }
            auto t1 = std::chrono::steady_clock::now();
            const double wall = std::chrono::duration<double>(t1 - t0).count();
            if (rep > 0) best_wall = std::min(best_wall, wall);
        }
        // recall against the top-k ground truth (original ids)
        double hit = 0;
        for (uint32_t i = 0; i < nq; ++i) {
            const int32_t* g_row = d.gt.data() + static_cast<size_t>(q0 + i) * d.gtk;
            for (uint32_t r : res[i]) {
                const uint32_t orig = d.perm[r];
                for (uint32_t j = 0; j < k; ++j) if (static_cast<uint32_t>(g_row[j]) == orig) { hit += 1; break; }
            }
        }
        std::vector<double> sorted = lat;
        std::sort(sorted.begin(), sorted.end());
        RunResult rr;
        rr.L = L;
        rr.recall = hit / (static_cast<double>(nq) * k);
        rr.qps = nq / best_wall;
        rr.mean_us = std::accumulate(lat.begin(), lat.end(), 0.0) / nq;
        rr.p50_us = sorted[nq / 2];
        rr.p99_us = sorted[std::min<size_t>(nq - 1, static_cast<size_t>(nq * 0.99))];
        double hops = 0, dists = 0;
        for (auto& s : qst) { hops += s.hops; dists += s.dists; }
        rr.hops = hops / nq; rr.dists = dists / nq;
        results.push_back(rr);
        std::printf("L=%-4u recall@%u=%.4f  QPS=%.0f  mean=%.1fus p50=%.1fus p99=%.1fus  hops=%.1f dists=%.1f\n",
                    L, k, rr.recall, rr.qps, rr.mean_us, rr.p50_us, rr.p99_us, rr.hops, rr.dists);
    }
    if (!a.profile.empty()) {
        std::vector<uint32_t> ex(N, 0), tc(N, 0);
        for (auto& c : ctxs)
            for (uint32_t i = 0; i < N; ++i) { ex[i] += c->expand_cnt[i]; tc[i] += c->touch_cnt[i]; }
        // written in the graph's (possibly relabeled) id space
        std::ofstream(a.profile + "_expand.u32", std::ios::binary).write(reinterpret_cast<const char*>(ex.data()), N * 4);
        std::ofstream(a.profile + "_touch.u32", std::ios::binary).write(reinterpret_cast<const char*>(tc.data()), N * 4);
        std::printf("wrote %s_expand.u32 / _touch.u32 (last L only)\n", a.profile.c_str());
    }
    if (!a.json.empty()) {
        std::ofstream js(a.json);
        js << "{\"backend\":\"" << a.backend << "\",\"offsets\":\"" << a.offsets << "\",\"graph_bytes\":" << graph_bytes
           << ",\"edges\":" << d.nbr.size() << ",\"nodes\":" << N << ",\"hot_count\":" << a.hot_count
           << ",\"partition\":" << a.partition << ",\"threads\":" << a.threads
           << ",\"hier_bytes\":" << d.hier.bytes() << ",\"runs\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            js << (i ? "," : "") << "{\"L\":" << r.L << ",\"recall\":" << r.recall << ",\"qps\":" << r.qps
               << ",\"mean_us\":" << r.mean_us << ",\"p50_us\":" << r.p50_us << ",\"p99_us\":" << r.p99_us
               << ",\"hops\":" << r.hops << ",\"dists\":" << r.dists << "}";
        }
        js << "]}\n";
    }
    return results;
}

template <size_t P>
void run_gef_family(const Dataset& d, const Args& a) {
    const gef::SplitPointStrategy strat = a.approximate ? gef::APPROXIMATE_SPLIT_POINT : gef::OPTIMAL_SPLIT_POINT;
    const uint32_t N = d.base.n;
    if (a.backend == "gef") {
        auto t0 = std::chrono::steady_clock::now();
        if (a.offsets == "gef") {
            GefGraph<GefNbr<P>, OffGef> g;
            g.nbr.build(d.nbr, strat);
            g.off.build(d.off);
            std::printf("gef build: %.1fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            run_backend(g, d, a, g.bytes());
        } else {
            GefGraph<GefNbr<P>, OffRaw32> g;
            g.nbr.build(d.nbr, strat);
            g.off.build(d.off);
            std::printf("gef build: %.1fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            run_backend(g, d, a, g.bytes());
        }
        return;
    }
    // hybrid
    std::vector<uint8_t> is_hot(N, 0);
    Hybrid<GefNbr<P>, OffRaw32> g;
    if (!a.hot_ids.empty()) {
        std::vector<uint32_t> ids = read_raw<uint32_t>(a.hot_ids);
        for (uint32_t u : ids) is_hot[u] = 1;
        g.contiguous = false;
        g.h = static_cast<uint32_t>(ids.size());
    } else {
        for (uint32_t u = 0; u < a.hot_count; ++u) is_hot[u] = 1;
        g.contiguous = true;
        g.h = a.hot_count;
    }
    std::vector<uint64_t> cold_off{0};
    std::vector<uint32_t> cold_nbr;
    g.hot_off.push_back(0);
    if (!g.contiguous) g.slot.resize(N);
    uint32_t hi = 0, ci = 0;
    for (uint32_t u = 0; u < N; ++u) {
        const uint64_t s = d.off[u], e = d.off[u + 1];
        if (is_hot[u]) {
            g.hot_nbr.insert(g.hot_nbr.end(), d.nbr.begin() + s, d.nbr.begin() + e);
            g.hot_off.push_back(static_cast<uint32_t>(g.hot_nbr.size()));
            if (!g.contiguous) g.slot[u] = hi | 0x80000000u;
            ++hi;
        } else {
            cold_nbr.insert(cold_nbr.end(), d.nbr.begin() + s, d.nbr.begin() + e);
            cold_off.push_back(cold_nbr.size());
            if (!g.contiguous) g.slot[u] = ci;
            ++ci;
        }
    }
    auto t0 = std::chrono::steady_clock::now();
    g.cold_nbr.build(cold_nbr, strat);
    g.cold_off.build(cold_off);
    std::printf("hybrid: hot=%u rows (%zu edges, %zu B raw)  cold=%u rows (%zu edges -> %zu B gef)  build %.1fs\n",
                hi, g.hot_nbr.size(), g.hot_nbr.size() * 4 + g.hot_off.size() * 4, ci, cold_nbr.size(),
                g.cold_nbr.bytes() + g.cold_off.bytes(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    run_backend(g, d, a, g.bytes());
}

int run(int argc, char** argv) {
    Args a = parse(argc, argv);
    auto t0 = std::chrono::steady_clock::now();
    Dataset d = load(a);
    std::printf("loaded N=%u edges=%zu dim=%u queries=%u maxdeg=%u entry=%u(new)  %.1fs\n",
                d.base.n, d.nbr.size(), d.base.dim, d.query.n, d.maxdeg, d.entry,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    if (a.backend == "raw") {
        RawCSR g{d.off, d.nbr};
        run_backend(g, d, a, g.bytes());
    } else if (a.backend == "packed") {
        Packed g;
        g.stride = d.maxdeg + 1;
        g.data.assign(static_cast<size_t>(d.base.n) * g.stride, 0u);
        for (uint32_t u = 0; u < d.base.n; ++u) {
            uint32_t* p = g.data.data() + static_cast<size_t>(u) * g.stride;
            p[0] = static_cast<uint32_t>(d.off[u + 1] - d.off[u]);
            std::memcpy(p + 1, d.nbr.data() + d.off[u], p[0] * 4);
        }
        run_backend(g, d, a, g.bytes());
    } else if (a.backend == "rowef" || a.backend == "rowbic" || a.backend == "rowpack") {
        auto t0 = std::chrono::steady_clock::now();
        if (a.backend == "rowef") {
            RowCodecGraph<rowcodec::RowEF> g(d.off, d.nbr);
            std::printf("%s build: %.1fs\n", g.name(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            run_backend(g, d, a, g.bytes());
        } else if (a.backend == "rowbic") {
            RowCodecGraph<rowcodec::RowBIC> g(d.off, d.nbr);
            std::printf("%s build: %.1fs\n", g.name(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            run_backend(g, d, a, g.bytes());
        } else {
            RowCodecGraph<rowcodec::RowPack> g(d.off, d.nbr);
            std::printf("%s build: %.1fs\n", g.name(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            run_backend(g, d, a, g.bytes());
        }
    } else if (a.backend == "hybrid-rowef") {
        // hot-first labeling required: rows [0,h) raw, the rest RowEF over the cold CSR
        HybridRow<rowcodec::RowEF> g;
        g.h = a.hot_count;
        const uint32_t N = d.base.n;
        g.hot_off.assign(d.off.begin(), d.off.begin() + g.h + 1);
        g.hot_nbr.assign(d.nbr.begin(), d.nbr.begin() + d.off[g.h]);
        std::vector<uint64_t> cold_off(N - g.h + 1);
        for (uint32_t i = g.h; i <= N; ++i) cold_off[i - g.h] = d.off[i] - d.off[g.h];
        std::vector<uint32_t> cold_nbr(d.nbr.begin() + d.off[g.h], d.nbr.end());
        g.cold = std::make_unique<rowcodec::RowEF>(cold_off, cold_nbr, N);   // ids span the whole graph
        std::printf("hybrid-rowef: hot=%u rows (%zu B raw) cold=%u rows (%zu B RowEF)\n", g.h,
                    g.hot_off.size() * 4 + g.hot_nbr.size() * 4, N - g.h, g.cold->bytes());
        run_backend(g, d, a, g.bytes());
    } else if (a.backend == "gef" || a.backend == "hybrid") {
        switch (a.partition) {
            case 1024: run_gef_family<1024>(d, a); break;
            case 4096: run_gef_family<4096>(d, a); break;
            case 8192: run_gef_family<8192>(d, a); break;
            case 32000: run_gef_family<32000>(d, a); break;
            case 128000: run_gef_family<128000>(d, a); break;
            default: throw std::runtime_error("unsupported --partition (1024|4096|8192|32000|128000)");
        }
    } else {
        throw std::runtime_error("unknown backend " + a.backend);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
