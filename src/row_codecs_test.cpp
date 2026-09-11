// row_codecs_test: build every codec in row_codecs.hpp over a CSR graph,
// verify exact row round-trips, and time whole-row decoding.
//
//   ./row_codecs_test <prefix>       loads <prefix>_offsets.bin / <prefix>_neighbors.bin
//   ./row_codecs_test --synthetic    only the built-in synthetic self-test
//
// The synthetic self-test (random sorted rows with duplicates and edge cases
// over small universes) always runs first.
//
// Timing: 2,000,000 pre-generated uniform random node ids (std::mt19937),
// decode each row and sum its values into a volatile sink. "random" draws
// from all N nodes (memory-latency bound); "hot" draws the same 2M ids from a
// 1024-node working set that stays in cache, isolating the decode CPU cost.
// Each timing is the best of three runs. The raw CSR (pointer into nbr, no
// copy) goes through the identical loop as the baseline.

#include "row_codecs.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using rowcodec::Row;

namespace {

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

// Baseline: the CSR itself, same interface, returns a pointer into nbr.
class RawCSR {
public:
    RawCSR(const std::vector<uint64_t>& off, const std::vector<uint32_t>& nbr)
        : off_(off.data()), nbr_(nbr.data()), bytes_(off.size() * 8 + nbr.size() * 4) {}
    Row row(uint32_t u, std::vector<uint32_t>&) const {
        return {nbr_ + off_[u], static_cast<uint32_t>(off_[u + 1] - off_[u])};
    }
    size_t bytes() const { return bytes_; }
    static constexpr const char* name() { return "raw"; }

private:
    const uint64_t* off_;
    const uint32_t* nbr_;
    size_t bytes_;
};

struct Result {
    const char* name;
    size_t bytes;
    double bits_per_edge;
    double build_s;
    bool ok;
    int64_t first_fail;
    double ns_random;
    double ns_hot;
};

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

template <class C>
int64_t verify(const C& c, const std::vector<uint64_t>& off, const std::vector<uint32_t>& nbr,
               std::vector<uint32_t>& buf) {
    const size_t N = off.size() - 1;
    for (size_t u = 0; u < N; ++u) {
        Row r = c.row(static_cast<uint32_t>(u), buf);
        const uint64_t deg = off[u + 1] - off[u];
        if (r.len != deg) return static_cast<int64_t>(u);
        const uint32_t* x = nbr.data() + off[u];
        for (uint32_t i = 0; i < r.len; ++i)
            if (r.ptr[i] != x[i]) return static_cast<int64_t>(u);
    }
    return -1;
}

template <class C>
double time_rows(const C& c, const std::vector<uint32_t>& ids, std::vector<uint32_t>& buf) {
    static volatile uint64_t sink = 0;
    double best = 1e300;
    for (int rep = 0; rep < 3; ++rep) {
        const double t0 = now_s();
        uint64_t acc = 0;
        for (uint32_t u : ids) {
            Row r = c.row(u, buf);
            uint64_t s = 0;
            for (uint32_t i = 0; i < r.len; ++i) s += r.ptr[i];
            acc += s;
        }
        const double t1 = now_s();
        sink = sink + acc;
        best = std::min(best, (t1 - t0) * 1e9 / static_cast<double>(ids.size()));
    }
    return best;
}

template <class C>
Result run_codec(const std::vector<uint64_t>& off, const std::vector<uint32_t>& nbr,
                 const std::vector<uint32_t>& ids_random, const std::vector<uint32_t>& ids_hot,
                 std::vector<uint32_t>& buf) {
    Result res{};
    res.name = C::name();
    const double t0 = now_s();
    C c(off, nbr);
    res.build_s = now_s() - t0;
    res.bytes = c.bytes();
    res.bits_per_edge = nbr.empty() ? 0.0 : 8.0 * static_cast<double>(res.bytes) / static_cast<double>(nbr.size());
    std::printf("%-8s %12zu bytes  %7.3f bits/edge  build %.3f s\n", res.name, res.bytes, res.bits_per_edge,
                res.build_s);
    std::fflush(stdout);
    res.first_fail = verify(c, off, nbr, buf);
    res.ok = res.first_fail < 0;
    if (res.ok)
        std::printf("%-8s verify OK\n", res.name);
    else
        std::printf("%-8s verify FAIL at node %lld\n", res.name, static_cast<long long>(res.first_fail));
    std::fflush(stdout);
    res.ns_random = time_rows(c, ids_random, buf);
    res.ns_hot = time_rows(c, ids_hot, buf);
    std::printf("%-8s %8.1f ns/row (random)  %8.1f ns/row (hot)\n", res.name, res.ns_random, res.ns_hot);
    std::fflush(stdout);
    return res;
}

// ----------------------------------------------------------------- synthetic
void csr_from_rows(const std::vector<std::vector<uint32_t>>& rows, std::vector<uint64_t>& off,
                   std::vector<uint32_t>& nbr) {
    off.assign(1, 0);
    nbr.clear();
    for (const auto& r : rows) {
        nbr.insert(nbr.end(), r.begin(), r.end());
        off.push_back(nbr.size());
    }
}

template <class C>
bool synthetic_one(const std::vector<uint64_t>& off, const std::vector<uint32_t>& nbr, uint64_t N) {
    std::vector<uint32_t> buf(256 + 1);
    C c(off, nbr);
    const int64_t f = verify(c, off, nbr, buf);
    if (f >= 0) {
        std::printf("  synthetic N=%llu %-8s FAIL at row %lld\n", static_cast<unsigned long long>(N), C::name(),
                    static_cast<long long>(f));
        return false;
    }
    std::printf("  synthetic N=%llu %-8s OK (%zu rows, %zu edges, %.2f bits/edge)\n",
                static_cast<unsigned long long>(N), C::name(), off.size() - 1, nbr.size(),
                nbr.empty() ? 0.0 : 8.0 * c.bytes() / nbr.size());
    return true;
}

bool synthetic_test() {
    bool all_ok = true;
    std::mt19937 rng(12345);
    for (uint64_t N : {1ull, 2ull, 7ull, 1000ull, 1ull << 20}) {
        std::vector<std::vector<uint32_t>> rows;
        const uint32_t maxid = static_cast<uint32_t>(N - 1);
        // edge cases
        rows.push_back({});
        rows.push_back({0});
        rows.push_back({maxid});
        rows.push_back({0, 0, 0, 0});
        rows.push_back(std::vector<uint32_t>(10, maxid));
        rows.push_back({0, maxid});
        {
            std::vector<uint32_t> r;
            for (uint32_t i = 0; i < 64 && i < N; ++i) r.push_back(i);
            rows.push_back(r);
            std::vector<uint32_t> r2;
            for (uint32_t i = 0; i < 64 && i < N; ++i) r2.push_back(maxid - (63 - i) < N ? maxid - (63 - i) : 0);
            std::sort(r2.begin(), r2.end());
            rows.push_back(r2);
        }
        // random rows: degree 0..255, uniform ids, duplicates allowed
        std::uniform_int_distribution<uint32_t> deg(0, 255);
        std::uniform_int_distribution<uint32_t> id(0, maxid);
        for (int k = 0; k < 4000; ++k) {
            std::vector<uint32_t> r(deg(rng) % (k % 2 ? 256 : 64));
            for (auto& v : r) v = id(rng);
            std::sort(r.begin(), r.end());
            rows.push_back(r);
        }
        // pad with empty rows so that the row count equals N (ids must be < N)
        while (rows.size() < N && rows.size() < 5000) rows.push_back({});
        // every row index must be a valid node: N nodes => exactly N rows
        rows.resize(static_cast<size_t>(std::max<uint64_t>(N, 1)));
        // for large N, the rows beyond those generated are empty, which is fine
        std::vector<uint64_t> off;
        std::vector<uint32_t> nbr;
        csr_from_rows(rows, off, nbr);
        all_ok &= synthetic_one<rowcodec::RowEF>(off, nbr, N);
        all_ok &= synthetic_one<rowcodec::RowBIC>(off, nbr, N);
        all_ok &= synthetic_one<rowcodec::RowPack>(off, nbr, N);
    }
    std::printf("synthetic self-test: %s\n\n", all_ok ? "OK" : "FAIL");
    std::fflush(stdout);
    return all_ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <prefix> | --synthetic\n", argv[0]);
        return 2;
    }
    if (!synthetic_test()) return 1;
    if (std::string(argv[1]) == "--synthetic") return 0;

    const std::string prefix = argv[1];
    const auto off = read_raw<uint64_t>(prefix + "_offsets.bin");
    const auto nbr = read_raw<uint32_t>(prefix + "_neighbors.bin");
    if (off.empty() || off.back() != nbr.size()) throw std::runtime_error("offsets/neighbors mismatch");
    const size_t N = off.size() - 1;
    uint64_t maxdeg = 0;
    for (size_t u = 0; u < N; ++u) maxdeg = std::max(maxdeg, off[u + 1] - off[u]);
    std::printf("graph %s: N=%zu nnz=%zu maxdeg=%llu\n\n", prefix.c_str(), N, nbr.size(),
                static_cast<unsigned long long>(maxdeg));

    constexpr size_t kQueries = 2'000'000;
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> uid(0, static_cast<uint32_t>(N - 1));
    std::vector<uint32_t> ids_random(kQueries);
    for (auto& u : ids_random) u = uid(rng);
    std::vector<uint32_t> hot_set(1024);
    for (auto& u : hot_set) u = uid(rng);
    std::uniform_int_distribution<uint32_t> hidx(0, 1023);
    std::vector<uint32_t> ids_hot(kQueries);
    for (auto& u : ids_hot) u = hot_set[hidx(rng)];

    std::vector<uint32_t> buf(static_cast<size_t>(maxdeg) + 1);
    std::vector<Result> results;
    results.push_back(run_codec<RawCSR>(off, nbr, ids_random, ids_hot, buf));
    results.push_back(run_codec<rowcodec::RowEF>(off, nbr, ids_random, ids_hot, buf));
    results.push_back(run_codec<rowcodec::RowBIC>(off, nbr, ids_random, ids_hot, buf));
    results.push_back(run_codec<rowcodec::RowPack>(off, nbr, ids_random, ids_hot, buf));

    std::printf("\n### %s (N=%zu, %zu edges, mean degree %.2f)\n\n", prefix.c_str(), N, nbr.size(),
                static_cast<double>(nbr.size()) / static_cast<double>(N));
    std::printf("| codec | bytes | bits/edge | ratio vs raw | build s | round-trip | ns/row random | ns/row hot |\n");
    std::printf("|---|---:|---:|---:|---:|:---:|---:|---:|\n");
    const double raw_bytes = static_cast<double>(results.front().bytes);
    for (const auto& r : results) {
        std::printf("| %s | %zu | %.3f | %.3f | %.2f | %s | %.1f | %.1f |\n", r.name, r.bytes, r.bits_per_edge,
                    static_cast<double>(r.bytes) / raw_bytes, r.build_s,
                    r.ok ? "OK" : ("FAIL@" + std::to_string(r.first_fail)).c_str(), r.ns_random, r.ns_hot);
    }
    bool all_ok = true;
    for (const auto& r : results) all_ok &= r.ok;
    return all_ok ? 0 : 1;
}
