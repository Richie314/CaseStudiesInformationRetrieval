// rowbench: microbenchmark of whole-row (adjacency list) decoding for the
// stores compared in REPORT.md — raw CSR, U-GEF at several partition sizes
// (raw uint32 offsets), and the per-row codecs (RowEF, RowBIC, RowPack).
// Times 2M random-row fetches (uniform node ids, pre-generated) and prints a
// markdown table with bytes, bits/edge (incl. offsets/headers) and ns/row.

#include "gef/gef.hpp"
#include "row_codecs.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_raw(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Failed to open " + path);
    const std::streamsize bytes = ifs.tellg();
    std::vector<T> v(static_cast<size_t>(bytes) / sizeof(T));
    ifs.seekg(0);
    ifs.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

volatile uint64_t sink = 0;

template <class F>
double time_rows(const std::vector<uint32_t>& ids, F&& fetch, int reps = 3) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        uint64_t acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t u : ids) acc += fetch(u);
        auto t1 = std::chrono::steady_clock::now();
        sink += acc;
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / ids.size());
    }
    return best;
}

template <size_t P>
void bench_gef(const std::vector<uint64_t>& off, const std::vector<uint32_t>& nbr,
               const std::vector<uint32_t>& ids, std::vector<uint32_t>& buf, size_t nnz) {
    gef::U_GEF<uint32_t, P> enc(nbr, gef::OPTIMAL_SPLIT_POINT);
    std::vector<uint32_t> off32(off.begin(), off.end());
    const size_t bytes = enc.size_in_bytes() + off32.size() * 4;
    double ns = time_rows(ids, [&](uint32_t u) {
        const uint32_t s = off32[u], e = off32[u + 1];
        const size_t w = enc.get_elements(s, e - s, buf);
        uint64_t a = 0;
        for (size_t j = 0; j < w; ++j) a += buf[j];
        return a;
    });
    std::printf("| U-GEF P=%zu + raw32 offsets | %zu | %.2f | %.0f |\n", P, bytes, bytes * 8.0 / nnz, ns);
}

template <class C>
void bench_codec(const std::vector<uint64_t>& off, const std::vector<uint32_t>& nbr,
                 const std::vector<uint32_t>& ids, std::vector<uint32_t>& buf, size_t nnz) {
    C c(off, nbr);
    double ns = time_rows(ids, [&](uint32_t u) {
        const rowcodec::Row r = c.row(u, buf);
        uint64_t a = 0;
        for (uint32_t j = 0; j < r.len; ++j) a += r.ptr[j];
        return a;
    });
    std::printf("| %s (5 B/node header) | %zu | %.2f | %.0f |\n", C::name(), c.bytes(), c.bytes() * 8.0 / nnz, ns);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: rowbench <csr prefix> [rows=2000000]\n"); return 1; }
    const std::string prefix = argv[1];
    const size_t nrows = argc > 2 ? std::stoul(argv[2]) : 2000000;
    auto off = read_raw<uint64_t>(prefix + "_offsets.bin");
    auto nbr = read_raw<uint32_t>(prefix + "_neighbors.bin");
    const size_t N = off.size() - 1, nnz = nbr.size();
    uint32_t maxdeg = 0;
    for (size_t i = 0; i < N; ++i) maxdeg = std::max<uint32_t>(maxdeg, off[i + 1] - off[i]);
    std::vector<uint32_t> buf(maxdeg + 8);
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> U(0, static_cast<uint32_t>(N - 1));
    std::vector<uint32_t> ids(nrows);
    for (auto& x : ids) x = U(rng);

    std::printf("graph %s: N=%zu edges=%zu, %zu random rows\n\n", prefix.c_str(), N, nnz, nrows);
    std::printf("| store | bytes | bits/edge | ns/row |\n|---|---:|---:|---:|\n");
    {
        const size_t bytes = off.size() * 8 + nbr.size() * 4;
        double ns = time_rows(ids, [&](uint32_t u) {
            uint64_t a = 0;
            for (uint64_t j = off[u]; j < off[u + 1]; ++j) a += nbr[j];
            return a;
        });
        std::printf("| raw CSR (uint64 offsets) | %zu | %.2f | %.0f |\n", bytes, bytes * 8.0 / nnz, ns);
    }
    bench_gef<1024>(off, nbr, ids, buf, nnz);
    bench_gef<4096>(off, nbr, ids, buf, nnz);
    bench_gef<32000>(off, nbr, ids, buf, nnz);
    bench_gef<128000>(off, nbr, ids, buf, nnz);
    bench_codec<rowcodec::RowEF>(off, nbr, ids, buf, nnz);
    bench_codec<rowcodec::RowBIC>(off, nbr, ids, buf, nnz);
    bench_codec<rowcodec::RowPack>(off, nbr, ids, buf, nnz);
    return 0;
}
