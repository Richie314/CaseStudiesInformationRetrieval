#include "gef/gef.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_raw_binary(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Failed to open " + path);
    }
    const std::streamsize bytes = ifs.tellg();
    if (bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error(path + " size is not a multiple of sizeof(T)");
    }
    const size_t count = static_cast<size_t>(bytes) / sizeof(T);
    std::vector<T> data(count);
    ifs.seekg(0);
    ifs.read(reinterpret_cast<char*>(data.data()), bytes);
    if (!ifs) {
        throw std::runtime_error("Short read on " + path);
    }
    return data;
}

void print_ratio(const char* label, size_t raw_bytes, size_t compressed_bytes) {
    const double ratio = compressed_bytes > 0
        ? static_cast<double>(raw_bytes) / static_cast<double>(compressed_bytes)
        : 0.0;
    std::printf("%-12s raw=%10zu B  compressed=%10zu B  ratio=%.2fx\n",
                label, raw_bytes, compressed_bytes, ratio);
}

int run(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <prefix> [--approximate]\n";
        return 1;
    }
    const std::string prefix = argv[1];
    const bool approximate = argc > 2 && std::string(argv[2]) == "--approximate";
    const gef::SplitPointStrategy strategy =
        approximate ? gef::APPROXIMATE_SPLIT_POINT : gef::OPTIMAL_SPLIT_POINT;

    // ---- Load raw CSR arrays produced by export_diskann_graph.py ----
    std::vector<uint64_t> offsets = read_raw_binary<uint64_t>(prefix + "_offsets.bin");
    std::vector<uint32_t> neighbors = read_raw_binary<uint32_t>(prefix + "_neighbors.bin");

    if (offsets.empty()) {
        std::cerr << "Error: " << prefix << "_offsets.bin holds no offsets\n";
        return 1;
    }
    if (offsets.back() != neighbors.size()) {
        std::cerr << "Error: offsets/neighbors mismatch: offsets.back()="
                  << offsets.back() << " but neighbors holds " << neighbors.size()
                  << " entries (stale or mismatched export?)\n";
        return 1;
    }
    const size_t N = offsets.size() - 1;
    const size_t nnz = neighbors.size();
    std::cout << "Loaded graph: N=" << N << " nodes, nnz=" << nnz << " edges\n";
    std::cout << "Strategy: " << (approximate ? "APPROXIMATE" : "OPTIMAL") << " split point\n\n";

    // ---- Compress with U_GEF ----
    gef::U_GEF<uint64_t> ef_offsets(offsets, strategy);
    gef::U_GEF<uint32_t> ef_neighbors(neighbors, strategy);

    const size_t raw_offsets_bytes = offsets.size() * sizeof(uint64_t);
    const size_t raw_neighbors_bytes = neighbors.size() * sizeof(uint32_t);

    print_ratio("offsets", raw_offsets_bytes, ef_offsets.size_in_bytes());
    print_ratio("neighbors", raw_neighbors_bytes, ef_neighbors.size_in_bytes());
    print_ratio("TOTAL",
                raw_offsets_bytes + raw_neighbors_bytes,
                ef_offsets.size_in_bytes() + ef_neighbors.size_in_bytes());

    // ---- Correctness check: reconstruct every row and compare ----
    std::cout << "\nVerifying full round-trip correctness...\n";
    bool ok = true;
    for (size_t i = 0; i < N && ok; ++i) {
        // Decompress offsets[i] and offsets[i+1] (O(1) random access each)
        const uint64_t start = ef_offsets[i];
        const uint64_t end = ef_offsets[i + 1];
        if (start != offsets[i] || end > neighbors.size() || end < start) {
            std::cerr << "OFFSET MISMATCH at node " << i << "\n";
            ok = false;
            break;
        }
        const size_t degree = end - start;
        if (degree == 0) continue;

        std::vector<uint32_t> decompressed(degree);
        const size_t written = ef_neighbors.get_elements(start, degree, decompressed);
        if (written != degree) {
            std::cerr << "SHORT READ at node " << i << ": requested " << degree
                      << " elements, got " << written << "\n";
            ok = false;
            break;
        }

        for (size_t j = 0; j < degree; ++j) {
            if (decompressed[j] != neighbors[start + j]) {
                std::cerr << "MISMATCH at node " << i << ", neighbor slot " << j << "\n";
                ok = false;
                break;
            }
        }
    }
    std::cout << (ok ? "OK: all rows reconstruct exactly.\n" : "FAILED verification!\n");
    if (!ok) return 1;

    // ---- Persist compressed structures to disk ----
    ef_offsets.serialize(prefix + "_offsets.gef");
    ef_neighbors.serialize(prefix + "_neighbors.gef");
    std::cout << "\nWrote " << prefix << "_offsets.gef and " << prefix << "_neighbors.gef\n";

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