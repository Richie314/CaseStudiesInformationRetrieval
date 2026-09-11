// row_codecs.hpp — per-row codecs for the adjacency lists of a CSR graph.
//
// Three self-contained, header-only encoders that compress every row of a
// CSR (`off`: N+1 uint64 row pointers, `nbr`: uint32 neighbour ids, each row
// sorted ascending, equal consecutive ids allowed) independently and decode a
// whole row from a per-node header:
//
//   RowEF    per-row Elias-Fano with universe N: n*l low bits (l = floor(log2(N/n)))
//            followed by the unary upper-bits bitmap, decoded by a sequential
//            ctz scan (no rank/select directories)
//   RowBIC   per-row binary interpolative coding (Moffat & Stuiver 2000):
//            recursive midpoint, "lo+k / hi-(n-1-k)" range shrink, centered
//            minimal-binary code of the middle value in its range
//   RowPack  first id absolute in ceil(log2 N) bits, then gaps bit-packed at the
//            per-row width w = bits(max gap) (w stored in 6 bits after the first id)
//
// All three share the same layout: one global little-endian bit buffer of
// uint64 words plus a 5-byte per-node header {uint32 bit offset, uint8 degree}.
// bytes() counts the bit buffer (including its 16 bytes of read padding), the
// header array and any constant decode tables (RowEF: 256-byte l-per-degree
// table; RowBIC: ~164 KB recursion schedule shared by all rows) — i.e.
// everything the object keeps to answer row(u).
//
// Limits (checked at build, std::runtime_error otherwise): degree <= 255
// (8-bit degree), total payload < 2^32 bits (32-bit offsets), ids < N and
// each row non-decreasing. Reads of the bit buffer are unaligned 64-bit
// memcpy loads, so any single field must be <= 57 bits wide; every field
// used here is <= 33 bits (ids are 32-bit).
//
// C++20, no dependencies beyond the standard library, GCC 12+.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rowcodec {

struct Row {
    const uint32_t* ptr;
    uint32_t len;
};

namespace detail {

inline constexpr uint64_t mask64(unsigned n) {
    return n >= 64 ? ~0ull : ((1ull << n) - 1);
}

// Append-only writer into a growable vector<uint64_t>; bits are LSB-first
// inside each word, i.e. bit position p lives in words[p / 64] bit (p % 64).
class BitWriter {
public:
    void reserve_bits(uint64_t nbits) { words_.reserve(nbits / 64 + 3); }

    uint64_t bits() const { return pos_; }

    // Append the low n bits of v (n <= 64).
    void write(uint64_t v, unsigned n) {
        if (n == 0) return;
        ensure(n);
        v &= mask64(n);
        const size_t wi = static_cast<size_t>(pos_ >> 6);
        const unsigned s = static_cast<unsigned>(pos_ & 63);
        words_[wi] |= v << s;
        if (s + n > 64) words_[wi + 1] |= v >> (64 - s);
        pos_ += n;
    }

    // Append n zero bits (any n): the buffer is zero-filled, so just advance.
    void write_zeros(uint64_t n) {
        ensure(n);
        pos_ += n;
    }

    // Return the words with two zero words of padding so that an unaligned
    // 8-byte load anywhere inside the written region stays in bounds.
    std::vector<uint64_t> finish() {
        words_.resize(static_cast<size_t>((pos_ + 63) / 64) + 2);
        words_.shrink_to_fit();
        return std::move(words_);
    }

private:
    void ensure(uint64_t nbits) {
        const size_t need = static_cast<size_t>((pos_ + nbits) >> 6) + 2;
        if (need > words_.size()) words_.resize(std::max(need, words_.size() + words_.size() / 2));
    }

    std::vector<uint64_t> words_;
    uint64_t pos_ = 0;
};

// Sequential reader over a padded word buffer. peek/read take n <= 57 bits
// (one unaligned 8-byte load at the byte holding the current bit).
class BitReader {
public:
    BitReader(const uint64_t* words, uint64_t bitpos)
        : base_(reinterpret_cast<const uint8_t*>(words)), pos_(bitpos) {}

    uint64_t peek(unsigned n) const {
        uint64_t w;
        std::memcpy(&w, base_ + (pos_ >> 3), 8);
        return (w >> (pos_ & 7)) & mask64(n);
    }
    void skip(unsigned n) { pos_ += n; }
    uint64_t read(unsigned n) {
        const uint64_t v = peek(n);
        pos_ += n;
        return v;
    }
    uint64_t pos() const { return pos_; }

private:
    const uint8_t* base_;
    uint64_t pos_;
};

// 5 bytes per node: uint32 bit offset + uint8 degree, read with one 8-byte load.
class NodeHeader {
public:
    explicit NodeHeader(size_t n) : h_(5 * n + 8, 0) {}

    void set(size_t u, uint64_t bit_offset, unsigned degree, const char* who) {
        if (bit_offset > 0xFFFFFFFFull)
            throw std::runtime_error(std::string(who) + ": payload exceeds 2^32 bits (32-bit offsets)");
        if (degree > 255)
            throw std::runtime_error(std::string(who) + ": degree " + std::to_string(degree) + " > 255 (8-bit degree)");
        const uint32_t o = static_cast<uint32_t>(bit_offset);
        std::memcpy(&h_[5 * u], &o, 4);
        h_[5 * u + 4] = static_cast<uint8_t>(degree);
    }

    void get(size_t u, uint32_t& bit_offset, unsigned& degree) const {
        uint64_t v;
        std::memcpy(&v, &h_[5 * u], 8);
        bit_offset = static_cast<uint32_t>(v);
        degree = static_cast<unsigned>((v >> 32) & 0xFF);
    }

    size_t bytes() const { return h_.size(); }

private:
    std::vector<uint8_t> h_;
};

inline void check_row(const uint32_t* x, unsigned n, uint64_t N, const char* who) {
    for (unsigned i = 0; i < n; ++i) {
        if (x[i] >= N) throw std::runtime_error(std::string(who) + ": id out of range");
        if (i && x[i] < x[i - 1]) throw std::runtime_error(std::string(who) + ": row not sorted");
    }
}

// ---- centered minimal-binary (truncated binary) code of v in [0, r) ----------
// b = ceil(log2 r); the 2^b - r "short" (b-1 bit) codewords are assigned to
// the middle of the range, as in Moffat & Stuiver. r == 1 is handled without
// a branch: with b forced to 1 the single value gets a 0-bit "short" code.
inline unsigned interp_bits(uint64_t r) {
    const unsigned b = static_cast<unsigned>(std::bit_width(r - 1));  // ceil(log2 r) for r >= 2
    return b ? b : 1u;
}

inline void write_interp(BitWriter& bw, uint64_t v, uint64_t r) {
    const unsigned b = interp_bits(r);
    const uint64_t half = 1ull << (b - 1);
    const uint64_t nshort = (2 * half) - r;   // 2^b - r
    const uint64_t c = r - half;              // first value with a short code
    const uint64_t vp = v >= c ? v - c : v - c + r;  // (v - c) mod r
    if (vp < nshort) {
        bw.write(vp, b - 1);
    } else {
        const uint64_t t = vp - nshort;
        const uint64_t x = nshort + (t >> 1);
        bw.write(x | ((t & 1) << (b - 1)), b);
    }
}

inline uint64_t read_interp(BitReader& br, uint64_t r) {
    const unsigned b = interp_bits(r);
    const uint64_t half = 1ull << (b - 1);
    const uint64_t nshort = (2 * half) - r;
    const uint64_t c = r - half;
    const uint64_t chunk = br.peek(b);
    const uint64_t x = chunk & (half - 1);
    const uint64_t islong = static_cast<uint64_t>(x >= nshort);
    const uint64_t vlong = ((x << 1) | (chunk >> (b - 1))) - nshort;
    const uint64_t vp = x + ((vlong - x) & (0 - islong));   // islong ? vlong : x
    br.skip(b - 1 + static_cast<unsigned>(islong));
    uint64_t v = vp + c;
    v -= r & (0 - static_cast<uint64_t>(v >= r));            // if (v >= r) v -= r
    return v;
}

}  // namespace detail

// =============================================================================
// RowEF — per-row Elias-Fano, universe N
// =============================================================================
class RowEF {
public:
    // `universe` = id range [0, universe); defaults to the number of rows
    // (ids of a full graph), but a sub-CSR (e.g. the cold rows of a two-tier
    // store) must pass the id universe of the whole graph.
    RowEF(const std::vector<uint64_t>& off, const std::vector<uint32_t>& nbr, size_t universe = 0)
        : N_(universe ? universe : off.size() - 1), hdr_(off.size() - 1) {
        for (unsigned d = 0; d < 256; ++d) {
            const uint64_t q = d ? N_ / d : 0;
            ell_[d] = q ? static_cast<uint8_t>(std::bit_width(q) - 1) : 0;  // floor(log2(N/d))
        }
        detail::BitWriter bw;
        bw.reserve_bits(nbr.size() * 20 + 64);
        for (size_t u = 0, rows = off.size() - 1; u < rows; ++u) {
            const uint64_t n64 = off[u + 1] - off[u];
            if (n64 > 255) throw std::runtime_error("RowEF: degree > 255");
            const unsigned n = static_cast<unsigned>(n64);
            hdr_.set(u, bw.bits(), n, "RowEF");
            const uint32_t* x = nbr.data() + off[u];
            detail::check_row(x, n, N_, "RowEF");
            const unsigned ell = ell_[n];
            for (unsigned i = 0; i < n; ++i) bw.write(x[i] & detail::mask64(ell), ell);
            uint64_t prev = 0;
            for (unsigned i = 0; i < n; ++i) {
                const uint64_t high = static_cast<uint64_t>(x[i]) >> ell;
                bw.write_zeros(high - prev);
                bw.write(1, 1);
                prev = high;
            }
        }
        words_ = bw.finish();
    }

    Row row(uint32_t u, std::vector<uint32_t>& buf) const {
        uint32_t off;
        unsigned n;
        hdr_.get(u, off, n);
        uint32_t* out = buf.data();
        if (n == 0) return {out, 0};
        const unsigned ell = ell_[n];
        const uint64_t* words = words_.data();
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(words);
        uint64_t lpos = off;
        const uint64_t upos = off + static_cast<uint64_t>(n) * ell;
        size_t wi = static_cast<size_t>(upos >> 6);
        const unsigned s = static_cast<unsigned>(upos & 63);
        uint64_t w = words[wi] & (~0ull << s);
        int64_t base = -static_cast<int64_t>(s);  // bitmap position of bit 0 of words[wi]
        const uint64_t lmask = detail::mask64(ell);
        for (unsigned i = 0; i < n; ++i) {
            while (w == 0) {
                w = words[++wi];
                base += 64;
            }
            const unsigned t = static_cast<unsigned>(std::countr_zero(w));
            w &= w - 1;
            const uint64_t high = static_cast<uint64_t>(base + t) - i;  // i-th one at high_i + i
            uint64_t lw;
            std::memcpy(&lw, bytes + (lpos >> 3), 8);
            const uint64_t low = (lw >> (lpos & 7)) & lmask;
            lpos += ell;
            out[i] = static_cast<uint32_t>((high << ell) | low);
        }
        return {out, n};
    }

    size_t bytes() const { return words_.size() * sizeof(uint64_t) + hdr_.bytes() + sizeof(ell_); }
    static constexpr const char* name() { return "RowEF"; }

private:
    uint64_t N_;
    detail::NodeHeader hdr_;
    std::array<uint8_t, 256> ell_{};
    std::vector<uint64_t> words_;
};

// =============================================================================
// RowBIC — per-row binary interpolative coding
// =============================================================================
// Rows may contain repeated ids, which the count-based range shrink cannot
// represent, so the row is coded as the strictly increasing y_i = x_i + i over
// [0, N + n - 2]; decoding subtracts i back. The cost is log2 C(N+n-1, n) -
// log2 C(N, n) bits per row, i.e. negligible for n << N.
//
// The recursion (middle element, then left half, then right half) depends
// only on the row length, so it is flattened once per degree into a schedule
// of n steps {slot m, left-bound slot lb, right-bound slot rb, k, n-1-k}.
// Slots are 1-based into a scratch array whose slot 0 holds "low - 1" and slot
// n+1 holds "high + 1" for the whole row, so every step is the same
// branch-free code: low = tmp[lb] + 1, high = tmp[rb] - 1, value in
// [low + k, high - (n-1-k)]. Encoder and decoder walk the same schedule.
class RowBIC {
public:
    // `universe` = id range [0, universe); defaults to the number of rows
    // (ids of a full graph), but a sub-CSR (e.g. the cold rows of a two-tier
    // store) must pass the id universe of the whole graph.
    RowBIC(const std::vector<uint64_t>& off, const std::vector<uint32_t>& nbr, size_t universe = 0)
        : N_(universe ? universe : off.size() - 1), hdr_(off.size() - 1) {
        if (N_ > 0xFFFFFF00ull) throw std::runtime_error("RowBIC: N too large for 32-bit ranges");
        build_schedule();
        detail::BitWriter bw;
        bw.reserve_bits(nbr.size() * 20 + 64);
        uint32_t tmp[kMaxDeg + 2];
        for (size_t u = 0, rows = off.size() - 1; u < rows; ++u) {
            const uint64_t n64 = off[u + 1] - off[u];
            if (n64 > kMaxDeg) throw std::runtime_error("RowBIC: degree > 255");
            const unsigned n = static_cast<unsigned>(n64);
            hdr_.set(u, bw.bits(), n, "RowBIC");
            if (n == 0) continue;
            const uint32_t* x = nbr.data() + off[u];
            detail::check_row(x, n, N_, "RowBIC");
            tmp[0] = ~0u;                                   // low - 1 = -1 (wraps to 0 on +1)
            for (unsigned i = 0; i < n; ++i) tmp[i + 1] = x[i] + i;
            tmp[n + 1] = static_cast<uint32_t>(N_ + n - 1);  // high + 1
            const Step* st = steps_.data() + step_off_[n];
            for (unsigned t = 0; t < n; ++t) {
                const Step s = st[t];
                const uint32_t low = tmp[s.lb] + 1, high = tmp[s.rb] - 1;
                const uint64_t lo = static_cast<uint64_t>(low) + s.k;
                const uint64_t hi = static_cast<uint64_t>(high) - s.rc;
                detail::write_interp(bw, tmp[s.m] - lo, hi - lo + 1);
            }
        }
        words_ = bw.finish();
    }

    Row row(uint32_t u, std::vector<uint32_t>& buf) const {
        uint32_t off;
        unsigned n;
        hdr_.get(u, off, n);
        uint32_t* out = buf.data();
        if (n == 0) return {out, 0};
        uint32_t tmp[kMaxDeg + 2];
        tmp[0] = ~0u;
        tmp[n + 1] = static_cast<uint32_t>(N_ + n - 1);
        detail::BitReader br(words_.data(), off);
        const Step* st = steps_.data() + step_off_[n];
        for (unsigned t = 0; t < n; ++t) {
            const Step s = st[t];
            const uint32_t low = tmp[s.lb] + 1, high = tmp[s.rb] - 1;
            const uint64_t lo = static_cast<uint64_t>(low) + s.k;
            const uint64_t hi = static_cast<uint64_t>(high) - s.rc;
            tmp[s.m] = static_cast<uint32_t>(lo + detail::read_interp(br, hi - lo + 1));
        }
        for (unsigned i = 0; i < n; ++i) out[i] = tmp[i + 1] - i;
        return {out, n};
    }

    size_t bytes() const {
        return words_.size() * sizeof(uint64_t) + hdr_.bytes() + steps_.size() * sizeof(Step) +
               step_off_.size() * sizeof(uint32_t);
    }
    static constexpr const char* name() { return "RowBIC"; }

private:
    static constexpr unsigned kMaxDeg = 255;
    struct Step {
        uint8_t m, lb, rb, k, rc;
    };

    static void gen(std::vector<Step>& out, unsigned base, unsigned n, unsigned lb, unsigned rb) {
        if (n == 0) return;
        const unsigned k = n / 2;
        const unsigned m = base + k + 1;  // 1-based slot
        out.push_back({static_cast<uint8_t>(m), static_cast<uint8_t>(lb), static_cast<uint8_t>(rb),
                       static_cast<uint8_t>(k), static_cast<uint8_t>(n - 1 - k)});
        gen(out, base, k, lb, m);
        gen(out, base + k + 1, n - 1 - k, m, rb);
    }

    void build_schedule() {
        step_off_.assign(kMaxDeg + 1, 0);
        steps_.clear();
        for (unsigned n = 1; n <= kMaxDeg; ++n) {
            step_off_[n] = static_cast<uint32_t>(steps_.size());
            gen(steps_, 0, n, 0, n + 1);
        }
    }

    uint64_t N_;
    detail::NodeHeader hdr_;
    std::vector<Step> steps_;        // all schedules back to back (~163 KB)
    std::vector<uint32_t> step_off_; // schedule start per degree
    std::vector<uint64_t> words_;
};

// =============================================================================
// RowPack — first id absolute, gaps bit-packed at a per-row width
// =============================================================================
class RowPack {
public:
    // `universe` = id range [0, universe); defaults to the number of rows
    // (ids of a full graph), but a sub-CSR (e.g. the cold rows of a two-tier
    // store) must pass the id universe of the whole graph.
    RowPack(const std::vector<uint64_t>& off, const std::vector<uint32_t>& nbr, size_t universe = 0)
        : N_(universe ? universe : off.size() - 1), hdr_(off.size() - 1) {
        first_bits_ = N_ > 1 ? static_cast<unsigned>(std::bit_width(N_ - 1)) : 0;  // ceil(log2 N)
        detail::BitWriter bw;
        bw.reserve_bits(nbr.size() * 20 + 64);
        for (size_t u = 0, rows = off.size() - 1; u < rows; ++u) {
            const uint64_t n64 = off[u + 1] - off[u];
            if (n64 > 255) throw std::runtime_error("RowPack: degree > 255");
            const unsigned n = static_cast<unsigned>(n64);
            hdr_.set(u, bw.bits(), n, "RowPack");
            if (n == 0) continue;
            const uint32_t* x = nbr.data() + off[u];
            detail::check_row(x, n, N_, "RowPack");
            bw.write(x[0], first_bits_);
            if (n >= 2) {
                uint32_t maxd = 0;
                for (unsigned i = 1; i < n; ++i) maxd = std::max(maxd, x[i] - x[i - 1]);
                const unsigned w = static_cast<unsigned>(std::bit_width(maxd));  // 0..32
                bw.write(w, kWidthBits);
                for (unsigned i = 1; i < n; ++i) bw.write(x[i] - x[i - 1], w);
            }
        }
        words_ = bw.finish();
    }

    Row row(uint32_t u, std::vector<uint32_t>& buf) const {
        uint32_t off;
        unsigned n;
        hdr_.get(u, off, n);
        uint32_t* out = buf.data();
        if (n == 0) return {out, 0};
        detail::BitReader br(words_.data(), off);
        uint32_t x = static_cast<uint32_t>(br.read(first_bits_));
        out[0] = x;
        if (n >= 2) {
            const unsigned w = static_cast<unsigned>(br.read(kWidthBits));
            for (unsigned i = 1; i < n; ++i) {
                x += static_cast<uint32_t>(br.read(w));
                out[i] = x;
            }
        }
        return {out, n};
    }

    size_t bytes() const { return words_.size() * sizeof(uint64_t) + hdr_.bytes(); }
    static constexpr const char* name() { return "RowPack"; }

private:
    static constexpr unsigned kWidthBits = 6;  // w in [0, 32]

    uint64_t N_;
    unsigned first_bits_;
    detail::NodeHeader hdr_;
    std::vector<uint64_t> words_;
};

}  // namespace rowcodec
