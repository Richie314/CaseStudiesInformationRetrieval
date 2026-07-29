#!/usr/bin/env python3

import argparse
import json
import struct

import numpy as np

HEADER_FMT = "<QIIQ"  # index_size(u64), max_observed_degree(u32), start(u32), num_frozen_points(u64)
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 24 bytes


def parse_diskann_graph(path: str):
    """
    Parse a DiskANN Vamana graph file.

    Returns
    -------
    adjacency : list[np.ndarray[uint32]], one array of neighbor ids per node
    meta : dict with index_size, max_observed_degree, start, num_frozen_points
    """
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise ValueError(f"{path} is too short to contain a valid DiskANN graph header")
        index_size, max_observed_degree, start, num_frozen_points = struct.unpack(HEADER_FMT, header)

        adjacency = []
        bytes_read = HEADER_SIZE
        while bytes_read < index_size:
            k_bytes = f.read(4)
            if len(k_bytes) < 4:
                raise ValueError(f"Unexpected EOF while reading degree at byte {bytes_read}")
            (k,) = struct.unpack("<I", k_bytes)
            row_bytes = f.read(4 * k)
            if len(row_bytes) < 4 * k:
                raise ValueError(f"Unexpected EOF while reading neighbor list at byte {bytes_read}")
            row = np.frombuffer(row_bytes, dtype="<u4")
            adjacency.append(row)
            bytes_read += 4 * (k + 1)

        if bytes_read != index_size:
            raise ValueError(
                f"Byte accounting mismatch after parsing: read {bytes_read}, "
                f"header declared index_size={index_size}. File may be truncated or corrupt."
            )

    meta = {
        "index_size": index_size,
        "max_observed_degree": max_observed_degree,
        "start": start,
        "num_frozen_points": num_frozen_points,
        "num_points": len(adjacency),
    }
    return adjacency, meta


def adjacency_to_csr(adjacency, sort_rows: bool = True):
    N = len(adjacency)
    row_ptr = np.zeros(N + 1, dtype=np.uint64)
    rows = []
    total = 0
    for i, row in enumerate(adjacency):
        r = np.sort(row) if sort_rows else row
        rows.append(r.astype(np.uint32))
        total += len(r)
        row_ptr[i + 1] = total
    neighbors = np.concatenate(rows) if rows else np.empty(0, dtype=np.uint32)
    return row_ptr, neighbors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True,
                         help="Path to the DiskANN graph file (the index_prefix passed to "
                              "diskannpy.build_memory_index, with NO .data/.tags suffix)")
    parser.add_argument("--prefix", default="graph", help="Output file prefix")
    parser.add_argument("--no-sort", action="store_true",
                         help="Keep original neighbor order (use B_GEF/B_STAR_GEF downstream instead of U_GEF)")
    args = parser.parse_args()

    adjacency, meta = parse_diskann_graph(args.index)
    print(f"Parsed DiskANN graph: N={meta['num_points']}, start={meta['start']}, "
          f"max_observed_degree={meta['max_observed_degree']}, "
          f"num_frozen_points={meta['num_frozen_points']}")

    row_ptr, neighbors = adjacency_to_csr(adjacency, sort_rows=not args.no_sort)
    nnz = len(neighbors)
    avg_degree = nnz / meta["num_points"] if meta["num_points"] else 0
    print(f"CSR built: nnz={nnz}, avg out-degree={avg_degree:.2f}")

    row_ptr.tofile(f"{args.prefix}_offsets.bin")
    neighbors.tofile(f"{args.prefix}_neighbors.bin")
    with open(f"{args.prefix}_meta.json", "w") as f:
        json.dump({**meta, "sorted": not args.no_sort}, f, indent=2)

    print(f"Wrote {args.prefix}_offsets.bin ({row_ptr.nbytes} bytes raw)")
    print(f"Wrote {args.prefix}_neighbors.bin ({neighbors.nbytes} bytes raw)")
    print(f"Wrote {args.prefix}_meta.json")


if __name__ == "__main__":
    main()