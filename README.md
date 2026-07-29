# Case Studies of Information Retrieval

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r ./scripts/requirements.txt
```

```bash
curl -L "http://ann-benchmarks.com/sift-128-euclidean.hdf5" -o "./data/sift-128-euclidean.hdf5"

python3 ./scripts/diskann-index.py --input ./data/sift-128-euclidean.hdf5 --index-dir ./data/sift_diskann_index

python3 ./scripts/export_diskann_graph.py --index ./data/sift_diskann_index/sift_diskann --prefix ./data/graph
```

```bash
cmake -S . -B build
cmake --build build --target information_retrieval -- -j$(nproc)
```

```bash
./build/information_retrieval ./data/graph
```