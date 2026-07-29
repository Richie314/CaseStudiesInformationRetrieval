# Case Studies of Information Retrieval

```bash
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
pip install -r .\scripts\requirements.txt
```

```bash
Invoke-WebRequest -OutFile ".\data\sift-128-euclidean.hdf5" -Uri "http://ann-benchmarks.com/sift-128-euclidean.hdf5"

python .\scripts\diskann-index.py --input .\data\sift-128-euclidean.hdf5 --index-dir .\data\sift_diskann_index 

python .\scripts\export_diskann_graph_for_gef.py --index .\data\sift_diskann_index\sift_diskann --prefix graph
```

```bash
cmake -S . -B build
```