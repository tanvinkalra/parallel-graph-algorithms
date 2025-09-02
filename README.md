# Accelerating Graph Traversals: Parallel BFS and DFS Optimization on CUDA-enabled GPUs

This repository contains the final project for the *Parallel Systems* course, titled **Accelerating Graph Traversals: Parallel BFS and DFS Optimization on CUDA-enabled GPUs**. The project explores and implements optimized versions of Breadth-First Search (BFS) and Depth-First Search (DFS) algorithms using CUDA to accelerate graph traversal on GPU hardware.

## 📈 Project Goals
Analyze and compare sequential vs. parallel traversal methods.

Utilize different memory and load-balancing optimizations (e.g., bitmap frontiers, warp-centric design).

Study performance gains and bottlenecks in large-scale graph traversal.

## Repository Structure

- `breadth-first-search/`
  - `bfs.c` – Sequential BFS implementation
  - `bfs-level-sync-cuda.cu` – Queue-based level-synchronous CUDA BFS
  - `bfs-level-sync-bitmap-cuda.cu` – Bitmap-based level-synchronous CUDA BFS
  - `bfs-warp-centric.cu` – Warp-centric bitmap CUDA BFS
  - `Makefile` – Build rules for BFS implementations
  - `example_graphs/`
    - `roadNet-CA.mtx` – Sample graph (real-world road network)

- `depth-first-search/`
  - `dfs.c` – Sequential DFS implementation
  - `dfs-cuda.cu` – Topological order-based parallel CUDA DFS
  - `Makefile` – Build rules for DFS implementations
  - `dataset/`
    - `asia.mtx` – Sample graph (DAG)
    - `dag.py` – Script to generate synthetic DAGs in `.mtx` format

## ⚙️ Requirements
 - NVIDIA CUDA Toolkit (version 10.0 or higher)
 - A CUDA-capable GPU
 - C/C++ compiler supporting C99 or newer
 - Python 3 (for DAG generation)


## 📌 BFS Implementations


### Compilation:
Run the following inside the `breadth-first-search/` directory:
```bash
make 
```
### Example Usage
After compilation, to run the warp-centric BFS implementation:
```
./bfs-warp-centric ./example_graphs/roadNet-CA.mtx
```

Other executables (e.g., bfs-level-sync-cuda, bfs-level-sync-bitmap-cuda) can also be run similarly by replacing the binary name.

## 📌 DFS Implementations


### Compilation:
Run the following inside the `depth-first-search/` directory:
```bash
make 
```
### Example Usage
After compilation, to run the CUDA-accelerated DFS:

```
./dfs ./dataset/asia.mtx
```

Sample graphs for testing are provided in the dataset/ folder.