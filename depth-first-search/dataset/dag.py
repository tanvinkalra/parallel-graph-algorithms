import networkx as nx
import scipy.sparse as sp
from scipy.io import mmwrite
import random  # ← use this instead

# Step 1: Generate a random DAG
def generate_random_dag(num_nodes, edge_prob):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # ensures no cycles
            if random.random() < edge_prob:
                G.add_edge(i, j)
    return G

# Step 2: Convert to sparse matrix (COO format)
def dag_to_sparse_matrix(G):
    return nx.to_scipy_sparse_array(G, format='coo')

# Step 3: Save as .mtx file
def save_dag_as_mtx(G, filename):
    A = dag_to_sparse_matrix(G)
    mmwrite(filename, A)

# Usage
dag = generate_random_dag(num_nodes=20000, edge_prob=0.4)
save_dag_as_mtx(dag, "europe.mtx")
