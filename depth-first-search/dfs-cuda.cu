#include <stdio.h>
#include <limits.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"
#include <cuda_runtime.h>


#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })
 
void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

void convert_to_csr(graph *g, csr_graph *csr) {
    csr->num_vertices = g->num_vertices;
    csr->row_offsets = (int *)malloc((g->num_vertices + 1) * sizeof(int));
    
    int edge_count = 0;
    for(int i = 0; i < g->num_vertices; ++i) {
        csr->row_offsets[i] = edge_count;
        node *temp = g->adj_lists[i];
        while(temp) {
            edge_count++;
            temp = temp->next;
        }
    }
    csr->row_offsets[g->num_vertices] = edge_count;
    
    csr->col_indices = (int *)malloc(edge_count * sizeof(int));
    csr->num_edges = edge_count;
    
    int idx = 0;
    for(int i = 0; i < g->num_vertices; i++) {
        node *temp = g->adj_lists[i];
        while(temp) {
            csr->col_indices[idx++] = temp->vertex;
            temp = temp->next;
        }
    }
}

__global__ void topological_step(
    const int *row_offsets, const int *col_indices,
    int *in_degree, int *frontier_curr, int frontier_size,
    int *frontier_next, int *next_frontier_size, 
    int *topo_order, int *topo_counter
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = frontier_curr[tid];
    int pos = atomicAdd(topo_counter, 1);
    topo_order[pos] = u;

    for (int eid = row_offsets[u]; eid < row_offsets[u + 1]; eid++) {
        int v = col_indices[eid];
        int old_val = atomicSub(&in_degree[v], 1);
        if (old_val == 1) {
            int next_pos = atomicAdd(next_frontier_size, 1);
            frontier_next[next_pos] = v;
        }
    }
}

// MIN_ITER, MAX_ITER, TIME_LIMIT, 
double benchmark_dfs_cuda(csr_graph *csr, int source, int* distances) {
    int num_vertices = csr->num_vertices;

    // Host allocations
    int *h_in_degree = (int *)calloc(num_vertices, sizeof(int));
    int *h_topo_order = (int *)malloc(num_vertices * sizeof(int));
    int *h_frontier = (int *)malloc(num_vertices * sizeof(int));

    // Device allocations
    int *d_row_offsets, *d_col_indices;
    int *d_in_degree, *d_frontier_curr, *d_frontier_next, *d_frontier_size;
    int *d_topo_order, *d_topo_counter;

    cudaMalloc(&d_row_offsets, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_col_indices, csr->num_edges * sizeof(int));
    cudaMalloc(&d_in_degree, num_vertices * sizeof(int));
    cudaMalloc(&d_frontier_curr, num_vertices * sizeof(int));
    cudaMalloc(&d_frontier_next, num_vertices * sizeof(int));
    cudaMalloc(&d_frontier_size, sizeof(int));
    cudaMalloc(&d_topo_order, num_vertices * sizeof(int));
    cudaMalloc(&d_topo_counter, sizeof(int));

    cudaMemcpy(d_row_offsets, csr->row_offsets, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, csr->col_indices, csr->num_edges * sizeof(int), cudaMemcpyHostToDevice);

    int THREADS_PER_BLOCK = 256;

    // warmup
    for (int u = 0; u < num_vertices; u++) {
        for (int eid = csr->row_offsets[u]; eid < csr->row_offsets[u + 1]; eid++) {
            int v = csr->col_indices[eid];
            h_in_degree[v]++;
        }
    }

    // Prepare initial frontier
    int frontier_curr_size = 0;
    for (int i = 0; i < num_vertices; i++) {
        if (h_in_degree[i] == 0)
            h_frontier[frontier_curr_size++] = i;
    }

    cudaMemcpy(d_in_degree, h_in_degree, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier_curr, h_frontier, frontier_curr_size * sizeof(int), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_topo_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);

    timer time_one_iteration;
    timer_start(&time_one_iteration);

    int curr_size = frontier_curr_size;
    while (curr_size > 0) {
        int NUM_BLOCKS = (curr_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaMemcpy(d_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice);

        topological_step<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_row_offsets, d_col_indices,
            d_in_degree,
            d_frontier_curr, curr_size,
            d_frontier_next, d_frontier_size,
            d_topo_order, d_topo_counter
        );
        cudaDeviceSynchronize();

        // Swap
        int *temp = d_frontier_curr;
        d_frontier_curr = d_frontier_next;
        d_frontier_next = temp;

        cudaMemcpy(&curr_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(h_topo_order, d_topo_order, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_vertices; i++) distances[i] = INT_MAX;
    distances[source] = 0;

    for (int idx = 0; idx < num_vertices; idx++) {
        int u = h_topo_order[idx];
        if (distances[u] == INT_MAX) continue;
        for (int eid = csr->row_offsets[u]; eid < csr->row_offsets[u + 1]; eid++) {
            int v = csr->col_indices[eid];
            if (distances[v] == INT_MAX) {
                distances[v] = distances[u] + 1;
            }
        }
    }

    double estimated_time = seconds_elapsed(&time_one_iteration);
    //    printf("estimated time for once %f\n", (float) estimated_time);

// determine # of seconds dynamically
    int num_iterations;
    num_iterations = MAX_ITER;

    if (estimated_time == 0)
        num_iterations = MAX_ITER;
    else {
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int) (TIME_LIMIT / estimated_time)) ); 
    }
    printf("\tPerforming %d iterations\n", num_iterations);

    num_iterations = 10;

    // time several DFS iterations
    timer t;
    timer_start(&t);
    for (int j = 0; j < num_iterations; j++) {
        // Reset host in-degree + frontier
        memset(h_in_degree, 0, num_vertices * sizeof(int));
        for (int u = 0; u < num_vertices; u++) {
            for (int eid = csr->row_offsets[u]; eid < csr->row_offsets[u + 1]; eid++) {
                int v = csr->col_indices[eid];
                h_in_degree[v]++;
            }
        }

        frontier_curr_size = 0;
        for (int i = 0; i < num_vertices; i++) {
            if (h_in_degree[i] == 0)
                h_frontier[frontier_curr_size++] = i;
        }

        cudaMemcpy(d_in_degree, h_in_degree, num_vertices * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier_curr, h_frontier, frontier_curr_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_topo_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);

        int curr_size = frontier_curr_size;
        while (curr_size > 0) {
            cudaMemcpy(d_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice);
            int NUM_BLOCKS = (curr_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            topological_step<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
                d_row_offsets, d_col_indices,
                d_in_degree,
                d_frontier_curr, curr_size,
                d_frontier_next, d_frontier_size,
                d_topo_order, d_topo_counter
            );
            cudaDeviceSynchronize();

            int *temp = d_frontier_curr;
            d_frontier_curr = d_frontier_next;
            d_frontier_next = temp;

            cudaMemcpy(&curr_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
        }
    }

        double msec_per_iteration = milliseconds_elapsed(&t) / (double)num_iterations;
        printf("\tbenchmarking GPU-TopoSort-DFS: %8.4f ms \n", msec_per_iteration);

        cudaMemcpy(h_topo_order, d_topo_order, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < num_vertices; i++) distances[i] = INT_MAX;
        distances[source] = 0;

        for (int idx = 0; idx < num_vertices; idx++) {
            int u = h_topo_order[idx];
            if (distances[u] == INT_MAX) continue;
            for (int eid = csr->row_offsets[u]; eid < csr->row_offsets[u + 1]; eid++) {
                int v = csr->col_indices[eid];
                if (distances[v] == INT_MAX) {
                    distances[v] = distances[u] + 1;
                }
            }
        }

    // Cleanup
    free(h_in_degree);
    free(h_topo_order);
    free(h_frontier);

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_in_degree);
    cudaFree(d_frontier_curr);
    cudaFree(d_frontier_next);
    cudaFree(d_frontier_size);
    cudaFree(d_topo_order);
    cudaFree(d_topo_counter);

    return msec_per_iteration;
}

int main(int argc, char** argv) {
    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        return 0;
    }

    char * mm_filename = NULL;
    if (argc == 1) {
        printf("Give a MatrixMarket file.\n");
        return -1;
    } else 
        mm_filename = argv[1];

    graph g;
    printf("Filename: %s\n", mm_filename);
    read_graph_matrix(&g, mm_filename);

#ifdef TESTING
    //print in adjacency list format
        printf("Writing matrix in adjacency list format to test_adj_list ...");
        FILE *fp = fopen("test_adj_list", "w");
        fprintf(fp, "%d\n", g.num_vertices);
        for (int v = 0; v < g.num_vertices; v++) {
            node* temp = g.adj_lists[v];
            fprintf(fp, "Vertex %d:", v);
            while (temp) {
                fprintf(fp, " -> %d", temp->vertex);
                temp = temp->next;
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("... done!\n");
#endif 
    
    csr_graph csr;
    convert_to_csr(&g, &csr);
    delete_graph(&g);

    int* distances = (int *)malloc(csr.num_vertices * sizeof(int));
    int source = 0;
    benchmark_dfs_cuda(&csr, source, distances);

/* Test correctnesss */
#ifdef TESTING
    printf("Writing distance values from source: %d ...", source);
    fp = fopen("test_dist", "w");
    for (int i=0; i<csr.num_vertices; i++)
    {
      fprintf(fp, "Node %d : %d\n", i, distances[i]);
    }
    fclose(fp);
    printf("... done!\n");
#endif
    
    delete_csr_graph(&csr);
    return 0;
}