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

// CUDA kernel to initialize distances to INT_MAX
__global__ void set_int_max(int* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = INT_MAX;
}

// CUDA BFS kernel
__global__ void bfs_kernel_level_sync(
    const int *row_offsets,
    const int *col_indices,
    int *distances,
    int *current_frontier,
    int *next_frontier,
    int *next_frontier_size,
    int level,
    int curr_frontier_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < curr_frontier_size) {
        int u = current_frontier[idx];
        int start = row_offsets[u];
        int end = row_offsets[u+1];

        for(int i = start; i < end; i++) {
            int v = col_indices[i];
            if(atomicCAS(&distances[v], INT_MAX, level) == INT_MAX) {
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
            }
        }
    }
}

// The main benchmarking function
double benchmark_bfs_gpu_level_sync(csr_graph *csr, int source, int* distances) {
    // Device allocations
    int *d_row, *d_col, *d_dist, *d_curr_frontier, *d_next_frontier;
    int *d_curr_frontier_size, *d_next_frontier_size;
    cudaMalloc(&d_row, (csr->num_vertices+1)*sizeof(int));
    cudaMalloc(&d_col, csr->num_edges*sizeof(int));
    cudaMalloc(&d_dist, csr->num_vertices*sizeof(int));
    cudaMalloc(&d_curr_frontier, csr->num_vertices*sizeof(int));
    cudaMalloc(&d_next_frontier, csr->num_vertices*sizeof(int));
    cudaMalloc(&d_curr_frontier_size, sizeof(int));
    cudaMalloc(&d_next_frontier_size, sizeof(int));

    // Copy graph data
    cudaMemcpy(d_row, csr->row_offsets, (csr->num_vertices+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, csr->col_indices, csr->num_edges*sizeof(int), cudaMemcpyHostToDevice);

    // Warmup
    timer time_one_iteration;
    timer_start(&time_one_iteration);

    int h_curr_frontier_size = 1;
    int zero = 0;
    set_int_max<<<(csr->num_vertices + 255)/256, 256>>>(d_dist, csr->num_vertices);
    cudaDeviceSynchronize();
    cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_curr_frontier, &source, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_curr_frontier_size, &h_curr_frontier_size, sizeof(int), cudaMemcpyHostToDevice);

    int level = 1;
    while (h_curr_frontier_size > 0) {
        cudaMemcpy(d_next_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice);

        dim3 block(256);
        dim3 grid((h_curr_frontier_size + block.x - 1) / block.x);

        bfs_kernel_level_sync<<<grid, block>>>(
            d_row, d_col, d_dist,
            d_curr_frontier, d_next_frontier,
            d_next_frontier_size, level, h_curr_frontier_size
        );
        cudaDeviceSynchronize();

        // Get the size of the next frontier
        cudaMemcpy(&h_curr_frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap frontiers and sizes
        int *temp_frontier = d_curr_frontier;
        d_curr_frontier = d_next_frontier;
        d_next_frontier = temp_frontier;

        int *temp_size = d_curr_frontier_size;
        d_curr_frontier_size = d_next_frontier_size;
        d_next_frontier_size = temp_size;

        level++;
    }

    double estimated_time = seconds_elapsed(&time_one_iteration); 
    //    printf("estimated time for once %f\n", (float) estimated_time);

    // determine # of seconds dynamically
    int num_iterations;
    num_iterations = MAX_ITER;

    if (estimated_time == 0)
        num_iterations = MAX_ITER;
    else {
        num_iterations = (int)(TIME_LIMIT / estimated_time);
        if (num_iterations < MIN_ITER) num_iterations = MIN_ITER;
        if (num_iterations > MAX_ITER) num_iterations = MAX_ITER;
    }
    printf("\tPerforming %d iterations\n", num_iterations);

    // time several BFS iterations
    timer t;
    timer_start(&t);
    for(int j = 0; j < num_iterations; j++) {
        h_curr_frontier_size = 1;
        set_int_max<<<(csr->num_vertices + 255)/256, 256>>>(d_dist, csr->num_vertices);
        cudaDeviceSynchronize();
        cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy(d_curr_frontier, &source, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_curr_frontier_size, &h_curr_frontier_size, sizeof(int), cudaMemcpyHostToDevice);

        level = 1;
        while (h_curr_frontier_size > 0) {
            cudaMemcpy(d_next_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice);

            dim3 block(256);
            dim3 grid((h_curr_frontier_size + block.x - 1) / block.x);

            bfs_kernel_level_sync<<<grid, block>>>(
                d_row, d_col, d_dist,
                d_curr_frontier, d_next_frontier,
                d_next_frontier_size, level, h_curr_frontier_size
            );
            cudaDeviceSynchronize();

            cudaMemcpy(&h_curr_frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);

            // Swap frontiers and sizes
            int *temp_frontier = d_curr_frontier;
            d_curr_frontier = d_next_frontier;
            d_next_frontier = temp_frontier;

            int *temp_size = d_curr_frontier_size;
            d_curr_frontier_size = d_next_frontier_size;
            d_next_frontier_size = temp_size;

            level++;
        }
    }
    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    printf("\tbenchmarking GPU-Level-Sync-Breadth-First-Search: %8.4f ms \n", msec_per_iteration);

    // Copy the result back to host.
    cudaMemcpy(distances, d_dist, csr->num_vertices*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_dist);
    cudaFree(d_curr_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_curr_frontier_size);
    cudaFree(d_next_frontier_size);

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
    benchmark_bfs_gpu_level_sync(&csr, source, distances);

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
