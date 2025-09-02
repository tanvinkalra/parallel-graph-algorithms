#include <stdio.h>
#include <limits.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"


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

// MIN_ITER, MAX_ITER, TIME_LIMIT, 
double benchmark_bfs(graph* g, int source,  int* distances)
{
    // warmup    
    timer time_one_iteration;
    timer_start(&time_one_iteration);
    for (int i = 0; i < g->num_vertices; ++i)
        distances[i] = INT_MAX;

    distances[source] = 0;

    int *queue = (int*) malloc(g->num_vertices * sizeof(int));
    int front = 0, rear = 0;

    /* enqueue the source vertex */
    queue[rear++] = source;
    while (front < rear) {
        int u = queue[front++];
        node* temp = g->adj_lists[u];
        
        while (temp) {
            int v = temp->vertex;
            if (distances[v] == INT_MAX) 
            {
                distances[v] = distances[u] + 1;
                queue[rear++] = v;
            }
            temp = temp->next;
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

    // time several SpMV iterations
    timer t;
    timer_start(&t);
    for(int j = 0; j < num_iterations; j++) {
        for (int i = 0; i < g->num_vertices; ++i)
            distances[i] = INT_MAX;

        distances[source] = 0;
        
        front = 0, rear = 0;

        /* enqueue the source vertex */
        queue[rear++] = source;
        while (front < rear) {
            int u = queue[front++];
            node* temp = g->adj_lists[u];
            
            while (temp) {
                int v = temp->vertex;
                if (distances[v] == INT_MAX) 
                {
                    distances[v] = distances[u] + 1;
                    queue[rear++] = v;
                }
                temp = temp->next;
            }
        }
    }
    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    printf("\tbenchmarking Sequential-Breadth-First-Searcj: %8.4f ms \n", msec_per_iteration); 

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

    int* distances = malloc(g.num_vertices * sizeof(int));
    int source = 0;
    benchmark_bfs(&g, source, distances);

/* Test correctnesss */
#ifdef TESTING
    printf("Writing distance values from source: %d ...", source);
    fp = fopen("test_dist", "w");
    for (int i=0; i<g.num_vertices; i++)
    {
      fprintf(fp, "Node %d : %d\n", i, distances[i]);
    }
    fclose(fp);
    printf("... done!\n");
#endif

    delete_graph(&g);
    
    return 0;
}
