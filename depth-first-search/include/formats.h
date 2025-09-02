
#pragma once

/* graph node reprsentation */
typedef struct node {
    int vertex;
    struct node* next;
} node;

typedef struct graph {
    int num_vertices;       // Number of vertices in the graph
    node** adj_lists;       // Array of linked lists for adjacency list representation
} graph;

typedef struct csr_graph{
    int num_vertices;
    int num_edges;
    int *row_offsets;
    int *col_indices;
} csr_graph;


void delete_graph(graph* g) {
    for (int i = 0; i < g->num_vertices; ++i) {
        node* temp = g->adj_lists[i];
        while (temp != NULL) {
            node* nextNode = temp->next;
            free(temp);
            temp = nextNode;
        }
    }
    free(g->adj_lists);
}

void delete_csr_graph(csr_graph* csr) {
    free(csr->row_offsets);  free(csr->col_indices); 
}

