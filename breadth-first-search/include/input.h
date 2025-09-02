#pragma once

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "formats.h"
#include "mmio.h"
#include "../config.h"

node* create_node(int v) {
    node* new_node = (node*)malloc(sizeof(node));
    new_node->vertex = v;
    new_node->next = NULL;
    return new_node;
}

void add_edge(graph* g, int source, int destination) {
    // Add edge from source to destination
    node* new_node = create_node(destination);
    new_node->next = g->adj_lists[source];
    g->adj_lists[source] = new_node;
}

void read_graph_matrix(graph *g, const char * mm_filename)
{
    FILE * fid;
    MM_typecode matcode;
    
    fid = fopen(mm_filename, "r");

    if (fid == NULL){
        printf("Unable to open file %s\n", mm_filename);
        exit(1);
    }

    if (mm_read_banner(fid, &matcode) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (!mm_is_valid(matcode)){
        printf("Invalid Matrix Market file.\n");
        exit(1);
    }

    if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) && mm_is_sparse(matcode) ) ){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        printf("Only sparse real-valued or pattern coordinate matrices are supported\n");
        exit(1);
    }

    int num_rows, num_cols, num_nonzeros;
    if ( mm_read_mtx_crd_size(fid,&num_rows,&num_cols,&num_nonzeros) !=0)
            exit(1);

    if (num_rows != num_cols) {
        printf("Matrix is not square (rows=%d, cols=%d)\n", num_rows, num_cols);
        fclose(fid);
        exit(1);
    }

    g->num_vertices     = (int) num_rows;

    g->adj_lists = (node**)malloc(g->num_vertices * sizeof(node*));

    // Initialize each adjacency list as NULL
    for (int i = 0; i < g->num_vertices; i++) {
        g->adj_lists[i] = NULL;
    }

    printf("Reading sparse matrix from file (%s):",mm_filename);
    fflush(stdout);

    // Handle symmetric matrices by mirroring edges
    int is_symmetric = mm_is_symmetric(matcode);

    for (int i = 0; i < num_nonzeros; i++) {
        int source, destination;
        double value = 1.0;

        if (mm_is_pattern(matcode)) {
            assert(fscanf(fid, " %d %d \n", &source, &destination) == 2);
        } else if (mm_is_real(matcode) || mm_is_integer(matcode)) {
            assert(fscanf(fid, " %d %d %lf \n", &source, &destination, &value) == 3);
        } else {
            printf("Unrecognized data type\n");
            exit(1);
        }

        /* Add edge(s) */
        if (is_symmetric) {
            if ((int) source != (int) destination) {
                add_edge(g, (int) source - 1, (int) destination - 1);
                add_edge(g, (int) destination - 1, (int) source - 1);
            } else {
                add_edge(g, (int) source - 1, (int) destination - 1);
            }
        } else {
            add_edge(g, (int) source - 1, (int) destination - 1);
        }
    }

    fclose(fid);
    printf(" done\n");
}