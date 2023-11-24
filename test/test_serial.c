#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <iostream>

#include "../include/parser.h"
#include "../include/sequential.h"

#define NUM_REPEAT 100
#define PRINT_MODE 2

int main(int argc, char **argv) {
    fprintf(stdout, "============================\n");
    fprintf(stdout, "****** Sequential SpMV *****\n");
    fprintf(stdout, "----------------------------\n");
    fprintf(stdout, "*********** TEST ***********\n");
    fprintf(stdout, "----------------------------\n");
    fprintf(stdout, "============================\n");
    
    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values, elapsed_time;
    
    int num_repeat = NUM_REPEAT;
    int print_mode = PRINT_MODE;
    const char *filename = 'data/arc130.mtx';
    
    read_matrix(&row_ptr, &col_ind, &values, filename, &num_rows, &num_cols, &num_vals);
    
    float *x = (float *) malloc(num_rows * sizeof(float));
    float *y = (float *) malloc(num_rows * sizeof(float));
    for (int i = 0; i < num_rows; ++i) {
        x[i] = 1.0;
        y[i] = 0.0;
    }
    
    if (print_mode == 2) {
        // Values Array
        fprintf(stdout, "Values Array:\n");
        for (int i = 0; i < num_vals; ++i) {
            fprintf(stdout, "%.6f ", values[i]);
        }

        // Row Pointer Array
        fprintf(stdout, "\n\nRow Pointer Array:\n");
        for (int i = 0; i < (num_rows + 1); ++i) {
            fprintf(stdout, "%d ", row_ptr[i]);
        }

        // Column Indices Array
        fprintf(stdout, "\n\nColumn Indices Array:\n");
        for (int i = 0; i < num_vals; ++i) {
            fprintf(stdout, "%d ", col_ind[i]);
        }
        
        fprintf(stdout, "\n\nInitial Vector:\n");
        for (int i = 0; i < num_rows; ++i) {
            fprintf(stdout, "%.1f ", x[i]);
        }
    }
    
    // Time the iterations
    clock_t start = clock();
    for (int i = 0; i < num_repeat; ++i) {
        spmv_csr(row_ptr, col_ind, values, num_rows, x, y);
        
        // Moving the ith result in order to use it in the ith + 1 iteration
        for (int i = 0; i < num_rows; ++i) {
            x[i] = y[i];
            y[i] = 0.0;
        }
    }
    clock_t stop = clock();
    elapsed_time = (((float) (stop - start)) / CLOCKS_PER_SEC) * 1000; // ms

    // Print resulting vector
    if (print_mode == 2) {
        fprintf(stdout, "\n\n Resulting Vector:\n");
        for (int i = 0; i < num_rows; i++) {
            fprintf(stdout, "%.6f ", x[i]);
        }
        fprintf(stdout, "\n");
    }
    printf("\nSerial Running time:  %.4f ms\n", elapsed_time);
    free(row_ptr);
    free(col_ind);
    free(values);
    return EXIT_SUCCESS;
}