#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/parser.h"

int main(int argc, const char * argv[]) {
    fprintf(stdout, "============================\n");
    fprintf(stdout, "*** Parallel SpMV (CUDA) ***\n");
    fprintf(stdout, "----------------------------\n");
    fprintf(stdout, "**** Author: Zonta Elia ****\n");
    fprintf(stdout, "----------------------------\n");
    fprintf(stdout, "============================\n");
    if (argc != 5) {
        fprintf(stdout, "Usage : \n <./bin/SpVM-CUDA> <threads num> <num of iterations> <print mode [1 default, 2 view data info]> <file path>\n");
        exit(-1);
    }
    
    int *row_ptr, *col_ind, num_rows, num_cols, num_vals, numSMs;
    float *values;
    
    int num_thread = atoi(argv[1]);
    int num_repeat = atoi(argv[2]);
    int print_mode = atoi(argv[3]);
    const char *filename = argv[4];
    
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
        for (int i = 0; i < num_vals; i++) {
            fprintf(stdout, "%.6f ", values[i]);
        }
        
        // Column Indices Array
        fprintf(stdout, "\n\nColumn Indices Array:\n");
        for (int i = 0; i < num_vals; i++) {
            fprintf(stdout, "%d ", col_ind[i]);
        }
        
        // Row Pointer Array
        fprintf(stdout, "\n\nRow Pointer Array:\n");
        for (int i = 0; i < (num_rows + 1); i++) {
            fprintf(stdout, "%d ", row_ptr[i]);
        }
        
        fprintf(stdout, "\n\nInitial Vector:\n");
        for (int i = 0; i < num_rows; i++) {
            fprintf(stdout, "%.1f ", x[i]);
        }
        
        fprintf(stdout, "\n\nResulting Vector:\n");
    }
    
    // Allocate on device
    int *d_row_ptr, *d_col_ind;
    float *d_values, *d_x, *d_y;
    cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_ind, num_vals * sizeof(int));
    cudaMalloc((void**)&d_values, num_vals * sizeof(float));
    cudaMalloc((void**)&d_x, num_rows * sizeof(float));
    cudaMalloc((void**)&d_y, num_rows * sizeof(float));
    
    // Get number of SMs
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    
    // Copy from host to device
    cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice);
    
    // Time the iterations
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int i = 0; i < num_repeat; i++) {
        cudaMemcpy(d_x, x, num_rows * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, num_rows * sizeof(float), cudaMemcpyHostToDevice);
        
        // Call kernel function
        spmv_csr<<<32 * numSMs, num_thread>>>(d_row_ptr, d_col_ind, d_values, num_rows, d_x, d_y);
        
        // Copy the result to x_{i} at the end of each iteration, and use it in iteration x_{i+1}
        cudaMemcpy(y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_rows; i++) {
            x[i] = y[i];
            y[i] = 0.0;
        }
    }
    
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    // Print resulting vector
    if (print_mode == 2) {
        for (int i = 0; i < num_rows; i++) {
            fprintf(stdout, "%.6f ", x[i]);
        }
        fprintf(stdout, "\n");
    }
    
    // Print elapsed time
    printf("\nParallel Running time:  %.4f ms\n", elapsed_time);
    printf("Num SMs: %d\n", numSMs);
    
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    
    free(row_ptr);
    free(col_ind);
    free(values);
    
    return 0;
}

