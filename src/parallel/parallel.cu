#include "../../include/parallel.cuh"
// Parallel SpMV 
__global__ 
void spmv_csr(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const float *x, float *y) {
    // grid stride loop for dot product
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += blockDim.x * gridDim.x) {
        float dp = 0;
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; ++j) {
            dp += values[j] * x[col_ind[j]];
        }
        
        y[i] = dp;
    }
 }