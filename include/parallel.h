#ifndef PARALLEL_H
#define PARALLEL_H

__global__ 
void spmv_csr(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const float *x, float *y);

#endif // PARALLEL_H