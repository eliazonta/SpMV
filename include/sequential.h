#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

void spmv_csr(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const float *x, float *y);

#endif // SEQUENTIAL_H