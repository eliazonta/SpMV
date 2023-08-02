#ifndef PARSER_H
#define PARSER_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


void read_matrix(int **row_ptr, int **col_ind, float **values, const char *filename, int *num_rows, int *num_cols, int *num_vals);

#endif // PARSER_H

