/*

	Header file for cuda_utilities.c implementationfile  
*/

#ifndef CUDA_UTILITIES_H
#define  CUDA_UTILITIES_H


__global__
void calc_meanshift2(float* y_new, float* y_old, float* meanshift);

__global__
void matrix_mult(int D, int d, float* KernelMatrix, float* X, float* resultMatrix);

__global__
void matrix_sum_row_SM(int N,  float* K, float* reducted_vec);

__global__
void reduction_SM(int N, float* x, float* reducted_vec);


void reduction_GM(int threads_num, int blocks_num, int N, float* x, float* reduction_vec );

#endif