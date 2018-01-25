/*

	Header file for cuda_utilities.c implementationfile  
*/

#ifndef ARRAY_UTILITIES_H
#define  ARRAY_UTILITIES_H


__global__
void vectorSub(float* A, float* B, float* C);

__global__
void vectorPow2(float* A);

__global__
void matrix_mult(int D, int d, float* KernelMatrix, float* X, float* resultMatrix);

__global__
void reduction_SM(int N, float* x, float* reducted_vec);


void reduction_GM(int threads_num, int blocks_num, int N, float* x, float* reduction_vec );

#endif