/*
	Header file for cuda_utilities.c implementation file.

	This library contains all the meanshift-specific functions that are implemented in CUDA. 

	Author: Savvas Sampaziotis
*/

#ifndef CUDA_MEANSHIFT_H
#define  CUDA_MEANSHIFT_H


/* 	Calculates the Squared Meanshift 	m(y) = (y_new-y_old).^2 

	We can avoid STRIDED ACCESS by ignoring the dataset dimensionality [N,D]. 
	The threads are aligned with the data and this provides efficient global memory access.

	This kernel must be launched with a total of N*D threads. 
	*/
__global__ void calc_meanshift2(float* y_new, float* y_old, float* meanshift);


/*  Implements the gaussian kernel function specified for this version of the meanshift algorithm.

	This is a __device__ function, called by kernel-functions only.  */
__device__ float kernel_fun(float x, float sigma2);


/*	This calculates the NxN Distance-Kernel Matrix. 

	The kernel must be called with a NxN threads in total. The block and grid size can be arbitrary. */
__global__ void calc_Kernel_Matrix(int N, int D, float *x, float *y, float *K, int sigma2);


/*
	Performs element-wise division between nominator matrix [NxD] and
	 denominator matrix [Nx1], along the first dimension of size N 
*/
__global__ void kernel_sum_div(int D, float* y_new, float* K_sum);

/*
	Matrix Multiplication between KernelMatrix and X dataset 	
*/
__global__ void kernelX_dot_product(int N, int D, int d, float* K, float* x, float* reducted_vec);


/*
	Element-wise multiplication between Kernel Matrix and Vector of dimention D. 
*/
__global__ void kernel_Dvec_mult(int N, int D, float* K, float* x, float* Kx, int d);


__global__ void copy_to_y(int D, float* d_y_new, float* kernelXsum, int d);



__global__ void calc_reduce_meanshift(int L, float* y_new, float* y_old, float* meanshift);
#endif