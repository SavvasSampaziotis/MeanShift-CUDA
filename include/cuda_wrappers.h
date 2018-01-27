/*
	Header file for cuda_wrapper.c implementation file  
	
	This library header contains various wrapper function for CUDA-reduction kernels,
	as well as containers and handlers for temporary data structures. 

	Author: Savvas Sampaziotis

*/


#ifndef CUDA_WRAPPER_H
#define  CUDA_WRAPPER_H

#include "cuda_reduction.h"

/*
	This struct contains various metadata needed for reduction-kernel launching.
	
	It purpose is to make the main code more readable, and relieve me from numerous device-allocated arrays and blocksizes.

	This also enables me to write more efficient code, since memory (de)allocation needs to be kept at a minimum.

	Ref:
		https://www.cs.virginia.edu/~mwb7w/cuda_support/memory_management_overhead.html
*/
typedef struct ReductionCacheStruct
{
	
	dim3 gridDim;
	dim3 blockDim;
	float* d_reduced_vec;
	float* d_sum;
	size_t cache_size;
} ReductionCache;


/*
	"Constructor" of reductionCache struct.

	Input Args:
		int N: the length of the vector targeted for reduction
		int threads_num: Must be a power of 2. You need to choose its value wisely, 

	Output Args	
		ReductionCache* rc: the struct that holds the reduction metadata. 
*/
void init_reduction_cache_1D(int N, int threads_num, /*out*/ ReductionCache* rc);


/*
	Same as init_reduction_cache1D.

	N = number of rows as well as length of columns...
*/
void init_reduction_cache2D(int N, int threads_num, /*out*/ ReductionCache* rc);


/*
	"Deconstructor" of reductionCache struct.

	Input Args:	
		ReductionCache* reductionCache

	This Basically deallocates the reduction cache vector from the GPU memory. 	
*/
void delete_reduction_cache(ReductionCache* reductionCache);

/*
	launches 2 reduction-kernels.

	The first one runs on gridDim and BlockDim specified by arg ReductionCache.

	The second one sums-up the reduction_vec produxed by the first launch.   
*/
void WR_vector_reduction(int N, float* d_x,  /*out*/ ReductionCache* rc);

void WR_matrix_row_reduction(int N, float* d_x,  /*out*/ ReductionCache* rc);

/*	
	NOT WOKRING

	This launches the CUDA kernel functions for the simple reduction 
		without the global memory access. 
*/
void WR_reduction_GM(int blocks_num, int threads_num, int N, float* x, float* reduction_cache );

#endif
