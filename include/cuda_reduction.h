/*
	Header file for cuda_utilities.c implementationfile  
	
	This library header contains various CUDA-reduction implementations

	Author: Savvas Sampaziotis
*/

#ifndef CUDA_REDUCTION_H
#define  CUDA_REDUCTION_H


/*
	Simple add-reduction, with SHARED MEMORY.
	
	Arguments
		int N:	length of x
		float* x
		float* reducted_vec[BLOCK_NUM]: reduced sum of each block
	 
	TIP:
		reduction_vec must be of size [BLOCK_NUM] and stores the reduced sum of each block of threads.
		
		The final result can be calculated by calling again reduction_SM for the reducted_vec like this (e.g):
		reduction_SM <<< 150/64, 64, cache_mem >>> (150, x, d_reduced_vec); // 3 blocks - 64 threads per block.
		reduction_SM <<< 1, 8, cache_mem >>> (N/256, d_reduced_vec, sum); // 1 block - 4 threads per block.
		 //There is ONE extra thread that will contribute zero to the final result. 

	Notes:
		Number of threads per block MUST be a power of 2. 

		Number of total threads blockNum*threadNum can be less or greater 
			than length of vector x. This works either way. 
			However, having less threads than vector elements mean that  there 
			will be more non-parallel global memory accesses. 

	Ref. Code
		http://cuda-programming.blogspot.gr/2013/01/vector-dot-product-in-cuda-c.html
		and
		Reduction.pdf from the courses material
*/
__global__ void reduction_SM(int N, float* x, float* reducted_vec);


/* 
	Performs reduction on each row of a matrix. 

	This function need to 
*/
__global__ void matrix_sum_row_SM(int N,  float* K, float* reducted_vec);


/*
Simple add-reduction, WITHOUT shared memory. This function is split into 2 seperate kernels, so that the blocks can sync

This functions needs an extra 1D array of length NUM_BLOCKS*NUM_THREADS, to replace the sweet-sweet shared
memory between threads.

Ref. Code
	http://cuda-programming.blogspot.gr/2013/01/vector-dot-product-in-cuda-c.html
	Reduction.pdf from the courses material
*/
__global__ void reduction_GM_step1(int N, float* x, float* reduction_cache);


/*
Simple add-reduction, WITHOUT shared memory. This function is split into 2 seperate kernels, so that the blocks can sync

This functions needs an extra 1D array of length NUM_BLOCKS*NUM_THREADS, to replace the sweet-sweet shared
memory between threads.

This is the part where we iterate over and over. But because there is no thread-sync betweem blocks, 
the iteration is achieved with multiple kernel-launching. Enjoy.

Ref. Code
	http://cuda-programming.blogspot.gr/2013/01/vector-dot-product-in-cuda-c.html
	Reduction.pdf from the courses material
*/
__global__ void reduction_GM_step2(float* reduction_cache, int i);


#endif