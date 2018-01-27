#include <stdio.h>
#include <stdlib.h>

#include "cuda_reduction.h"

#include "cuda_wrappers.h"


void init_reduction_cache_1D(int N, int threads_num, /*ouit*/ ReductionCache* rc)
{
	rc->blockDim.x = threads_num;
	rc->blockDim.y = 1;
	rc->blockDim.z = 1;

	rc->gridDim.x = N/threads_num; //block size
	rc->gridDim.y = 1; // One row of block for each matrix row 
	rc->gridDim.z = 1;

	rc->cache_size = threads_num*sizeof(float);

	cudaMalloc((void**) &(rc->d_reduced_vec), rc->gridDim.x*sizeof(float));
	cudaMalloc((void**) &(rc->d_sum), 1*sizeof(float));
}

void init_reduction_cache2D(int N, int threads_num, /*ouit*/ ReductionCache* rc)
{
	rc->blockDim.x = threads_num;
	rc->blockDim.y = 1;
	rc->blockDim.z = 1;

	rc->gridDim.x = N/threads_num; //block size
	rc->gridDim.y = N; // One row of block for each matrix row 
	rc->gridDim.z = 1;

	rc->cache_size = N*threads_num*sizeof(float);

	cudaMalloc((void**) &(rc->d_reduced_vec), N*(rc->gridDim.x)*sizeof(float));
	cudaMalloc((void**) &(rc->d_sum), N*sizeof(float));
}

void delete_reduction_cache(ReductionCache* reductionCache)
{
	cudaFree(reductionCache->d_reduced_vec);
	cudaFree(reductionCache->d_sum);
}


/**********************************************************************************
*
*
*
**********************************************************************************/

void WR_vector_reduction(int N, float* d_x,  /*out*/ ReductionCache* rc)
{

	reduction_SM <<<rc->gridDim, rc->blockDim, rc->cache_size>>>\
		(N, d_x, rc->d_reduced_vec);

	// WARNING: launching with original thread_num might be too much. 
	// SOLUTION: Find power-of-2 nearest to block_num 
	reduction_SM <<<1, rc->blockDim, rc->cache_size>>>\
		(rc->gridDim.x, rc->d_reduced_vec, rc->d_sum);
}

void WR_matrix_row_reduction(int N, float* d_A, /*out*/ ReductionCache* rc )
{
	matrix_sum_row_SM <<<rc->gridDim, rc->blockDim, rc->cache_size>>>(N, d_A, rc->d_reduced_vec);		
	
	// Reduct the final reduction vector!
	// WARNING: launching with original thread_num might be too much. 
	// SOLUTION: Find power-of-2 nearest to block_num 
	dim3 gridDim2(1,N,1);
	matrix_sum_row_SM<<<gridDim2, rc->blockDim, rc->cache_size>>>\
	(rc->gridDim.x, rc->d_reduced_vec, rc->d_sum);
}


/**********************************************************************************
*
*
*
**********************************************************************************/

void WR_reduction_GM(int blocks_num, int threads_num, int N, float* x, float* reduction_cache )
{
	reduction_GM_step1<<<blocks_num, threads_num>>>( N,  x, reduction_cache);

	cudaDeviceSynchronize(); // <- IMPORTANT.
	
	// The way we address the reduction_cache doesnt really matter. This is gonna suck anyway
	for(int i=threads_num/2; i>0; i>>=1)
	{
		reduction_GM_step2<<<blocks_num, threads_num>>>(reduction_cache, i);
		cudaDeviceSynchronize(); // <- IMPORTANT.
	}	
}
