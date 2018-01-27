#include <stdio.h>
#include <stdlib.h>


#include "cuda_reduction.h"


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


void WR_vector_reduction(float* d_x, int N, int threads_num, /*out*/ float * sum)
{
	int blocks_num = N/threads_num; // The result will be automatically floored.
	size_t cache_mem = thread_num*sizeof(float);

	float* d_reducted_vec
	float* reducted_vec = (float*) malloc(blocks_num*sizeof(float));
	cudaMalloc((void**) &d_reducted_vec, blocks_num*sizeof(float) ); 


	reduction_SM <<<blocks_num, thread_num, cache_mem>>>(N, d_x, d_reducted_vec);

	// We call reduction kernel again to sum up the rest of the d_reducted_ven  


	cudaFree(d_reducted_vec);


}