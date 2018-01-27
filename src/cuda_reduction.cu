#include <stdio.h>
#include <stdlib.h>

#include "cuda_reduction.h"

 __global__
void reduction_sum(int N, int rowNum, float* X, float* reducted_vec)
{
	extern __shared__ float reduction_cache[] ;

	//thread ID on each row of blocks
	int tid = blockDim.x * blockIdx.x + threadIdx.x; 
	int cache_i = threadIdx.x;


	/* This UNROLLS the elements of x, "outside" the grid's index range.
		In the case of N=600, threadsPerBlock=256 and 2 blocks in total, 
		we have 600-256*2=88 additions done in parallel, before the reduction of the 512 threads.

		incase the index-range > N, the reduction scheme will simply add some zeros to the vector. 
		This allows as to oversubscribe in terms of threads and blocks. 
	*/
	int offset = rowNum*blockIdx.y;
	float temp=0;
	while (tid < N)
	{
		temp += X[tid+offset]; 
		tid += blockDim.x * gridDim.x;
	}

	/* Load x-data  into local shared memory. 
		As mentioned before, some entries are small sums of
		 x's outside the grid's range  */
	reduction_cache[cache_i] = temp;	
	__syncthreads();
	
	//Begin the reduction per shared-memory-block
	for(int i=blockDim.x/2; i>0; i>>=1)
	{	
		if(cache_i < i)
			reduction_cache[cache_i] += reduction_cache[cache_i+i];  

		__syncthreads();
	}

	// Final Sum is stored in global array.
	if(cache_i==0)
		reducted_vec[blockIdx.y*gridDim.x + blockIdx.x] = reduction_cache[0];	
}


void init_reduction_cache(int N, int rowNum, int threads_num, /*ouit*/ ReductionCache* rc)
{
	rc->blockDim.x = threads_num;
	rc->blockDim.y = 1;
	rc->blockDim.z = 1;

	rc->gridDim.x = N/threads_num; //block size
	rc->gridDim.y = rowNum; // One row of block for each matrix row 
	rc->gridDim.z = 1;

	rc->rowNum = rowNum;
	rc->cache_size = rowNum*threads_num*sizeof(float);

	cudaMalloc((void**) &(rc->d_reduced_vec), rowNum*(rc->gridDim.x)*sizeof(float));
	cudaMalloc((void**) &(rc->d_sum), rowNum*sizeof(float));
}


void delete_reduction_cache(ReductionCache* reductionCache)
{
	cudaFree(reductionCache->d_reduced_vec);
	cudaFree(reductionCache->d_sum);
}


void WR_vector_reduction(int N, float* d_A, /*out*/ ReductionCache* rc )
{
	reduction_sum <<<rc->gridDim, rc->blockDim, rc->cache_size>>>(N, rc->rowNum, d_A, rc->d_reduced_vec);		
	
	// Reduct the final reduction vector!
	// WARNING: launching with original thread_num might be too much. 
	// SOLUTION: Find power-of-2 nearest to block_num 
	dim3 gridDim2(1,rc->rowNum,1);
	reduction_sum<<<gridDim2, rc->blockDim, rc->cache_size>>>(rc->gridDim.x, rc->rowNum , rc->d_reduced_vec, rc->d_sum);
}
