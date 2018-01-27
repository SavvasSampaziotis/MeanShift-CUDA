#include <stdio.h>
#include <stdlib.h>


__global__
void reduction_SM(int N, float* x, /*out*/ float* reducted_vec)
{
	extern __shared__ float reduction_cache[] ;

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cache_i = threadIdx.x;

	/* This UNROLLS the elements of x, "outside" the grid's index range.
		In the case of N=600, threadsPerBlock=256 and 2 blocks in total, 
		we have 600-256*2=88 additions done in parallel, before the reduction of the 512 threads.

		incase the index-range > N, the reduction scheme will simply add some zeros to the vector. 
		This allows as to oversubscribe in terms of threads and blocks. 
	*/
	float temp=0;
	while (tid < N)
	{
		temp += x[tid];
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
		reducted_vec[blockIdx.x] = reduction_cache[0];	
}


 __global__
void matrix_sum_row_SM(int N, float* K, float* reducted_vec)
{
	extern __shared__ float reduction_cache[] ;

	//thread ID on each row of blocks
	int tid = blockDim.x * blockIdx.x + threadIdx.x; 
	
	int cache_i = threadIdx.x;

	int offset = N*blockIdx.y;
	float temp=0;
	while (tid < N)
	{
		temp += K[tid+offset]; 
		tid += blockDim.x * gridDim.x;
	}

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




__global__
void reduction_GM_step1(int N, float* x, float* reduction_cache)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	/* This UNROLLS the elements of x, "outside" the grid's index range.
		In the case of N=600, threadsPerBlock=256 and 2 blocks in total, 
		we have 600-256*2=88 additions done in parallel, before the reduction of the 512 threads.
	*/
	float temp=0;
	while (tid < N)
	{
		// temp += x[tid] + y[tid];
		temp += x[tid];
		tid += blockDim.x * gridDim.x;
	}

	/* Load x-data  into GLOBAL memory.  
		As mentioned before, some entries are small sums of
		 x's outside the grid's range  */
	reduction_cache[tid] = temp;	
}


__global__
void reduction_GM_step2(float* reduction_cache, int i)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	//int i_gd = blockDim.x * blockIdx.x + i;  // Transform thread id, to global memory index...

	//Begin the reduction of individual memory block, from global mem	
	if( threadIdx.x < i)
		reduction_cache[tid] += reduction_cache[tid+i];  		
}

