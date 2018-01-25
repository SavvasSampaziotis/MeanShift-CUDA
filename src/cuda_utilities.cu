#include <stdio.h>
#include <stdlib.h>



/* 
	A - B = C 
*/
__global__
void vectorSub(float* A, float* B, float* C)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	C[i] = A[i] - B[i];
}

/* 
	A <- A.^2
*/
__global__
void vectorPow2(float* A)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	A[i] = A[i]*A[i];
}


/* 
	THis function is for the explicit purpose of 
	element-wise multiplication of KernelSum and X[N,D] dataset.

	Due to the dact that each X is 2D matrix with coalescent data, 
	this cannot be implemented with basic vector multiplication

	The result of each
*/
__global__
void matrix_mult(int D, int d, float* KernelMatrix, float* X, float* resultMatrix)
{
	// int j = blockDim.y*blockIdx.y + threadIdx.y;
	// int i = blockDim.x*blockIdx.x + threadIdx.x;
	
	// for(int d=0; d<D; d++)
		// resultMatrix[i*D+d] = KernelSum[i]*X[i*D+d]; 
}

/*
	Simple add-reduction, with SHARED MEMORY.

	Ref. Code
		http://cuda-programming.blogspot.gr/2013/01/vector-dot-product-in-cuda-c.html
		and
		Reduction.pdf from the courses material
*/
__global__
void reduction_SM(int N, float* x, float* reducted_vec)
{
	extern __shared__ float reduction_cache[] ;

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int cache_i = threadIdx.x;


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


/*
	Simple add-reduction, WITHOUT shared memory. This is split into 2 parts

	This functions needs an extra 1D array of length NUM_BLOCKS*NUM_THREADS, 
	to replace the sweet-sweet shared memory between threads.

	Ref. Code
		http://cuda-programming.blogspot.gr/2013/01/vector-dot-product-in-cuda-c.html
		and
		Reduction.pdf from the courses material
*/
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

/*
	Simple add-reduction, WITHOUT shared memory. This is split into 2 parts.

	This functions needs an extra 1D array of length NUM_BLOCKS*NUM_THREADS, 
	to replace the sweet-sweet shared memory between threads.

	This is the part where we iterate over and over.
	 But because there is no thread-sync betweem blocks,
	  the iteration is achieved with multiple kernel-launching. Enjoy.

	Ref. Code
		http://cuda-programming.blogspot.gr/2013/01/vector-dot-product-in-cuda-c.html
		and
		Reduction.pdf from the courses material
*/
__global__
void reduction_GM_step2(float* reduction_cache, int i)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int i_gd = blockDim.x * blockIdx.x + i;  // Transform thread id, to global memory index...

	//Begin the reduction of individual memory block, from global mem	
	if( threadIdx.x < i)
		reduction_cache[tid] += reduction_cache[tid+i];  		
}

/*
	This is NOT a CUDA kernel function. 
	It's just a wrapper for the kernel launching that needs to be done
*/
void reduction_GM(int blocks_num, int threads_num, int N, float* x, float* reduction_cache )
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