#include <stdio.h>
#include <stdlib.h>

__global__
void calc_meanshift2(float* y_new, float* y_old, float* meanshift)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	float tempY_new = y_new[i];
	float tempY_old = y_old[i];
	
	meanshift[i] = (tempY_new-tempY_old)*(tempY_new-tempY_old);
}

__device__
float kernel_fun(float x, float sigma2)
{	
	if( x > sigma2) 
		return 0;
	else
		return exp(-x/2/sigma2);
}

__global__
void calc_Kernel_Matrix(int N, int D, float *x, float *y, float *K, int sigma2)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	// It is also true that ( k == (N*i + j) )

	// Calc Dist...
	float dist = 0;
	 for(int d=0; d<D; d++)
	 	dist+= (y[i*D+d] - x[j*D+d])*(y[i*D+d] - x[j*D+d]); 
	
	K[i*N+j] = kernel_fun(dist, sigma2);
}


__global__
void kernel_sum_div(int D, float* y_new, float* K_sum)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	for(int d=0; d<D; d++)
		y_new[i*D+d] = y_new[i*D+d]/K_sum[i];
}




__global__ void kernel_Dvec_mult(int N, int D, float* K, float* x, float* Kx, int d)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;

	Kx[i*N+j] = K[i*N+j]*x[j*D+d]; 
}


__global__ void copy_to_y(int D, float* d_y_new, float* kernelXsum, int d)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	d_y_new[i*D+d] = kernelXsum[i];
}




 // __global__
// void kernelX_dot_product(int N, int D, int d, float* K, float* x, float* reducted_vec)
// {
// 	extern __shared__ float reduction_cache[] ;

// 	//thread ID on each row of blocks
// 	int tid = blockDim.x * blockIdx.x + threadIdx.x; 
// 	int cache_i = threadIdx.x;


// 	/* This UNROLLS the elements of x, "outside" the grid's index range.
// 		In the case of N=600, threadsPerBlock=256 and 2 blocks in total, 
// 		we have 600-256*2=88 additions done in parallel, before the reduction of the 512 threads.

// 		incase the index-range > N, the reduction scheme will simply add some zeros to the vector. 
// 		This allows as to oversubscribe in terms of threads and blocks. 
// 	*/
// 	int offset = N*blockIdx.y;
// 	float temp=0;
// 	while (tid < N)
// 	{
// 		temp += K[tid+offset]*x[tid*D+d]; 
// 		tid += blockDim.x * gridDim.x;
// 	}

// 	/* Load x-data  into local shared memory. 
// 		As mentioned before, some entries are small sums of
// 		 x's outside the grid's range  */
// 	reduction_cache[cache_i] = temp;	
// 	__syncthreads();
	
// 	// Begin the reduction per shared-memory-block
// 	for(int i=blockDim.x/2; i>0; i>>=1)
// 	{	
// 		if(cache_i < i)
// 			reduction_cache[cache_i] += reduction_cache[cache_i+i];  
// 		__syncthreads();
// 	}

// 	// Final Sum is stored in global array, with stride d, to match the NxD dimensionality of the input dataset.
// 	if(cache_i==0)
// 		reducted_vec[blockIdx.y*gridDim.x + blockIdx.x + d] = reduction_cache[cache_i];	
// }


// void WR_kernelX_dow_product(int N, float* d_K, float* d_x, /*out*/ ReductionCache* rc )
// {

// 	dim3 blockDim2(4, 1, 1); 
//   		dim3 gridDim2(N/4,N,1); 
//   		size_t cache_size = 4*N*sizeof(float);
// 	  	kernelX_dot_product<<<gridDim2, blockDim2, cache_size>>>(N,D,0, d_KernelMatrix, d_x, d_y_new);
// 		kernelX_dot_product<<<gridDim2, blockDim2, cache_size>>>(N,D,1, d_KernelMatrix, d_x, d_y_new);
// 	  	//reduction_sum<<<L/256, 256, 256*sizeof(float) >>>(N/4, d_y_new, d_y_new); 




// 	if(rc->blocksNum == 1)
// 	{	
// 		kernelX_dot_product<<<rc->gridDim, rc->blockDim, rc->cache_size>>>(N,D,0, d_K,d_x, rc->d_sum);
// 		kernelX_dot_product<<<rc->gridDim, rc->blockDim, rc->cache_size>>>(N,D,1, d_K,d_x, rc->d_sum);
// 	}
// 	else
// 	{	
// 		// We need multiple reduction calls!
// 		reduction_sum <<<rc->gridDim, rc->blockDim, rc->cache_size>>>(N, d_A, rc->d_reduced_vec);		
			
// 		/* Reduct the final reduction vector! */
	
// 		// Ideally we would like threads_num==length(reduced_vec)/numRow. 
// 		However threads_num2 must be a power of 2. Thus:
		
// 		int threads_num2 = exp2f(floor(log2f(rc->reduced_vec_length/rc->rowNum))); 
// 		if(threads_num2>512)
// 			threads_num2=512;
// 		//printf("THREADS: %d RED_VEC %d\n", threads_num2, rc->reduced_vec_length/rc->rowNum );

// 		dim3 gridDim2(1,rc->rowNum,1);
// 		dim3 blockDim2(threads_num2,1,1);
// 		reduction_sum<<<gridDim2, blockDim2, threads_num2*sizeof(float)>>>\
// 			(rc->gridDim.x, rc->d_reduced_vec, rc->d_sum); //

// 		// WARNING: launching with original thread_num might be too much. 
// 		// SOLUTION: Find power-of-2 nearest to block_num 
// 	}	
// }