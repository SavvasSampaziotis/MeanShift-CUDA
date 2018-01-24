#include <stdio.h>
#include <stdlib.h>

#include "time_measure.h"

#include "array_utilities.h"


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
	// int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	// int k = blockId * (blockDim.x * blockDim.y) + \
	(threadIdx.y * blockDim.x) + threadIdx.x;

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	// It is also true that ( k == (N*i + j) )

	// Calc Dist...
	float dist = 0;
	 for(int d=0; d<D; d++)
	 	dist+= (y[i*D+d] - x[j*D+d])*(y[i*D+d] - x[j*D+d]); 
	
	K[i*N+j] = kernel_fun(dist, sigma2);
}

__global__
void calc_next_Y(int N, int D, float *x, float *y, float *K)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x; // [0..N-1] -> N kernels are launched
	
	int j,d;
	float sumD = 0; // = 
	
	for(d=0; d<D; d++)
		y[i*D+d] = 0;

	for (j = 0; j < N; j++)
	{		
		float temp = K[i*N + j];
		// float temp = K[j*N + i];

		/* Calculate the sum of the denominator*/
		sumD += temp;
		
		/* Inner Product between K-matrix and X. */
		for(d=0; d<D; d++)
			y[i*D+d] += temp*x[j*D+d];
			// y[i*D+d] = x[j*D+d];
	}

	for(d=0; d<D; d++)
		y[i*D+d] = y[i*D+d]/sumD; 
}



int main(int argc, char** argv)
{
	// int i;	// temp index
	// cudaError_t error;

	int N,	D;
	float* X, *Y; // Original and mean-shifted Datapoints  
  	float* d_x; // Same as X and Y, but in CUDA-memory
  	
  	float *d_y_new, *d_y; // CUDA ptrs for 2 sets of Y. 
  	// These are used alternatively for the efficient calculation of m=y_new-y_prev
  	
  	// CUDA-mem: stores the result of the kernel function k(|y_i-x_j|), for each i,j. 
	float* d_KernelMatrix; 

	float* d_meanshift; 
	
	// Read Feature-Datapoints
	read_dataset(&N, &D, &X);
	int L = N*D;

	// Allocate memory for mean-shifted cluster centers
	Y = (float*) malloc(L*sizeof(float));
	
	// Allocate CUDA memory.
	cudaMalloc((void**) &d_x, L*sizeof(float)); 
	cudaMalloc((void**) &d_y, L*sizeof(float));
	cudaMalloc((void**) &d_y_new, L*sizeof(float));
	cudaMalloc((void**) &d_KernelMatrix, N*N*sizeof(float)); 
	cudaMalloc((void**) &d_meanshift, L*sizeof(float));

	// Copy Dataset to DUVA global mem
	cudaMemcpy(d_x, X, L*sizeof(float), cudaMemcpyHostToDevice);

	// Y:=X  Initial Conditions of the algorithm 
	cudaMemcpy(d_y, X, L*sizeof(float), cudaMemcpyHostToDevice); 

  	// Mean Shift Start!
  	dim3 blockDim1(5,5,1); 
  	dim3 gridDim1(N/blockDim1.x, N/blockDim1.x,1); 
  	
  	dim3 blockDim2(N/2,1,1); 
  	dim3 gridDim2(N/blockDim2.x,1,1); 
  	
  	dim3 blockDim3(N/2,1,1); 
  	dim3 gridDim3(N/blockDim3.x,1,1); 
  	

  	TimeInterval timeInterval;
  	double seqTime;
  	tic(&timeInterval);
  	for (int i = 0; i < 15; ++i)
  	{  		
	  	calc_Kernel_Matrix<<< gridDim1, blockDim1>>>(N, D, d_x, d_y, d_KernelMatrix, 1);

		calc_next_Y<<< gridDim2, blockDim2>>>(N, D, d_x, d_y_new, d_KernelMatrix);	
	

		/* Calc Frobenius Norm: sum(sum(d_meanshift.^2))*/
		// m = y'-y;
		vectorSub<<< gridDim2, blockDim2>>>(d_y_new, d_y, d_meanshift);	
		// m = m.^2
		vectorPow2<<<gridDim2, blockDim2>>>(d_meanshift);
		// Calc Sum with reduction
		reduction0<<< N,1 >>>(d_meanshift);

		// Switch pointers, so that there wont be any nedd for memcpy and stuff..
		float* temp = d_y;
		d_y = d_y_new;  
		d_y_new = temp;
  	}

  	seqTime = toc(&timeInterval);
	printf("Calc Time = %f\n", seqTime);

	cudaMemcpy(Y, d_y, L*sizeof(float), cudaMemcpyDeviceToHost); 
	write_meanshift_result(N,D,Y);
	
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_KernelMatrix);	
	free(X);
	free(Y);
}

