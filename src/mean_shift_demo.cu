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
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int k = blockId * (blockDim.x * blockDim.y) + \
	(threadIdx.y * blockDim.x) + threadIdx.x;

	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	// It is also true that ( k == (N*i + j) )

	// Calc Dist...
	float dist = 0;
	 for(int d=0; d<D; d++)
	 	dist+= (y[i*D+d] - x[j*D+d])*(y[i*D+d] - x[j*D+d]); 
	
	K[k] =  kernel_fun(dist, sigma2);
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
		/* Calculate the sum of the denominator*/
		sumD += K[i*N + j];
		
		/* Inner Product between K-matrix and X. */
		for(d=0; d<D; d++)
			y[i*D+d] += x[j*D+d]*K[i*N + j];
	}

	 for(d=0; d<D; d++)
		y[i*D+d] = y[i*D+d]/sumD; 

	for(int d=0; d<D; d++)
		y[i*D+d] = K[i*N + 0]; 	
}



int main(int argc, char** argv)
{
	// int i;	// temp index
	cudaError_t error;

	int N,	D;
	float* X, *Y; // Original and mean-shifted Datapoints  
  	float* d_x, *d_y; // Same as X and Y, but in CUDA-memory
  	
  	// CUDA-mem: stores the result of the kernel function k(|y_i-x_j|), for each i,j. 
	float* d_KernelMatrix;  
	
	// Read Feature-Datapoints
	read_dataset(&N, &D, &X);
	int L = N*D;

	// Allocate memory for mean-shifted cluster centers
	Y = (float*) malloc(L*sizeof(float));
	
	// Allocate CUDA memory.
	cudaMalloc((void**) &d_x, L*sizeof(float)); 
	cudaMalloc((void**) &d_y, L*sizeof(float));
	cudaMalloc((void**) &d_KernelMatrix, N*N*sizeof(float)); 

	// Copy Dataset to DUVA global mem
	cudaMemcpy(d_x, X, L*sizeof(float), cudaMemcpyHostToDevice);

	// Y:=X  Initial Conditions of the algorithm 
	cudaMemcpy(d_y, X, L*sizeof(float), cudaMemcpyHostToDevice); 

  	// Mean
  	dim3 blockDim; 
  	dim3 gridDim; 
  	TimeInterval timeInterval;
  	double seqTime;
  	tic(&timeInterval);
  	for (int i = 0; i < 10; ++i)
  	{  		
	  	blockDim.x = 5;
	  	blockDim.y = 5;
	  	gridDim.x = N/blockDim.x;
	  	gridDim.y = N/blockDim.y;

		calc_Kernel_Matrix<<<gridDim, blockDim>>>(N, D, d_x, d_y, d_KernelMatrix, 1);	

	  	blockDim.x = N/2; // NUmber of threads/block
	  	blockDim.y = 1; 	
	  	gridDim.x = N/blockDim.x; // We want N-blocks 
	  	gridDim.y = 1;
		//calc_next_Y<<<gridDim, blockDim>>>(N, D, d_x, d_y, d_KernelMatrix);	
  	}
  	seqTime = toc(&timeInterval);
	printf("Calc Time = %f\n", seqTime);

	cudaMemcpy(Y, d_y, L*sizeof(float), cudaMemcpyDeviceToHost); 
	// write_meanshift_result(N,D,Y);
	

	float * K = (float*) malloc(N*N*sizeof(float));
	error = cudaMemcpy(K, d_KernelMatrix, N*N*sizeof(float), cudaMemcpyDeviceToHost); 
	printf("%d\n", error);
	write_meanshift_result(N,N,K);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_KernelMatrix);	
	free(X);
	free(Y);
}

