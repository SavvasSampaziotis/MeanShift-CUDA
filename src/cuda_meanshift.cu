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
void calc_Kernel_Matrix(int N, int D, float *x, float *y, float *K)
{
	int sigma2 = 1;
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