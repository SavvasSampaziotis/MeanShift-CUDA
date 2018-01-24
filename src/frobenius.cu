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


__global__
void reduction0(float *d_meanshift)
{

}


void reduction_kernel_calls()
{

}