#include <stdio.h>
#include <stdlib.h>


void read_dataset(int* N, int* D, float** data);
void print_dataset(int N, int D, float* data);
void write_meanshift_result(int N, int D, float* data);

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
	int i = blockIdx.x + blockIdx.y * gridDim.x; // block ID
	int j = (threadIdx.y * blockDim.x) + threadIdx.x; // Thread ID inside Block
	int k = i*(blockDim.x * blockDim.y) + j; //Kernel Matrix ID
	
	
	K[k] = k;
}

 

int main(int argc, char** argv)
{
	//cudaError_t error;
	int N = 600;
  	// CUDA-mem: stores the result of the kernel function k(|y_i-x_j|), for each i,j. 
	float* d_KernelMatrix;  
	float* K_ = (float*) malloc( N*N*sizeof(float) );
	cudaMalloc((void**) &d_KernelMatrix, N*N*sizeof(float)); 

	
  	// Mean
  	dim3 blockDim; 
  	blockDim.x = 2;
  	blockDim.y = 4;
  	blockDim.z = 1; 	
  	dim3 gridDim; 
  	gridDim.x = N/blockDim.x; 
  	gridDim.y = N/blockDim.y;
  	gridDim.z = 1;

  	printf("[%d %d] - [%d, %d]\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y );

	calc_Kernel_Matrix<<<gridDim, blockDim>>>(N, 0, 0, 0, d_KernelMatrix, 1);
	
	cudaMemcpy( K_, d_KernelMatrix, N*N*sizeof(float), cudaMemcpyDeviceToHost );
	
	write_meanshift_result( N, N, K_ );
	
	cudaFree(d_KernelMatrix);	
	free(K_);
}
