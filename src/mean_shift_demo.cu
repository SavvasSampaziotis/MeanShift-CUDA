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
		return expf(-x/2/sigma2);
}

__global__
void mean_shift(int N, int D, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float sigma2 = 1;
	float sumDen = 0;
	float dist; // this stores the ||y-x|| ^2 
	float temp; // This stores the kernel funciton result k(.)
	
	float* nomSum = cudaMalloc((void))

	int d,j;

	for(j=0; j<N; j++)
	{	
		dist = 0;
		for(d=0; d<D; d++)
			dist+= (y[i*D+d] - x[j*D+d])*(y[i*D+d] - x[j*D+d]); 

		temp = kernel_fun(dist, sigma2);
		sumDen += temp;

		for(d=0; d<D; d++)
			y[i*D+d] += temp*x[i*D+d];
	}

	for(d=0; d<D; d++)
		y[i*D+d] = y[i*D+d]/sumDen;

	// D = 2;
	y[i*D] = x[i*D];
	y[i*D+1] = x[i*D+1];
	// x[i*D+1] = -i;
}

 

int main(int argc, char** argv)
{
	//cudaError_t error;
	int N,	D;
	float* X, *Y;
	float* d_x, *d_y;
  	int i;
	
	// Read Feature-Datapoints
	read_dataset(&N, &D, &X);
	int L = N*D;

	// Allocate memory for mean-shifted cluster centers
	Y = (float*) malloc(L*sizeof(float));
	for(i=0; i<L; i++)
		Y[i] = 0;

	// Allocate memory of datasets
	cudaMalloc((void**) &d_x, L*sizeof(float)); 
	cudaMalloc((void**) &d_y, L*sizeof(float));

	cudaMemcpy(d_x, X, L*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_y, Y, L*sizeof(float), cudaMemcpyHostToDevice);

  	// Run mean_shift for 
  	dim3 gridDim;
  		gridDim.x = N; // Number of blocks
  		gridDim.y = 1; // Number of blocks
  	
  	dim3 blockDim; // Number of Threads per block
  		blockDim.x = 1;
  		blockDim.y = 1;
  		blockDim.z = 1;
  	

  	for(i=0;i<15;i++)
		mean_shift<<<gridDim, blockDim>>>(N, D, d_x, d_y);
	
	cudaMemcpy(Y, d_y, L*sizeof(float), cudaMemcpyDeviceToHost);

	write_meanshift_result(N,D,Y);

	print_dataset(100, D, Y);

	cudaFree(d_x);
	cudaFree(d_y);	
	free(X);
	free(Y);
}

void write_meanshift_result(int N, int D, float* data)
{
	FILE *fp = fopen( "./data/mean_shift_output.bin", "w+");
	if(fp < 0)
	{
		printf("ERROR OPENING DATA FILE\n");
		return;
	}

	fwrite(&N, 1, sizeof(int), fp);
	fwrite(&D, 1, sizeof(int), fp);

	fwrite(data, D*N, sizeof(float), fp);

	fclose(fp);
}

void read_dataset(int* N, int* D, float** data)
{
	FILE *fp = fopen( "./data/r15.bin", "r");
	if(fp < 0)
	{
		printf("ERROR OPENING DATA FILE\n");
		*N = 0;
		*D = 0;
		return;
	}
	
	fread(N, 1, sizeof(int), fp);
	fread(D, 1, sizeof(int), fp);
	
	int L = (*N)*(*D);
	*data = (float*) malloc( L*sizeof(float));
	fread(*data, L, sizeof(float), fp);

	//printf("%d %d\n",*N,*D);
	//print_dataset(*N,*D, *data);

	fclose(fp);
}

void print_dataset(int N, int D, float* data)
{
	int i,j;
	printf("\n--------------------\n");
	for(i=0; i<N; i++)
	{
		printf("%d\t",i);
		for(j=0; j<D; j++)
		{
			 printf("%f\t", data[i*D+j]);
		}
		printf("\n");
	}
		

}
