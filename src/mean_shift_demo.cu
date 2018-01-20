#include <stdio.h>
#include <stdlib.h>


void read_dataset(int* N, int* D, double** data);
void print_dataset(int N, int D, double** data);


__global__
void mean_shift(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] - y[i];
}
 

int main(int argc, char** argv)
{

	int N = -1;
	int	D = -1;
	double* data;
	read_dataset(&N, &D, &data);
/*
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));

	cudaMalloc(&d_x, N*sizeof(float)); 
	cudaMalloc(&d_y, N*sizeof(float));
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
	saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

	cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = max(maxError, abs(y[i]-4.0f));
	printf("Max error: %f\n", maxError);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);*/
}



void read_dataset(int* N, int* D, double** data)
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
	*data = (double*) malloc( L*sizeof(double));
	fread(*data, L, sizeof(double), fp);

	//printf("%d %d\n",*N,*D);
	//print_dataset(*N,*D, data);

	fclose(fp);
}

void print_dataset(int N, int D, double** data)
{
	int i,j;
	printf("\n--------------------\n");
	for(i=0; i<N; i++)
	{
		for(j=0; j<D; j++)
		{
			 printf("%lf\t", (*data)[i*D+j]);
		}
		printf("\n");
	}
		

}
