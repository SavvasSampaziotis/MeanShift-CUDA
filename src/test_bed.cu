#include <stdio.h>
#include <stdlib.h>

#include "time_measure.h"
#include "array_utilities.h"
#include "cuda_reduction.h"

void compare_reduction(int thread_num);
void test_frobenius();
void test_sumrow(int colNum, int rowNum, int thread_num);

/*Some global variables*/
int N;
float* d_A; 
float* A;
float result;

int main(int argc, char** argv)
{
	N = 50000;

	A = (float*) malloc( N*sizeof(float) );
	cudaMalloc((void**) &d_A, N*sizeof(float) ); 
	result = 0;
	for (int i = 0; i < N; ++i)
	{
		// A[i] = i;
		A[i] = 1;
		// A[i] = 0.1;
		result += A[i]; // Correct Result for testing purposes 
	}
	cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);


	printf("Correct result = %f\n",result);
	compare_reduction(512);
	compare_reduction(256);
	compare_reduction(128);
	compare_reduction(64);
	compare_reduction(32);
	compare_reduction(16);
	compare_reduction(8);
	compare_reduction(4);
	compare_reduction(2);

	
	//test_sumrow(7,7,2);

	cudaFree(d_A);	
	free(A);
}

void test_sumrow(int colNum, int rowNum, int thread_num)
{
	// N = 7;
	ReductionCache rc;

	init_reduction_cache(colNum, rowNum, thread_num, &rc);

	WR_reduction(colNum, d_A, &rc);	
  	
	float* sumR = (float*) malloc(rc.rowNum*sizeof(float)); 	
	float* reduced_vec = (float*) malloc(rc.reduced_vec_length*sizeof(float)); 	
	
	cudaMemcpy(reduced_vec, rc.d_reduced_vec, rc.reduced_vec_length*sizeof(float), cudaMemcpyDeviceToHost);	
	cudaMemcpy(sumR, rc.d_sum, rc.rowNum*sizeof(float), cudaMemcpyDeviceToHost);	

	printf("\nOriginal Array");
	print_dataset(rowNum,colNum,A);

	printf("\nReduction Array");
	print_dataset(rowNum, rc.gridDim.x, reduced_vec);

	printf("\nSum Rows");
	print_dataset(rc.rowNum, 1, sumR);

	delete_reduction_cache(&rc);
}


void test_frobenius()
{
	// int N = 10;
	// int thread_num = 2;

	// int blocks_num = N/thread_num;
	// float* A = (float*) malloc( N*sizeof(float) );
	// float* d_A; // Vector to be summed with reductions
	// cudaMalloc((void**) &d_A, N*sizeof(float) ); 

	// // Init Test Vector
	// for (int i = 0; i < N; ++i)
	// {
	// 	A[i] = i;
	// }

	// cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);

	// //frobeniusNorm<<< L/300, 300>>>(d_y_new, d_y, d_meanshift);
		
	// // Memory Economy! The reduction vector is stored at the base of original data vector d_meanshift!
	// reduction_SM<<< N/256, 256, 256*sizeof(float) >>>(N, d_A, d_A); 
	// reduction_SM<<<1, 256, 256*sizeof(float) >>>(256, d_A, d_A);
	// float m_frob;
	// cudaMemcpy(&m_frob, d_A, 1*sizeof(float), cudaMemcpyDeviceToHost);  
	
	// printf("%f\n", m_frob);

	// printf("Frobenius Norm = %f\n", sumR);

	
	// free(A);
	// cudaFree(d_A);
}


/*
	This implements a simple time-efficiency experiment, 
	for the additive reduction algorithm with and without Shared Memory
*/
void compare_reduction(int thread_num)
{	
	int ITERATIONS = 100;
	
	double calcTime;
	TimeInterval timeInterval;
		
	ReductionCache rc;
	init_reduction_cache(N, 1, thread_num, &rc);
  	tic(&timeInterval);
	{
		// for(int i=0; i<ITERATIONS; i++)
		WR_reduction(N, d_A, &rc);

		//reduction_sum <<<rc->gridDim, rc->blockDim, rc->cache_size>>>(N, rc->rowNum, d_A, rc->d_reduced_vec);		
			// WR_vector_reduction(N, d_A, &rc);	
	}
	toc(&timeInterval);
	calcTime = 1000*timeInterval.seqTime/ITERATIONS; // Get average calc time

	float sumR;
	cudaMemcpy(&sumR, rc.d_sum, 1*sizeof(float), cudaMemcpyDeviceToHost);	
		
	delete_reduction_cache(&rc);
	
	// if (sumR-result != 0)
	// 	printf("[Reduction] Test Failed: %f %f \tCalc-time: %lf msec\n", sumR,result, calcTime );		
	// else
	// 	printf("[Reduction] Test Passed!\t Threads: %d\tCalc-time: %lf msec\n", thread_num, calcTime );
	printf("%lf ",calcTime );
}