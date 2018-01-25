#include <stdio.h>
#include <stdlib.h>

#include "cuda_utilities.h"
#include "time_measure.h"


void compare_reduction(int N, int thread_num);

int main(int argc, char** argv)
{
	//cudaError_t error;
	compare_reduction(512	, 256);
}

/*
	This implements a simple time-efficiency experiment, 
	for the additive reduction algorithm with and without Shared Memory
*/
void compare_reduction(int N, int thread_num)
{	
	int ITERATIONS = 100;
	double calcTime;
	TimeInterval timeInterval;
	int blocks_num = N/thread_num;

	// Test Data preperation
	float* A = (float*) malloc( N*sizeof(float) );
	float* d_A; // Vector to be summed with reductions
	cudaMalloc((void**) &d_A, N*sizeof(float) ); 

	// Init Test Vector
	int result = 0;
	for (int i = 0; i < N; ++i)
	{
		A[i] = i;
		result += i; // Correct Result for testing purposes 
	}
	cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);


	/*********************************************/

	// Run Reduction with Shared memory 100 times and measure time...  	
	float *reducted_SM = (float*) malloc(blocks_num*sizeof(float));
	float* d_reducted_SM;
	cudaMalloc((void**) &d_reducted_SM, N/thread_num*sizeof(float) ); 
  	
  	tic(&timeInterval);
	
	for(int i=0; i<ITERATIONS; i++)
		reduction_SM <<<blocks_num, thread_num,\
		 thread_num*sizeof(float)>>>(N, d_A, d_reducted_SM);		
	
	toc(&timeInterval);
	calcTime = timeInterval.seqTime/ITERATIONS; // Get average calc time

	cudaMemcpy(reducted_SM, d_reducted_SM, \
		blocks_num*sizeof(float), cudaMemcpyDeviceToHost);	
	
	float sumR =0;
	for(int i=0; i<N/thread_num; i++)
		sumR += reducted_SM[i];
	
	// Print if 
	if (sumR-result != 0)
		printf("[ReductionSM] Test Failed: %f \tCalc-time: %lf sec\n", result-sumR, calcTime );		
	else
		printf("[ReductionSM] Test Passed!\tCalc-time: %lf sec\n", calcTime );		
	
	cudaFree(d_reducted_SM);	
	free(reducted_SM);

	/*********************************************/

	// Run Reduction WITHOUT Shared memory 100 times and measure time...  
	float * reducted_GM = (float*) malloc(thread_num*blocks_num*sizeof(float));
	float * d_reducted_GM;
	
	cudaMalloc((void**) &d_reducted_GM, \
		thread_num*blocks_num*sizeof(float) );

  	tic(&timeInterval);
		for(int i=0; i<ITERATIONS; i++)
		{
			// THis is a wrapper of the stuff that needs to be launched...
			reduction_GM(blocks_num, thread_num, N, d_A, d_reducted_GM);	
		}
	toc(&timeInterval);
	calcTime = timeInterval.seqTime/ITERATIONS; // Get average calc time

	cudaMemcpy(reducted_GM, d_reducted_GM, \
		thread_num*blocks_num*sizeof(float), cudaMemcpyDeviceToHost);	
	
	// Final Sum is stored in global array, at indeces 0,threads_num, 2*threads_num etc...
	float sumR2 =0;
	for(int i=0; i<blocks_num; i++)
		sumR2 += reducted_GM[i*thread_num];

	
	if (sumR2-result != 0)
		printf("[ReductionGM] Test Failed: %f \tCalc-time: %lf sec\n", result-sumR2, calcTime );		
	else
		printf("[ReductionGM] Test Passed!\tCalc-time: %lf sec\n", calcTime );		
	
	cudaFree(d_reducted_GM);	
	free(reducted_GM);



	cudaFree(d_A);	
	free(A);
}