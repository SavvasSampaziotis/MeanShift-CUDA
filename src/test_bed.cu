#include <stdio.h>
#include <stdlib.h>

#include "cuda_wrappers.h"
#include "cuda_wrappers.h"
#include "time_measure.h"
#include "array_utilities.h"

void compare_reduction(int thread_num);
void test_frobenius();
void test_sumrow(int thread_num);

/*Some global variables*/
int N;
float* d_A; 
float* A;
int result;

int main(int argc, char** argv)
{
	N = 7*7;

	A = (float*) malloc( N*sizeof(float) );
	cudaMalloc((void**) &d_A, N*sizeof(float) ); 
	int result = 0;
	for (int i = 0; i < N; ++i)
	{
		A[i] = i;
		result += i; // Correct Result for testing purposes 
	}
	cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);


	//compare_reduction(256);
	N = 7;
	test_sumrow(2);



	cudaFree(d_A);	
	free(A);
}

void test_sumrow(int thread_num)
{
	N = 7;
	thread_num = 2;
	ReductionCache reductionCache;

	float* sumR = (float*) malloc(N*sizeof(float)); 	
	float* reduced_vec = (float*) malloc(N*(N/thread_num)*sizeof(float)); 	

	init_reduction_cache2D(N, thread_num, &reductionCache);
  	
	
	WR_matrix_row_reduction(N, d_A, &reductionCache);	


	cudaMemcpy(sumR, reductionCache.d_sum, N*sizeof(float), cudaMemcpyDeviceToHost);	
	
	cudaMemcpy(reduced_vec, reductionCache.d_reduced_vec, \
		N*(reductionCache.blockDim.x)*sizeof(float), cudaMemcpyDeviceToHost);	


	delete_reduction_cache(&reductionCache);

	
	// float* reducted_SM = (float*) malloc(N*blocks_num*sizeof(float));


	print_dataset(N,N,A);
	print_dataset(N, reductionCache.gridDim.x, reduced_vec);
	print_dataset(N, 1, sumR);
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
	double calcTime;
	TimeInterval timeInterval;
	
	int ITERATIONS = 100;
	
	ReductionCache reductionCache;

	// Test Data preperation
	

	/*********************************************/
	float sumR;
	init_reduction_cache_1D(N, thread_num, &reductionCache);
  	tic(&timeInterval);
	{
		for(int i=0; i<ITERATIONS; i++)
		WR_vector_reduction(N, d_A, &reductionCache);	
	}
	toc(&timeInterval);
	calcTime = timeInterval.seqTime/ITERATIONS; // Get average calc time

	cudaMemcpy(&sumR, reductionCache.d_sum, 1*sizeof(float), cudaMemcpyDeviceToHost);	
	delete_reduction_cache(&reductionCache);
	
	/*********************************************/
	if (sumR-result != 0)
		printf("[ReductionSM] Test Failed: %f \tCalc-time: %lf sec\n", result-sumR, calcTime );		
	else
		printf("[ReductionSM] Test Passed!\tCalc-time: %lf sec\n", calcTime );		

	/*********************************************/

	// float sumR;
	// init_reduction_cache(N, 64, &reductionCache);
 //  	tic(&timeInterval);
	// {
	// 	for(int i=0; i<ITERATIONS; i++)
	// 	WR_vector_reduction(N, d_A, &reductionCache);	
	// }
	// toc(&timeInterval);
	// calcTime = timeInterval.seqTime/ITERATIONS; // Get average calc time

	// cudaMemcpy(&sumR, reductionCache.d_sum, 1*sizeof(float), cudaMemcpyDeviceToHost);	
	// delete_reduction_cache(&reductionCache);


	// Run Reduction WITHOUT Shared memory 100 times and measure time...  
	// float * reducted_GM = (float*) malloc(thread_num*blocks_num*sizeof(float));
	// float * d_reducted_GM;
	
	// cudaMalloc((void**) &d_reducted_GM, \
	// 	thread_num*blocks_num*sizeof(float) );

 //  	tic(&timeInterval);
	// 	for(int i=0; i<ITERATIONS; i++)
	// 	{
	// 		// THis is a wrapper of the stuff that needs to be launched...
	// 		reduction_GM(blocks_num, thread_num, N, d_A, d_reducted_GM);	
	// 	}
	// toc(&timeInterval);
	// calcTime = timeInterval.seqTime/ITERATIONS; // Get average calc time

	// float sumR2;  
	// float* d_sumR2;
	// cudaMalloc((void**) &d_sumR2, sizeof(float));
	// reduction_SM <<<1, blocks_num, 
	// 	 thread_num*sizeof(float)>>>(blocks_num, d_reducted_GM, d_sumR2);		
	
	// cudaMemcpy(&sumR2, d_sumR2, 1*sizeof(float), cudaMemcpyDeviceToHost);	

	
	// if (sumR2-result != 0)
	// 	printf("[ReductionGM] Test Failed: %f \tCalc-time: %lf sec\n", result-sumR2, calcTime );		
	// else
	// 	printf("[ReductionGM] Test Passed!\tCalc-time: %lf sec\n", calcTime );		
	
	// cudaFree(d_reducted_GM);
	// cudaFree(d_sumR2);	
	// free(reducted_GM);



}