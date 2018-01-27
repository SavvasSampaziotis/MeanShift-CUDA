#include <stdio.h>
#include <stdlib.h>

#include "cuda_utilities.h"
#include "time_measure.h"
#include "array_utilities.h"

void compare_reduction(int N, int thread_num);
void test_frobenius();
void test_sumrow();

int main(int argc, char** argv)
{
	
	// compare_reduction(512	, 256);

	//test_frobenius();
	
}

void test_sumrow()
{
	int N = 7;
	int thread_num = 2;
	int blocks_num = N/thread_num;
	

	float* d_A; 
	float* A = (float*) malloc( N*N*sizeof(float) );
	cudaMalloc((void**) &d_A, N*N*sizeof(float) ); 
	for (int i = 0; i < N*N; ++i)
		A[i] = i;
	cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);

	
	float* d_reducted_SM;
	float* reducted_SM = (float*) malloc(N*blocks_num*sizeof(float));
	cudaMalloc((void**) &d_reducted_SM, N*blocks_num*sizeof(float) ); 
  	
	dim3 blockNum(blocks_num,N,1);
	dim3 threadNum(thread_num,1, 1);
	matrix_sum_row_SM <<<blockNum, threadNum, thread_num*sizeof(float)>>>(N, d_A, d_reducted_SM);		
	
	cudaMemcpy(reducted_SM, d_reducted_SM, N*blocks_num*sizeof(float), cudaMemcpyDeviceToHost);		
	
	print_dataset(N, blocks_num, reducted_SM);
	
	
	// Reduct the final reduction vector!
	float* d_sumR;
	float* sumR = (float*) malloc(N*sizeof(float));  
	cudaMalloc((void**) &d_sumR, N*sizeof(float));

	dim3 blockNum2(1,N,1);
	dim3 threadNum2(blocks_num,1, 1);

	matrix_sum_row_SM<<<blockNum2, 4, 4*sizeof(float)>>>(blocks_num, d_reducted_SM, d_sumR);
	cudaMemcpy(sumR, d_sumR, N*sizeof(float), cudaMemcpyDeviceToHost);	

	
	print_dataset(N,N,A);
	print_dataset(N, 1, sumR);

	cudaFree(d_reducted_SM);
	cudaFree(d_sumR);	
	cudaFree(d_A);	
	free(reducted_SM);
	free(A);
	free(sumR);
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
	
	// Reduct the final reduction vector!
	float sumR;  
	float* d_sumR;
	cudaMalloc((void**) &d_sumR, sizeof(float));
	reduction_SM <<<1, blocks_num, 
		 thread_num*sizeof(float)>>>(blocks_num, d_reducted_SM, d_sumR);		
	
	cudaMemcpy(&sumR, d_sumR, 1*sizeof(float), cudaMemcpyDeviceToHost);	
	

	if (sumR-result != 0)
		printf("[ReductionSM] Test Failed: %f \tCalc-time: %lf sec\n", result-sumR, calcTime );		
	else
		printf("[ReductionSM] Test Passed!\tCalc-time: %lf sec\n", calcTime );		
	
	cudaFree(d_reducted_SM);
	cudaFree(d_sumR);	
	free(reducted_SM);

	/*********************************************/

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



	cudaFree(d_A);	
	free(A);
}