/*

	Header file for cuda_wrapper.c implementation file  
*/

#ifndef CUDA_WRAPPER_H
#define  CUDA_WRAPPER_H

/*	
	NOT WOKRING

	This launches the CUDA kernel functions for the simple reduction 
		without the global memory access. 
*/
void WR_reduction_GM(int blocks_num, int threads_num, int N, float* x, float* reduction_cache );

#endif
