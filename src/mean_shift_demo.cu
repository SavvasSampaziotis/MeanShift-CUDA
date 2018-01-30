#include <stdio.h>
#include <stdlib.h>

#include "time_measure.h"

#include "array_utilities.h"
#include "cuda_meanshift.h"
#include "cuda_reduction.h"


void writeDEBUG(float* d_A, int N, int D)
{
	float *temp = (float*) malloc(N*D*sizeof(float));
	cudaMemcpy(temp, d_A, N*D*sizeof(float), cudaMemcpyDeviceToHost); 
	write_meanshift_result(N,D,temp);
}

int main(int argc, char** argv)
{
	bool FAST_MEANSHIFT_FROP = 0;
	
	
	printf("To enable FAST_MEANSHIFT_FROB run with extra arg 1\n");

	if(argc == 2)
	{
		FAST_MEANSHIFT_FROP = atoi(argv[1]);
		printf("FAST_MEANSHIFT_FROB = %d\n", FAST_MEANSHIFT_FROP);
	}

/*Local Host Pointers*/
	int N,	D;
	float* X; // Original  Datapoints  
	float *Y; // Final mean-shifted Datapoints  

/* CUDA-Mememory Pointers*/
  	float* d_x; // Same as X and Y, but in CUDA-memory
  	
  	float *d_y_new, *d_y; // CUDA ptrs for 2 sets of Y. 
  	// These are used alternatively for the efficient calculation of m=y_new-y_prev
  	
	float* d_KernelMatrix; // The whole kernel matrix
	float* d_KernelSum; // The sumRow of the kernel matrix [N,1]
	float* d_KernelMatrixTEMP;
	float* d_KernelX; 	// The dot product of KernelSum and X [N,D]
	float* d_meanshift; // The meanshift of each iteration


/* Read Input-dataset and allocate Host memory  */
	read_dataset(&N, &D, &X);
	int L = N*D;

	Y = (float*) malloc(L*sizeof(float));

/*Memory Allocation in Cuda-Device */	
	cudaMalloc((void**) &d_x, L*sizeof(float)); 
	cudaMalloc((void**) &d_y, L*sizeof(float));
	cudaMalloc((void**) &d_y_new, L*sizeof(float));

	cudaMalloc((void**) &d_KernelMatrix, N*N*sizeof(float)); 
	cudaMalloc((void**) &d_KernelMatrixTEMP, N*N*sizeof(float)); 
	cudaMalloc((void**) &d_KernelX, L*sizeof(float)); 
	cudaMalloc((void**) &d_meanshift, L*sizeof(float)); 

/* Initiater Meanshift Algorithm: Copy data to device memory*/
	// d_x:=X
	cudaMemcpy(d_x, X, L*sizeof(float), cudaMemcpyHostToDevice);
	// d_y:=X  Initial Conditions of the algorithm 
	cudaMemcpy(d_y, X, L*sizeof(float), cudaMemcpyHostToDevice); 

/* Init Reduction-objects*/
	ReductionCache kernelSumRC;
	init_reduction_cache(N, N, 4, &kernelSumRC); //Shared memory wont fit any more data :(
	
	ReductionCache frobeniusRC;
	init_reduction_cache(L, 1, 256, &frobeniusRC);


/* Mean Shift Start!*/
    
  	TimeInterval timeInterval;
  	double seqTime;

  	int i=0;
  	float sigma=1;
  	float m_frob=1;
  	
  	tic(&timeInterval);
  	while(m_frob > 10e-8) // (10e-04)^2
  	{  		
  		i++;
  		if(i>50) break;

  		dim3 blockDim2D(4,4,1); 
  		dim3 gridDim2D(N/blockDim2D.x, N/blockDim2D.y,1); 
	  	calc_Kernel_Matrix<<<gridDim2D, blockDim2D>>>(N, D, d_x, d_y, d_KernelMatrix, sigma);

	  	// WR_kernelX_dot_product<<<kernelXproductRC>>>(N, D, 0, d_KernelMatrix, d_x, reducted_vec);
		for(int d=0; d<D; d++)
		{
			// MAtrix multiplication done in D-steps. On x-column per step.
			kernel_Dvec_mult<<<gridDim2D, blockDim2D>>>\
				(N, D, d_KernelMatrix, d_x, d_KernelMatrixTEMP, d);
			WR_reduction(N, d_KernelMatrixTEMP, &kernelSumRC);

			// Progressively reshapes the K*x array
			copy_to_y<<<N/100,100>>>(D, d_y_new, kernelSumRC.d_sum, d); 
		}
		
		WR_reduction(N, d_KernelMatrix, &kernelSumRC);
		d_KernelSum = kernelSumRC.d_sum;
		kernel_sum_div<<<N/100,100>>>(D, d_y_new,  d_KernelSum);

	/* Calc Frobenius Norm: sum(sum(d_meanshift.^2))*/
		if(!FAST_MEANSHIFT_FROP)
		{
			calc_meanshift2<<< L/100, 100>>>(d_y_new, d_y, d_meanshift);
			WR_reduction(L, d_meanshift, &frobeniusRC);
		}
		else
		{
			/* ALTERNATIVE APPROACH */
			calc_reduce_meanshift<<<L/256, 256, 256*sizeof(float)>>>(L, d_y_new, d_y, d_meanshift);
			reduction_sum<<<1,256,256*sizeof(float)>>>(L/256, d_meanshift, frobeniusRC.d_sum);
		}
		
	
		/* THis isnt as bad as you think. It contributes very little 
		to performance. Check for yourself bu uncommenting next if-statement */
		// if((i%6)==0) //Check only every 6th iteration... 
		cudaMemcpy(&m_frob, frobeniusRC.d_sum, 1*sizeof(float), cudaMemcpyDeviceToHost);  
		
	
		/* Switch pointers, so that there wont be any need for memcpy and stuff.
		 Really efficient! */
		float* temp = d_y;
		d_y = d_y_new;  
		d_y_new = temp;

  	}
  	seqTime = toc(&timeInterval);

  	printf("Number of iterations = %d\n", i);
	printf("Final Frobenius Error =  %f *10e-04\n", m_frob*10e+4);
	printf("Calc Time = %f\n", seqTime);
	

	cudaMemcpy(Y, d_y_new, L*sizeof(float), cudaMemcpyDeviceToHost); 
	write_meanshift_result(N,D,Y);


	delete_reduction_cache(&kernelSumRC);
	delete_reduction_cache(&frobeniusRC);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_y_new);
	cudaFree(d_KernelMatrix);	
	cudaFree(d_KernelMatrixTEMP);	
	//cudaFree(d_KernelSum); 
	cudaFree(d_KernelX); 
	cudaFree(d_meanshift); 
	
	free(X);
	free(Y);
}

