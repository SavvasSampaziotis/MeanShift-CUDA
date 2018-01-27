/*
	Header File for my utilites library


*/
#ifndef ARRAY_UTILITIES_H
#define  ARRAY_UTILITIES_H

/*
	Reads binary file of floats.  

	The File format is this: [HEADER][DATA....]
		HEADER = N,D // N=number of datapoints, D=dimensionality of datapoints
		DATA=[x(1,1)x(1,2)...x(1,D)...x(N,1)x(N,2)...x(N,D)]

		NO DELIMITERS of any kind. 
*/
void read_dataset(int* N, int* D, float** data);

/*
	Creates and writes binary file of floats.  

	The File format is this: [HEADER][DATA....]
		HEADER = N,D // N=number of datapoints, D=dimensionality of datapoints
		DATA=[x(1,1)x(1,2)...x(1,D)...x(N,1)x(N,2)...x(N,D)]

		NO DELIMITERS of any kind. 
*/
void write_meanshift_result(int N, int D, float* data);

/*
	Prints array 1D of N*D length on screen.

	The dimensions of the printed array are NxD.  
*/
void print_dataset(int N, int D, float* data);

#endif