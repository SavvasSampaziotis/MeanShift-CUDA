#include <stdio.h>
#include <stdlib.h>



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

	printf("\n[READ_DATASET]: ./data/mean_shift_output.bin N=%d\tD=%d\n",*N,*D);
	
	fclose(fp);
}

void print_dataset(int N, int D, float* data)
{
	int i,j;
	printf("\n--------------------\n");
	for(i=0; i<N; i++)
	{
		printf("%d ",i);
		for(j=0; j<D; j++)
		{
			 printf("%f\t", data[i*D+j]);
		}
		printf("\n");
	}
		

}