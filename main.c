#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<time.h>


#define numThreads 100000000
#define arraySize 100000000
#define blockSize 100000000


void printArray(int* array, long int size)
{
	for (long int i = 0; i < size; i++)
		printf("%d ", *(array + i));
}

__global__ void incrementNaive(int *g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	i = i%arraySize;
	g[i] = g[i] + 1;
}

int main()
{
	clock_t start, end;
	printf("%d total threads in %d blocks writing into %d array elements\n", numThreads, numThreads / blockSize, arraySize);
	int * h_array;
	h_array = (int *) malloc(sizeof(int) * arraySize);
	const size_t arrayBytes = arraySize * sizeof(int);
	int *d_array;
	printf("Max int is %d and size of one is %d \n", INT_MAX, sizeof(int));
	printf("Using size : %ld B or %ld KB or %ld MB or %ld GB \n", arrayBytes,
																arrayBytes >> 10,
																arrayBytes >> 20,
																arrayBytes >> 30
																);
	cudaMalloc((void**)&d_array, arrayBytes);
	cudaMemset((void*)d_array, 0, arrayBytes);
	start = clock();
	incrementNaive <<<blockSize / numThreads, numThreads>>> (d_array);
	cudaDeviceSynchronize();
	end = clock();
	cudaMemcpy(h_array, d_array, arrayBytes, cudaMemcpyDeviceToHost);
	//printArray(h_array, arraySize);
	printf("\nTime elapsed= %6.3lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	cudaFree(d_array);
	return 0;
}
