#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<time.h>

#define numThreads 1000000
#define arraySize 1000000
#define blockSize 1000000


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
	printf("%d total threads in %d blocks writing into %d array elements\n", numThreads, numThreads/blockSize, arraySize);
	int h_array[arraySize];
	const int arrayBytes = arraySize * sizeof(int);
	int *d_array;
	cudaMalloc((void**)&d_array, arrayBytes);
	cudaMemset((void*)d_array, 0, arrayBytes);
	start = clock();
	incrementNaive <<<numThreads / blockSize, blockSize >>> (d_array);
	end = clock();
	cudaMemcpy(h_array, d_array, arrayBytes, cudaMemcpyDeviceToHost);
	printArray(h_array,arraySize);
	printf("\nTime elapsed=%d\n", (double)end-start);
	cudaFree(d_array);
	return 0;
}
