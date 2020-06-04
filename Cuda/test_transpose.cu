#include <algorithm> 
#include <iterator>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>


using namespace std;
using namespace std::chrono; 

#define ll long long

const int BLOCK_DIM = 8;


// Prints first 20 rows of the matrix
__host__ void print_data(unsigned ll* matrix, const int numrows, const int numcols)
{
    for (int i = 0; i < numrows;  ++i) {
        for (int j = 0; j < numcols; ++j) {
            std::cout << matrix[i * numcols + j] << ' ';
        }
        std::cout << std::endl;
    }
}

__global__ void transposeCoalesced(const unsigned ll* A, unsigned ll* AT, const int numrows, const int numcols)
{
	__shared__ unsigned ll tile[BLOCK_DIM][BLOCK_DIM]; // add plus one to avoid bank conflict
	int j = blockIdx.x * blockDim.x+ threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if(i <= numcols && j <= numrows) {

		tile[threadIdx.y][threadIdx.x]=A[i*numcols+j];
		__syncthreads();

		////block (by,bx) in AT
		int tj=blockIdx.y*blockDim.x+threadIdx.x;	////x for column
		int ti=blockIdx.x*blockDim.y+threadIdx.y;	////y for row
		AT[ti*numrows+tj] = tile[threadIdx.x][threadIdx.y];
	
	}
}


__host__ void Test_Transpose()
{

	// Initialize data matrix
	const int numcols = 8;
	const int numrows = 16;

	unsigned ll* data_host = new unsigned ll[numrows * numcols];
	for (unsigned int i = 0; i < numrows; i++){
		for (unsigned int j = 0; j < numcols; j++) {
			// simply make value equal to 1D index + 1 
			data_host[i * numcols + j] = i * numcols + j + 1;
		}
	}

	// Test Transpose Function:
	printf("Data before transpose\n");
	print_data(data_host, numrows, numcols);

	// **** CPU TRANSPOSE **** //
	unsigned ll *transpose_host = new unsigned ll[numcols * numrows];
	for (int i = 0; i < numrows; i++) {
		for (int j = 0; j < numcols; j++) {
			transpose_host[j * numrows + i] = data_host[i * numcols + j];
		}
	}

	printf("\nData after CPU transpose\n");
	print_data(transpose_host, numcols, numrows);


	// **** GPU TRANSPOSE **** //
	memset(transpose_host, 0x00, numrows * numcols * sizeof(unsigned ll));

	unsigned ll* data_gpu;
	cudaMalloc((void**)&data_gpu, numrows * numcols * sizeof(unsigned ll*));
	cudaMemcpy(data_gpu, data_host, numrows * numcols * sizeof(unsigned ll),
			cudaMemcpyHostToDevice);

	unsigned ll *transpose_gpu; 
	cudaMalloc((void**)&transpose_gpu, numrows * numcols * sizeof(unsigned ll*));

	const int block_size = BLOCK_DIM;
	const int block_num_x = numcols / block_size; // will always be 1 since numcols = 1
	const int block_num_y= ceil((double) numrows / (double) block_size);

	transposeCoalesced<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
	(data_gpu, transpose_gpu, numrows, numcols);

	cudaMemcpy(transpose_host, transpose_gpu, numrows * numcols * sizeof(unsigned ll), cudaMemcpyDeviceToHost);

	printf("\nData after GPU transpose\n");
	print_data(transpose_host, numcols, numrows);

}

int main()
{
	Test_Transpose();
	return 0;
}
