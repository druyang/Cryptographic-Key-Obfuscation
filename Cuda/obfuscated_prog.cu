#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define ll long long

const unsigned int data_rows = 1000;
const unsigned int data_cols = 10;

const unsigned ll e = 287387641;
const unsigned ll d = 512204809;
const unsigned ll n = 522213911;

// Prints first 20 rows of the matrix
__host__ void print_data(ll* matrix)
{
    for (int i = 0; i < 10;  ++i) {
        for (int j = 0; j < data_cols; ++j) {
            std::cout << matrix[i * data_cols + j] << ' ';
        }
        std::cout << std::endl;
    }
}


__host__ ll modexp(ll msg, unsigned ll exponent, unsigned ll n)
{
	ll res = 1; // Initialize result
	msg = msg % n; // Update msg if it is more than or
	// equal to n
	while (exponent > 0)
 	{
 		// If exponent is odd, multiply x with result
 		if (exponent % 2 == 1)
 			res = (res * msg) % n;
 		// exponent must be even now
 		exponent = exponent >> 1; // exponent = exponent/2
 		msg = (msg * msg) % n;
 	}

 return res;
} 


__host__ void cpu_decrypt(ll* cipher, ll* data)
{
	for (unsigned int i = 0; i < data_rows; i++){
		for (unsigned int j = 0; j < data_cols; j++) {
			// simply make value equal to 1D index
			data[i * data_cols + j] = modexp(cipher[i * data_cols +j], d, n);
		}
	}
}


__host__ void cpu_encrypt(ll* data, ll* cipher)
{
	for (unsigned int i = 0; i < data_rows; i++){
		for (unsigned int j = 0; j < data_cols; j++) {
			// simply make value equal to 1D index
			cipher[i * data_cols +j] = modexp(data[i * data_cols +j], e, n);
		}
	}
}

__device__ ll modexp_dev(ll msg, unsigned ll exponent, unsigned ll n)
{
	ll res = 1; // Initialize result
	msg = msg % n; // Update msg if it is more than or
	// equal to n
	while (exponent > 0)
 	{
 		// If exponent is odd, multiply x with result
 		if (exponent % 2 == 1)
 			res = (res * msg) % n;
 		// exponent must be even now
 		exponent = exponent >> 1; // exponent = exponent/2
 		msg = (msg * msg) % n;
 	}

 return res;
}


__global__ void gpu_decrypt(ll* cipher, ll* data)
{

	int rows_per_block = blockDim.y;
	int global_row = blockIdx.x * rows_per_block + threadIdx.y;
	int local_row = threadIdx.y; 
	int col = threadIdx.x;

	// If thread is in the bounds of the data array
	if (global_row < data_rows && col < data_cols) {
		data[global_row * data_cols + col] = modexp_dev(cipher[global_row * data_cols + col], d, n);
	}
}




__host__ void Test_GPU_Decypt()
{

	// Initialize data matricies
	ll* cipher_host = new ll[data_rows * data_cols];

	ll* data_host = new ll[data_rows * data_cols];


	for (unsigned int i = 0; i < data_rows; i++){
		for (unsigned int j = 0; j < data_cols; j++) {
			// simply make value equal to 1D index + 1 
			data_host[i * data_cols + j] = i * data_rows + j + 1;
		}
	}


	// Test Encrypt and Decrypt Functions:
	printf("Data before encryption/decryption\n");
	print_data(data_host);

	// Encrypt CPU
	cpu_encrypt(data_host, cipher_host);
	printf("\nCipher (result of encryption)\n");
	print_data(cipher_host);

	// Decrypt CPU
	cpu_decrypt(cipher_host, data_host);
	printf("\nData after decryption on CPU \n");
	print_data(data_host);

	// Reset data_host to zero before passing to GPU:
	memset(data_host, 0x00, data_rows * data_cols * sizeof(ll));

	// Copy cipher and blank data matrix to GPU;
	ll* cipher_gpu;
	ll* data_gpu;

	cudaMalloc((void**)&cipher_gpu, data_rows * data_cols * sizeof(ll*));
	cudaMemcpy(cipher_gpu, cipher_host, data_rows * data_cols * sizeof(ll),
			cudaMemcpyHostToDevice);

	cudaMalloc((void**)&data_gpu, data_rows * data_cols * sizeof(ll*));
	cudaMemcpy(data_gpu, data_host, data_rows * data_cols * sizeof(ll),
			cudaMemcpyHostToDevice);

	// Set # of threads and blocks:
	int blockdim_x = data_cols;  // columns
	int blockdim_y = 128 / data_cols;  // rows
	int num_blocks = ceil((double) data_rows / (double) blockdim_y);


	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);
	
	// TODO: Call Kernel to Decrypt Data:
	gpu_decrypt<<<num_blocks, dim3(blockdim_x, blockdim_y)>>>(cipher_gpu, data_gpu);


	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	cudaMemcpy(data_host, data_gpu, data_rows * data_cols * sizeof(ll),
			cudaMemcpyDeviceToHost);

	printf("\nData after decryption on GPU\n");
	print_data(data_host);



	
}
	
int main()
{
	Test_GPU_Decypt();
	return 0;
}
