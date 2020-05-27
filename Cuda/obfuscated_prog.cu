#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

const unsigned int data_rows = 1000;
const unsigned int data_cols = 20;
const long e = 1783517723733006701;
const long d = 1672094126469673101;
const long n = 2078627964087837859;

__host__ void print_data(long** matrix)
{
    for (int i = 0; i < data_rows;  ++i) {
        for (int j = 0; j < data_cols; ++j) {
            std::cout << matrix[i][j] << ' ';
        }
        std::cout << std::endl;
    }
}


__host__ int modexp(long msg, long exponent, long n)
{
	int res = 1; // Initialize result
	msg = msg % n; // Update x if it is more than or
	// equal to n
	while (exponent > 0)
 	{
 	// If exponent is odd, multiply x with result
 	if (exponent & 1)
 	res = (res*msg) % n;
 	// exponent must be even now
 	exponent = exponent>>1; // exponent = exponent/2
 	x = (msg*msg) % n;
 	}

 return res;
} 


__host__ void cpu_decrypt(const long** cipher, long** data)
{
	for (unsigned int i = 0; i < data_rows; i++){
		for (unsigned int j = 0; j < data_cols; j++) {
			// simply make value equal to 1D index
			data[i][j] = modexp(cipher[i][j], e, n);
		}
	}
}


__host__ void cpu_encrypt(const long** data, long** cipher)
{
	for (unsigned int i = 0; i < data_rows; i++){
		for (unsigned int j = 0; j < data_cols; j++) {
			// simply make value equal to 1D index
			cipher[i][j] = modexp(data[i][j], e, n);
		}
	}
}



// __global__ decrypt(long** data_host, 

__host__ void CPU_Encrypt_Decrypt()
{
	long ** cipher_host = new long[data_rows][data_cols];

	for (unsigned int i = 0; i < data_rows; i++){
		for (unsigned int j = 0; j < data_cols; j++) {
			// simply make value equal to 1D index
			cipher_host[i][j] = i * data_rows + j;
		}
	}
	print_data(cipher_host);

}



__host__ void Test_GPU_Decypt()
{
	long ** cipher_host = new long[data_rows][data_cols];

	for (unsigned int i = 0; i < data_rows; i++){
		for (unsigned int j = 0; j < data_cols; j++) {
			// simply make value equal to 1D index
			cipher_host[i][j] = i * data_rows + j 
		}
	}

	long** data_gpu;
	cudaMalloc((void**)&data_gpu, data_rows * data_cols * sizeof(long),
			cudaMemcpyHostToDevice);
	cudaMemcpy(data_gpu, data_host, data_rows * data_cols * sizeof(long),
			cudaMemcpyHostToDevice);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);
	
	// TODO: Call Kernel to Decrypt Data:


	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	cudaMemcpy(data_host, data_gpu, data_rows * data_cols * sizeof(long),
			cudaMemcpyDeviceToHost);


	
}
	
int main()
{
	Encrypt_Data();
	return 0;
}
