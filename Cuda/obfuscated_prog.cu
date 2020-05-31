#include <algorithm> 
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>


using namespace std;
using namespace std::chrono; 

#define ll long long

const unsigned int data_rows = 1000;
const unsigned int data_cols = 10;
const unsigned ll e = 963443092119039113;
const unsigned ll d = 920403722748280569;
const unsigned ll n = 2108958572404460311;



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


// Computes (a * b) % n
__host__ ll modmult(ll a, ll b, ll n)
{
	ll res = 0;
	a = a % n;
	
	while (b > 0) {

		if (b % 2 == 1)
			res = (res + a) % n;

		b = b >> 1;
		a = (a * 2) % n;
	}

	return res;
}


// Computes (msg ** exponent) % n
__host__ ll modexp(ll msg, ll exponent, ll n)
{
	ll res = 1; // Initialize result
	msg = msg % n; // Update msg if it is more than or equal to n
	while (exponent > 0) {
 		// If exponent is odd, multiply x with result
 		if (exponent % 2 == 1)
 			res = modmult(res, msg, n);

 		// exponent must be even now
 		exponent = exponent >> 1; // exponent = exponent/2
 		msg = modmult(msg, msg, n); // compute (msg^2) % n
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

// Computes (a * b) % n
__device__ ll modmult_dev(ll a, ll b, ll n)
{
	ll res = 0;
	a = a % n;
	
	while (b > 0) {

		if (b % 2 == 1)
			res = (res + a) % n;

		b = b >> 1;
		a = (a * 2) % n;
	}
	return res;
}


// Computes (msg ** exponent) % n
__device__ ll modexp_dev(ll msg, ll exponent, ll n)
{
	ll res = 1; // Initialize result
	msg = msg % n; // Update msg if it is more than or equal to n
	while (exponent > 0) {
 		// If exponent is odd, multiply x with result
 		if (exponent % 2 == 1)
 			res = modmult_dev(res, msg, n);

 		// exponent must be even now
 		exponent = exponent >> 1; // exponent = exponent/2
 		msg = modmult_dev(msg, msg, n); // compute (msg^2) % n
 	}
	return res;
}


// Computes (a * b) % n
// __device__ ll modmult_dev(ll a, ll b, ll n)
// {
// 	double res = 0.0;
// 	double a_d = (double) a;
// 	double b_d = (double) b;
// 	double n_d = (double) n;
// 
// 	a_d = fmod(a_d, n_d);
// 	
// 	while (b_d > 0) {
// 
// 		if ((int)fmod(b_d, 2.0) == 1)
// 			res = fmod((res + a_d), n_d);
// 
// 		b_d = b_d / 2;
// 		a_d = fmod((a_d * 2), n_d);
// 	}
// 	return (ll) res;
// }
// 
// 
// // Computes (msg ** exponent) % n
// __device__ ll modexp_dev(ll msg, ll exponent, ll n)
// {
// 	ll res = 1; // Initialize result
// 	msg = msg % n; // Update msg if it is more than or equal to n
// 	while (exponent > 0) {
//  		// If exponent is odd, multiply x with result
//  		if (exponent % 2 == 1)
//  			res = modmult_dev(res, msg, n);
// 
//  		// exponent must be even now
//  		exponent = exponent >> 1; // exponent = exponent/2
//  		msg = modmult_dev(msg, msg, n); // compute (msg^2) % n
//  	}
// 	return res;
// }



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


__host__ void Test_Decypt()
{

	//////////////
	// Test CPU //
	//////////////

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
    auto start_time = high_resolution_clock::now(); 
	cpu_decrypt(cipher_host, data_host);
	printf("\nData after decryption on CPU \n");
	print_data(data_host);
    auto stop_time = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop_time - start_time); 

	cout << "\n\n";
	cout << "CPU Runtime: " << duration.count() << " ms" << endl; 


	//////////////
	// Test GPU //
	//////////////


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
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	cudaMemcpy(data_host, data_gpu, data_rows * data_cols * sizeof(ll),
			cudaMemcpyDeviceToHost);

	printf("\nData after decryption on GPU\n");
	print_data(data_host);

	printf("\nGPU runtime: %.4f ms\n", gpu_time);

	printf("\nSpeedup: %.4f X\n", (float)(duration.count()) / gpu_time);

	
}
	
int main()
{
	Test_Decypt();
	return 0;
}
