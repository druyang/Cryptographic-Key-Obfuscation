#include <algorithm> 
#include <iterator>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include <boost/math/distributions/students_t.hpp>

// #include <thrust/device_vector.h>
// #include <thrust/reduce.h>
// #include <thrust/transform.h>
// #include <thrust/transform_reduce.h>
// #include <thrust/functional.h>
// #include <thrust/count.h>
// #include <thrust/execution_policy.h>


using namespace std;
using namespace std::chrono; 

#define ll long long

const unsigned int data_rows = 1000;
const unsigned int data_cols = 10;
const unsigned ll e = 963443092119039113;
const unsigned ll d = 920403722748280569;
const unsigned ll n = 2108958572404460311;
const int BLOCK_DIM = 8;


void Statistics_CPU(unsigned ll *indep, unsigned ll *dep, int numcols);
// void Statistics_GPU(unsigned ll *indep, unsigned ll *dep, int numcols);

void CPU_One_Sample_T_Interval(int32_t *dep, int numcols);
void CPU_Two_Sample_T_Test(int32_t *dep, int32_t *indep, int numcols);

// void GPU_One_Sample_T_Interval(thrust::device_vector<int> data, int numcols);
// void GPU_Two_Sample_T_Test(thrust::device_vector<int> data, thrust::device_vector<int> categories, int numcols);


__host__ void print_1D(unsigned ll* data, int length) {
	for (int i = 0; i < min(length, 20); i++) {
		std::cout << data[i] << ' ';
	}
	std::cout << std::endl;
}

// Prints first 20 rows of the matrix
__host__ void print_data(unsigned ll* matrix)
{
    for (int i = 0; i < 10;  ++i) {
        for (int j = 0; j < data_cols; ++j) {
            std::cout << matrix[i * data_cols + j] << ' ';
        }
        std::cout << std::endl;
    }
}

// determines whether or not overflow will occur
__host__ bool is_overflow(unsigned ll a, unsigned ll b) 
{ 
    // Check if either of them is zero 
    if (a == 0 || b == 0)  
        return false; 

    unsigned long long result = a * b; 
    if (a == result / b) 
        return false; 
    else
        return true; 
} 


// Computes (a * b) % n
__host__ unsigned ll modmult(unsigned ll a, unsigned ll b, unsigned ll n)
{
	unsigned ll res = 0;
	a = a % n;

	// if a * b mod n can be computed direclty, compute it right away.
	if (!(is_overflow(a, b + 1))) return (a * b) % n;
	
	while (b > 0) {

		if (b % 2 == 1)
			res = (res + a) % n;
		
		b = b >> 1;
		a = (a * 2) % n;
	}

	return res;
}


// Computes (msg ** exponent) % n
__host__ ll modexp(unsigned ll msg, unsigned ll exponent, unsigned ll n)
{
	unsigned ll res = 1; // Initialize result
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


__host__ void cpu_decrypt(unsigned ll* cipher, unsigned ll* data, int length)
{
	for (unsigned int i = 0; i < length; i++) {
		data[i] = modexp(cipher[i], d, n);
	}
}

// __host__ void cpu_decrypt(ll* cipher, ll* data)
// {
// 	for (unsigned int i = 0; i < data_rows; i++){
// 		for (unsigned int j = 0; j < data_cols; j++) {
// 			// simply make value equal to 1D index
// 			data[i * data_cols + j] = modexp(cipher[i * data_cols +j], d, n);
// 		}
// 	}
// }


__host__ void cpu_encrypt(unsigned ll* data, unsigned ll* cipher)
{
	for (unsigned int i = 0; i < data_rows; i++){
		for (unsigned int j = 0; j < data_cols; j++) {
			// simply make value equal to 1D index
			cipher[i * data_cols +j] = modexp(data[i * data_cols +j], e, n);
		}
	}
}

// determines whether or not overflow will occur
__device__ bool is_overflow_dev(unsigned ll a, unsigned ll b) 
{ 
    // Check if either of them is zero 
    if (a == 0 || b == 0)  
        return false; 
    
    ll result = a * b; 
    if (a == result / b) 
        return false; 
    else
        return true; 
} 


// Computes (a * b) % n
__device__ unsigned ll modmult_dev(unsigned ll a, unsigned ll b, unsigned ll n)
{
	unsigned ll res = 0;
	a = a % n;

	// if (!(is_overflow_dev(a, b + 1))) return (a * b) % n;
	
	while (b > 0) {

		if (b % 2 == 1)
			res = (res + a) % n;

		b = b >> 1;
		a = (a * 2) % n;
	}
	return res;
}


// Computes (msg ** exponent) % n
__device__ unsigned ll modexp_dev(unsigned ll msg, unsigned ll exponent, unsigned ll n)
{
	unsigned ll res = 1; // Initialize result
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


__global__ void gpu_decrypt(unsigned ll* cipher, unsigned ll* data)
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


unsigned ll* File_To_Array(const char *filename, int &length, int &fd)
{
    // open() gives the system a call to identify a file on disk
    // describe its location with an int, called a "file descriptor (fd)",
    // and give certain permissions to the program
    fd = open(filename, O_RDONLY, S_IWUSR | S_IRUSR);
    struct stat sb;

    // fstat() reads a file descriptor and tries to get its length in bytes as a long int
    if (fstat(fd, &sb) == -1) {
        perror("bad file");
        exit(-1);
    }

    // get length of the file in int32_t
    length = sb.st_size / 4;

    // mmap asks the OS to provision a chunk of disk storage out to contiguous (read aligned, coalesced) RAM
    // this is the reverse of using 'swap space' to cache some RAM out to disk when under memory pressure
    // we give it the fd file descriptor and the size of the file to tell the OS which chunk of disk to allocate as memory
    // and also give it certain permissions
    // this is a direct array of data so we can cast it to whatever form we like, in this case bytes
    // and then we can address the pointer as an array as we're familiar with
    return (unsigned ll *)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
}

__global__ void transposeCoalesced(const unsigned ll* A, unsigned ll* AT, const int numrows, const int numcols)
{
	__shared__ unsigned ll tile[BLOCK_DIM][BLOCK_DIM + 1]; // add plus one to avoid bank conflict
	int j = blockIdx.x * BLOCK_DIM + threadIdx.x;
	int i = blockIdx.y * BLOCK_DIM + threadIdx.y;

	if(i <= numrows && j <= numcols) {

		tile[threadIdx.y][threadIdx.x]=A[i*numcols+j];
		__syncthreads();

		////block (by,bx) in AT
		int tj=blockIdx.y*blockDim.x+threadIdx.x;	////x for column
		int ti=blockIdx.x*blockDim.y+threadIdx.y;	////y for row
		AT[ti*numrows+tj] = tile[threadIdx.x][threadIdx.y];
	
	}
}




// __host__ void Test_Decypt()
// {

// 	//////////////
// 	// Test CPU //
// 	//////////////

// 	// Initialize data matricies
// 	unsigned ll* cipher_host = new unsigned ll[data_rows * data_cols];
// 	unsigned ll* data_host = new unsigned ll[data_rows * data_cols];


// 	for (unsigned int i = 0; i < data_rows; i++){
// 		for (unsigned int j = 0; j < data_cols; j++) {
// 			// simply make value equal to 1D index + 1 
// 			data_host[i * data_cols + j] = i * data_rows + j + 1;
// 		}
// 	}


// 	// Test Encrypt and Decrypt Functions:
// 	printf("Data before encryption/decryption\n");
// 	print_data(data_host);

// 	// Encrypt CPU
// 	cpu_encrypt(data_host, cipher_host);
// 	printf("\nCipher (result of encryption)\n");
// 	print_data(cipher_host);

	
// 	// Decrypt CPU
//     auto start_time = high_resolution_clock::now();
// 	cpu_decrypt(cipher_host, data_host);
// 	printf("\nData after decryption on CPU \n");
// 	print_data(data_host);
//     auto stop_time = high_resolution_clock::now();
// 	auto duration = duration_cast<microseconds>(stop_time - start_time); 

// 	cout << "\n\n";
// 	cout << "CPU Runtime: " << duration.count() << " ms" << endl; 


// 	//////////////
// 	// Test GPU //
// 	//////////////


// 	// Reset data_host to zero before passing to GPU:
// 	memset(data_host, 0x00, data_rows * data_cols * sizeof(unsigned ll));

// 	// Copy cipher and blank data matrix to GPU;
// 	unsigned ll* cipher_gpu;
// 	unsigned ll* data_gpu;

// 	cudaMalloc((void**)&cipher_gpu, data_rows * data_cols * sizeof(unsigned ll*));
// 	cudaMemcpy(cipher_gpu, cipher_host, data_rows * data_cols * sizeof(unsigned ll),
// 			cudaMemcpyHostToDevice);

// 	cudaMalloc((void**)&data_gpu, data_rows * data_cols * sizeof(unsigned ll*));
// 	cudaMemcpy(data_gpu, data_host, data_rows * data_cols * sizeof(unsigned ll),
// 			cudaMemcpyHostToDevice);

// 	// Set # of threads and blocks:
// 	int blockdim_x = data_cols;  // columns
// 	int blockdim_y = 128 / data_cols;  // rows
// 	int num_blocks = ceil((double) data_rows / (double) blockdim_y);


// 	cudaEvent_t start,end;
// 	cudaEventCreate(&start);
// 	cudaEventCreate(&end);
// 	float gpu_time=0.0f;
// 	cudaDeviceSynchronize();
// 	cudaEventRecord(start);
	
// 	// TODO: Call Kernel to Decrypt Data:
// 	gpu_decrypt<<<num_blocks, dim3(blockdim_x, blockdim_y)>>>(cipher_gpu, data_gpu);


// 	cudaEventRecord(end);
// 	cudaEventSynchronize(end);
// 	cudaEventElapsedTime(&gpu_time,start,end);
// 	cudaEventDestroy(start);
// 	cudaEventDestroy(end);

// 	cudaMemcpy(data_host, data_gpu, data_rows * data_cols * sizeof(unsigned ll),
// 			cudaMemcpyDeviceToHost);

// 	printf("\nData after decryption on GPU\n");
// 	print_data(data_host);

// 	printf("\nGPU runtime: %.4f ms\n", gpu_time);

// 	printf("\nSpeedup: %.4f X\n", (float)(duration.count()) / gpu_time);

	
// }

__host__ void Test_Entire_CPU(char *dataname)
{

	//////////////
	// Test CPU //
	//////////////

	const int linewidth = 64 * 8;
	int fd;
	int datalength; // already changed to bytes/8
	
	// we need to pass in dataname as the filename into this
	unsigned ll *data = File_To_Array(dataname, datalength, fd);

	// calculate matrix size
	const int numcols = 8;
	int numrows = datalength / numcols;

	// **** CPU TRANSPOSE **** //
	unsigned ll *transpose = new unsigned ll[numcols][numrows];
	for (int i = 0; i < numcols; i++) {
		for (int j = 0; j < numrows; j++) {
			transpose[j][i] = data[i][j];
		}
	}

	// 0 = pid, 1 = male, 2 = female, 3 = other gender, 4 = age
	// 5 = deceased, 6 = released, 7 = in progress
	const int independent_col = 2; // female
	const int dependent_col = 5; // rate of deceased
	
	unsigned ll *indep_cipher = &transpose[2][0];
	unsigned ll *dep_cipher = &transpose[5][0];
    // TODO: FIX 2D vs. 1D Array 
	
	printf("\nCipher text from file:\n");
	print_1D(indep_cipher);
	print_1D(dep_cipher);
	
	// Decrypt CPU
	auto start_time = high_resolution_clock::now();
	
	unsigned ll *indep_decoded = new unsigned ll[numrows];
	unsigned ll *dep_decoded = new unsigned ll[numrows];
	
	cpu_decrypt(indep_cipher, indep_decoded, numrows);
	cpu_decrypt(dep_cipher, dep_decoded, numrows);
	printf("\nData after decryption on CPU \n");
	print_1D(indep_decoded);
	print_1D(dep_decoded);
    auto stop_time = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop_time - start_time);

	cout << "\n\n";
	cout << "CPU Decryption Runtime: " << duration.count() << " ms" << endl; 

	// this assumes we transpose. if we don't, we need different parameters
	Statistics_CPU(indep_decoded, dep_decoded, numrows);

	munmap(data, datalength);
	close(fd);
}

// __host__ void Test_Entire_GPU(char *dataname)
// {
// 	//////////////
// 	// Test GPU //
// 	//////////////

// 	const int linewidth = 64 * 8;
// 	int fd;
// 	int datalength; // already changed to bytes/8

// 	// we need to pass in dataname as the filename into this
// 	unsigned ll *data = File_To_Array(dataname, datalength, fd);

// 	// calculate matrix size
// 	const int numcols = 8;
// 	int numrows = datalength / numcols;

// **** GPU TRANSPOSE **** //
// unsigned ll* data_gpu;
// cudaMalloc((void**)&data_gpu, numrows * numcols * sizeof(unsigned ll));
// cudaMemcpy(data_gpu, data, data_rows * data_cols * sizeof(unsigned ll),
// 		cudaMemcpyHostToDevice);

// unsigned ll *transpose_gpu; 
// cudaMalloc((void**)&transpose_gpu, numrows * numcols * sizeof(unsigned ll));

// const int block_size = BLOCK_DIM;
// const int block_num_x = numcols / block_size; // will always be 1 since numcols = 1
// const int block_num_y= ceil((double) numrows / (double) block_size);

// transposeCoalesced<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
// (data_gpu, transpose_gpu, numrows, numcols);



// 	// BELOW CODE UNDONE


// 	// Reset data_host to zero before passing to GPU:
// 	memset(data_host, 0x00, data_rows * data_cols * sizeof(ll));

// 	// Copy cipher and blank data matrix to GPU;
// 	ll* cipher_gpu;
// 	ll* data_gpu;

// 	cudaMalloc((void**)&cipher_gpu, data_rows * data_cols * sizeof(ll*));
// 	cudaMemcpy(cipher_gpu, cipher_host, data_rows * data_cols * sizeof(ll),
// 			cudaMemcpyHostToDevice);

// 	cudaMalloc((void**)&data_gpu, data_rows * data_cols * sizeof(ll*));
// 	cudaMemcpy(data_gpu, data_host, data_rows * data_cols * sizeof(ll),
// 			cudaMemcpyHostToDevice);

// 	// Set # of threads and blocks:
// 	int blockdim_x = data_cols;  // columns
// 	int blockdim_y = 128 / data_cols;  // rows
// 	int num_blocks = ceil((double) data_rows / (double) blockdim_y);


// 	cudaEvent_t start,end;
// 	cudaEventCreate(&start);
// 	cudaEventCreate(&end);
// 	float gpu_time=0.0f;
// 	cudaDeviceSynchronize();
// 	cudaEventRecord(start);

// 	// TODO: Call Kernel to Decrypt Data:
// 	gpu_decrypt<<<num_blocks, dim3(blockdim_x, blockdim_y)>>>(cipher_gpu, data_gpu);


// 	cudaEventRecord(end);
// 	cudaEventSynchronize(end);
// 	cudaEventElapsedTime(&gpu_time,start,end);
// 	cudaEventDestroy(start);
// 	cudaEventDestroy(end);

// 	cudaMemcpy(data_host, data_gpu, data_rows * data_cols * sizeof(ll),
// 			cudaMemcpyDeviceToHost);

// 	printf("\nData after decryption on GPU\n");
// 	print_data(data_host);

// 	printf("\nGPU runtime: %.4f ms\n", gpu_time);

// 	printf("\nSpeedup: %.4f X\n", (float)(duration.count()) / gpu_time);

// 	munmap(data, datalength);
// 	close(fd);
// }

void Statistics_CPU(unsigned ll *indep, unsigned ll *dep, int numcols)
{
    CPU_One_Sample_T_Interval(dep, numcols);

    CPU_Two_Sample_T_Test(dep, indep, numcols);
}

void CPU_One_Sample_T_Interval(unsigned ll *data, int numcols)
{
    // timer
    auto cpu_start = system_clock::now();

    const double confidence = 0.95;

    // calculate mean
    double mean;
    double castdata;
    for (int i = 0; i < numcols; i++) {
        castdata = (double)data[i];
        // std::cout<<(double)castdata<<"\n";
        mean += castdata / numcols;
    }

    double stddev;
    double difference;
    // calculate std deviation
    for (int i = 0; i < numcols; i++) {
        castdata = (double)data[i];
        difference = castdata - mean;
        // std::cout<<difference<<"\n";
        stddev += difference * difference / (numcols-1);
    }
    stddev = sqrt(stddev);

    auto cpu_end = system_clock::now();
    duration<double> cpu_time=cpu_end-cpu_start;

    // standard error
    double stderror = stddev / sqrt(numcols);

    // calculate t-statistic
    students_t dist(numcols - 1);
    double t_statistic = quantile(dist, confidence/2+0.5);

    // calculate margin of error
    double moe = t_statistic * stderror;


    // print out statistics
    std::cout<<"\nOne-Sample T-Interval CPU results: \n";
    std::cout<<"Sample size: \t\t"<<numcols<<"\n";
    std::cout<<"Sample mean: \t\t"<<mean<<"\n";
    std::cout<<"Sample std dev: \t"<<stddev<<"\n";
    std::cout<<"Standard error: \t"<<stderror<"\n";
    std::cout<<"\nT-statistic for 95 percent confidence interval: \t"<<t_statistic<<"\n";
    std::cout<<"Margin of error for this sample: \t\t"<<moe<<"\n";
    std::cout<<"95 percent confident that the true population mean lies between "<<mean-moe<<" and "<<mean+moe<<"\n";

    std::cout<<"CPU runtime: "<<cpu_time.count()*1000.<<" ms."<<std::endl;
}

void CPU_Two_Sample_T_Test(unsigned ll *data, unsigned ll *categories, int numcols)
{
    auto cpu_start = system_clock::now();

    // level for statistical significance
    const double alpha = 0.05;

    // calculate mean
    double mean[2] = { 0, 0 };
    int length[2] = { 0, 0 };
    double castdata;
    int index;
    for (int i = 0; i < numcols; i++) {
        castdata = (double)data[i];
        // std::cout<<(double)castdata<<"\n";
        index = categories[i];
        length[index]++; 
        mean[index] += castdata;
    }
    mean[0] /= length[0];
    mean[1] /= length[1];

    double stddev[2] = { 0, 0 };
    double variance[2] = { 0, 0 };
    double difference;
    // calculate std deviation and variance
    for (int i = 0; i < numcols; i++) {
        castdata = (double)data[i];
        index = categories[i];
        difference = castdata - mean[index];
        // std::cout<<difference<<"\n";
        variance[index] += difference * difference / (length[index]-1);
    }
    stddev[0] = sqrt(variance[0]);
    stddev[1] = sqrt(variance[1]);

    auto cpu_end = system_clock::now();
    duration<double> cpu_time=cpu_end-cpu_start;

    //// calculate pooled deviation and degrees of freedom
    // double df = length[0] + length[1] - 2;
    // double stddev_pool = (length[0]-1)*variance[0] + (length[1]-1)*variance[1];
    // stddev_pool = sqrt(stddev_pool/df);

    // use welch's t-test for more sophistication under different variances
    // calculate degrees of freedom
    double t1 = variance[0] / length[0];
    double t2 = variance[1] / length[1];
    double df = t1 + t2;
    df *= df;
    t1 *= t1;
    t2 *= t2;
    t1 /= (length[0] - 1);
    t2 /= (length[1] - 1);
    df /= (t1 + t2); // finished

    // calculate standard error
    double stderror = sqrt(variance[0]/length[0] + variance[1]/length[1]);

    // calculate difference and t-statistic
    double diffmeans = mean[1] - mean[0];
    // this uses pooled
    // double t_statistic = diffmeans / (stddev_pool * sqrt(1.0 / length[0] + 1.0 / length[1]));
    // now for welch's test
    double t_statistic = diffmeans / stderror;

    students_t dist(df);
    double p_value = cdf(complement(dist, fabs(t_statistic)));

    

    // print out statistics
    std::cout<<"\nTwo-Sample Two-Tailed T-Test CPU results: \n";
    std::cout<<"Sample size[0]: \t"<<length[0]<<"\n";
    std::cout<<"Sample mean[0]: \t"<<mean[0]<<"\n";
    std::cout<<"Sample std dev[0]: \t"<<stddev[0]<<"\n";
    std::cout<<"Sample size[1]: \t"<<length[1]<<"\n";
    std::cout<<"Sample mean[1]: \t"<<mean[1]<<"\n";
    std::cout<<"Sample std dev[1]: \t"<<stddev[1]<<"\n\n";
    std::cout<<"Difference of means: \t\t"<<diffmeans<<"\n";
    std::cout<<"Degrees of freedom: \t\t"<<df<<"\n";
    std::cout<<"Standard error of difference: \t"<<stderror<"\n";
    std::cout<<"\nT-statistic for difference of means compared to null hypothesis: "<<t_statistic<<"\n";
    std::cout<<"Alpha value: \t\t"<<alpha<<"\n";
    std::cout<<"P-value: \t\t"<<p_value<<"\n";
    std::cout<<"We "<<(p_value >= alpha/2 ? "fail to" : "")<<" reject the null hypothesis.\n";

    std::cout<<"CPU runtime: "<<cpu_time.count()*1000.<<" ms."<<std::endl;
}


int main(int argc, char *argv[])
{
	// Test_Decypt();
	Test_Entire_CPU(argv[1]);
	// Test_Entire_GPU(argv[1]);
	return 0;
}
