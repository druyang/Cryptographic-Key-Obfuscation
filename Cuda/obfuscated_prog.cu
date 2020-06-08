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

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>


using namespace std;
using namespace std::chrono;
using boost::math::students_t;

#define ll long long

__device__  __constant__ unsigned ll m1[16*16] = {
    3898138, 4272236, 3818297, 3809756, 4037852, 4676915, 3500886, 3619362, 4730634, 2829970, 3517506, 4087421, 4739601, 3912997, 3681388, 6359441, 3255282, 6285463, 5859578, 6306639, 5707605, 6090034, 3791157, 3023551, 6494436, 5639060, 2442072, 4931038, 3374951, 2973521, 3534381, 2006977, 4512006, 2730539, 4739724, 3385241, 4656916, 3938940, 3796672, 4762700, 6592658, 4096400, 3402646, 4623696, 6895815, 5140537, 3059772, 6976185, 4090503, 3847150, 5493941, 4084221, 5289630, 3366036, 3877302, 6044678, 5833958, 3454838, 5369570, 4076362, 4112938, 3652335, 5586920, 4455035, 3530871, 5166626, 2851545, 6459783, 3200179, 3905475, 4882530, 6386745, 4939154, 6054883, 6988990, 2800608, 4300093, 3029774, 6468896, 5294418, 4392603, 4249913, 5528383, 3092627, 5421793, 3344070, 6070234, 3644930, 4301352, 4607155, 5854072, 5832719, 5571846, 5018476, 1457339, 3594239, 3309544, 2658855, 3726900, 5894573, 5658810, 3933981, 6213873, 4204623, 4381714, 6075984, 5200611, 3378648, 3182854, 5359960, 5160722, 4496497, 6592430, 5971095, 4102753, 3446501, 2707085, 4192901, 3942405, 4480361, 5734622, 6092668, 5815420, 2193260, 3654966, 4195351, 4400242, 4859107, 4152135, 1848942, 5410256, 5432025, 4207561, 3319602, 4141769, 4403163, 2130807, 5398166, 4417568, 4774938, 1706853, 4677230, 6013745, 3612519, 4617344, 3915945, 6005133, 4442264, 5757908, 5010119, 3258015, 3542214, 3510823, 3080679, 4174442, 5279308, 3338275, 1425392, 4225674, 4111864, 5243952, 4014867, 3083188, 3784711, 3580232, 4758474, 3637293, 2206083, 2646303, 2474987, 6277880, 1158152, 3658919, 5452372, 3882484, 4304057, 3165794, 2191558, 5989865, 5078279, 6020807, 4652723, 3458259, 5856963, 4341712, 5928672, 6142960, 5079986, 4026505, 7370147, 5035209, 4660278, 3923764, 2550655, 0, 4165580, 4629051, 3069243, 3734965, 2812772, 2957817, 5182954, 4006232, 4623017, 6091379, 4084730, 3843241, 4139640, 5351058, 5051956, 6396350, 3120492, 4636888, 3984535, 6898848, 3942165, 4713424, 3750126, 6147513, 5964568, 5260514, 5227714, 4282723, 4550280, 6269824, 5316692, 6081364, 5527953, 4178586, 3084830, 4026779, 5714687, 5225745, 5787677, 4253001, 4572657, 3353132, 5175311, 3802632, 6154279, 4099225, 4429291, 3657324, 5316210, 4680313, 3756697, 5949820, 5268209, 4685384, 4284995, 3783679, 3954066, 3515489, 4653612, 5879525, 5531160
};

__device__  __constant__ unsigned ll m2[16*16] = {
    5402623, 3995627, 6743696, 3542762, 4575592, 3695284, 3918622, 4538491, 6181070, 2512172, 4698104, 4163423, 4292282, 4453561, 1477165, 4966421, 4736401, 4810828, 5344335, 4837246, 4286233, 2634311, 4147100, 1921287, 6340262, 4113615, 3795215, 4211793, 2113035, 2556419, 2938045, 5122808, 4298089, 4512884, 3691197, 5534985, 4851922, 2772637, 2236804, 6120396, 2929315, 4143837, 3787446, 3239459, 4775456, 865982, 4496660, 1703666, 1599063, 4549671, 3501246, 1249567, 3323392, 4137217, 4195107, 6887026, 4291745, 5286819, 4357805, 2681316, 3140604, 4414659, 2713437, 4684680, 4623493, 2717830, 5068866, 5504463, 4723859, 6560104, 3436816, 5727689, 1648325, 4410464, 3260375, 3527822, 3801152, 5313457, 3446586, 5860645, 4737773, 5422197, 2731218, 4057691, 4550777, 5336448, 3092465, 2570986, 4040807, 3075376, 4867819, 4488883, 5773716, 3522895, 2814470, 5650927, 4377915, 4178781, 4884170, 3743092, 2828557, 4740730, 3232547, 4097158, 4680480, 5160081, 4033217, 2703093, 1601640, 5648239, 3977663, 5292342, 1858079, 5437452, 5778958, 3868468, 4060999, 3947463, 5609988, 4810874, 5726724, 4889048, 3297829, 4738813, 6375997, 3406592, 5284049, 4546058, 5626721, 4157315, 6594087, 6505334, 7104890, 3101450, 4471290, 4693939, 1906608, 6790147, 2749917, 4280580, 4847134, 5819494, 2243550, 2962854, 1061216, 3903053, 5066876, 4876405, 4163034, 4753225, 6430668, 4729571, 4429114, 5100039, 6968806, 4146280, 4409023, 4459757, 2614041, 5123350, 4132817, 4575138, 3217081, 4959027, 3843245, 6940258, 4267578, 3587797, 3931136, 5217963, 6225758, 2514754, 4230712, 5868142, 4260445, 4583537, 4107203, 1696201, 3402111, 4265050, 5518600, 5989181, 1619745, 3704475, 4369304, 2739916, 4914357, 3048664, 5007979, 4977233, 2376675, 3990140, 1889822, 3240595, 2616037, 2789849, 4489124, 4465373, 4793428, 6241130, 674170, 3887638, 3952595, 3276410, 3403888, 3426655, 5523518, 3812274, 5021758, 4478267, 3835431, 2826298, 3810009, 3389970, 4590227, 5646501, 698792, 4943485, 3322046, 3383689, 2234016, 2160407, 2022331, 5654802, 5180190, 3407774, 3774041, 4347928, 5845572, 2056892, 6877570, 2917990, 3144028, 2784578, 4687680, 4418303, 4558507, 5042387, 3523901, 4224349, 3338556, 5510521, 2140311, 3958657, 4013730, 1295751, 3730229, 1555011, 4265546, 4450370, 2322128, 1232047, 5340683, 3901175, 5102033, 4321191
};

// const unsigned int data_rows = 1000;
// const unsigned int data_cols = 10;
const unsigned ll d = 920403722748280569;
const unsigned ll n = 2108958572404460311;
const unsigned ll offset = 845690870767227654;
const int TRANSPOSE_BLOCK_DIM = 8;

void Statistics_CPU(unsigned ll *indep, unsigned ll *dep, int numcols);
void Statistics_GPU(thrust::device_vector<unsigned ll> indep, thrust::device_vector<unsigned ll> dep, int numcols);

void CPU_One_Sample_T_Interval(unsigned ll *data, int numcols);
void CPU_Two_Sample_T_Test(unsigned ll *data, unsigned ll *categories, int numcols);

void GPU_One_Sample_T_Interval(thrust::device_vector<unsigned ll> data, int numcols);
void GPU_Two_Sample_T_Test(thrust::device_vector<unsigned ll> data, thrust::device_vector<unsigned ll> categories, int numcols);


__global__ void MM_Sum(const unsigned ll* Ae, const unsigned ll* Be, unsigned ll* sum,const int Am,const int An,const int Bn)
{
	// initialize memory
	__shared__ unsigned ll a_shared[16][16];
	__shared__ unsigned ll b_shared[16][16];
	__shared__ unsigned ll c_shared[16][16];

	int c_idx = threadIdx.y * 16 + threadIdx.x;
	c_shared[threadIdx.y][threadIdx.x] = 0; // set everything to zero just the first time
    a_shared[threadIdx.y][threadIdx.x] = Ae[c_idx];
    b_shared[threadIdx.y][threadIdx.x] = Be[c_idx];
    __syncthreads();

    // lmao loop unrolling time my dudes
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        c_shared[threadIdx.y][threadIdx.x] += a_shared[threadIdx.y][i] * b_shared[i][threadIdx.x];
    }
    __syncthreads();

	// save to global
    atomicAdd(&sum[0], c_shared[threadIdx.y][threadIdx.x]);
    if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(&sum[0], offset);
}

unsigned ll* Calculate_Key()
{
    unsigned ll* m1_raw = nullptr;
    cudaGetSymbolAddress((void**)&m1_raw, m1);
    unsigned ll* m2_raw = nullptr;
    cudaGetSymbolAddress((void**)&m2_raw, m2);

    unsigned ll* key_dev;
    cudaMalloc((void**)&key_dev, sizeof(unsigned ll));

    MM_Sum<<<1,dim3(16,16)>>>(m1_raw,m2_raw,key_dev,16,16,16);
    return key_dev;
}


__host__ void print_1D(unsigned ll* data, int length) {
	// for (int i = 0; i < min(length, 20); i++) {
	// 	std::cout << data[i] << ' ';
    // }
    for (int i = length-20; i < length; i++) {
        std::cout << data[i] << ' ';
    }
	std::cout << std::endl;
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
__device__ unsigned ll modmult_dev(unsigned ll a, unsigned ll b, 
                                   unsigned ll n)
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


__global__ void gpu_decrypt(unsigned ll* cipher, unsigned ll* data, int numcols, unsigned ll *d, unsigned ll n)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// If thread is in the bounds of the data array
	if (col < numcols) {
        // decrypt and depad
        data[col] = modexp_dev(cipher[col], d[0], n) & 0x00000000ffffffff;
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

    // get length of the file in 64-bit chunks
    length = sb.st_size / 8;

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
	__shared__ unsigned ll tile[TRANSPOSE_BLOCK_DIM][TRANSPOSE_BLOCK_DIM + 1]; // add plus one to avoid bank conflict
	int j = blockIdx.x * TRANSPOSE_BLOCK_DIM + threadIdx.x;
	int i = blockIdx.y * TRANSPOSE_BLOCK_DIM + threadIdx.y;

	if(i <= numrows && j <= numcols) {

		tile[threadIdx.y][threadIdx.x]=A[i*numcols+j];
		__syncthreads();

		////block (by,bx) in AT
		int tj=blockIdx.y*blockDim.x+threadIdx.x;	////x for column
		int ti=blockIdx.x*blockDim.y+threadIdx.y;	////y for row
		AT[ti*numrows+tj] = tile[threadIdx.x][threadIdx.y];
	
	}
}

__host__ void Test_Entire_CPU(char *dataname)
{
	//////////////
	// Test CPU //
	//////////////

	int fd;
	int datalength; // already changed to bytes/8
	
	// we need to pass in dataname as the filename into this
    unsigned ll *cipher = File_To_Array(dataname, datalength, fd);
    

	// calculate matrix size
	const int numcols = 8;
    int numrows = datalength / numcols;
    
    printf("Datalength:%d\n", datalength);
    printf("Cols:%d\n", numcols);
    printf("Rows:%d\n", numrows); // should be 

	// **** CPU TRANSPOSE **** //
	unsigned ll *transpose = new unsigned ll[datalength];
	for (int i = 0; i < numrows; i++) {
		for (int j = 0; j < numcols; j++) {
			transpose[j * numrows + i] = cipher[i * numcols + j];
		}
    }
    
	// 0 = pid, 1 = male, 2 = female, 3 = other gender, 4 = age
	// 5 = deceased, 6 = released, 7 = in progress
	const int independent_col = 2; // female
	const int dependent_col = 5; // rate of deceased
	
	unsigned ll *indep_cipher = &transpose[independent_col*numrows];
	unsigned ll *dep_cipher = &transpose[dependent_col*numrows];
	
	printf("\nCipher text from file:\n");
	print_1D(indep_cipher, numrows);
	print_1D(dep_cipher, numrows);
	
    // Decrypt CPU:
	auto start_time = high_resolution_clock::now();
	
	unsigned ll *indep_decoded = new unsigned ll[numrows];
	unsigned ll *dep_decoded = new unsigned ll[numrows];
	
	cpu_decrypt(indep_cipher, indep_decoded, numrows);
	cpu_decrypt(dep_cipher, dep_decoded, numrows);
    printf("\nData after decryption on CPU \n");
    
	print_1D(indep_decoded, numrows);
    print_1D(dep_decoded, numrows);

    printf("\nData after depadding on CPU \n");

    for (int i = 0; i < numrows; i++) {
        indep_decoded[i] &= 0x00000000ffffffff; //0x0000ffffffffffff
        dep_decoded[i] &= 0x00000000ffffffff;
        // printf("\n%ld\n", 0x0000ffffffffffff);
    }

    print_1D(indep_decoded, numrows);
    print_1D(dep_decoded, numrows);
    
    
    auto stop_time = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop_time - start_time);

	cout << "\n\n";
	cout << "CPU Decryption Runtime: " << duration.count()/1000. << " ms" << endl; 

	// this assumes we transpose. if we don't, we need different parameters
	Statistics_CPU(indep_decoded, dep_decoded, numrows);

	munmap(cipher, datalength);
	close(fd);
}

__host__ void Test_Entire_GPU(char *dataname)
{
	//////////////
	// Test GPU //
    //////////////

    unsigned ll* d = Calculate_Key();

    unsigned ll* host_d = (unsigned ll*)malloc(sizeof(unsigned ll));
    cudaMemcpy(host_d, d, sizeof(unsigned ll), cudaMemcpyDeviceToHost);
    cout<<host_d[0]<<endl;
    

	int fd;
	int datalength; // already changed to bytes/8

	// we need to pass in dataname as the filename into this
	unsigned ll *cipher = File_To_Array(dataname, datalength, fd);

	// calculate matrix size
	const int numcols = 8;
    int numrows = datalength / numcols;
	const int independent_col = 2; // female
	const int dependent_col = 5; // rate of deceased


    // **** GPU TRANSPOSE **** //
    unsigned ll* cipher_gpu;
    cudaMalloc((void**)&cipher_gpu, numrows * numcols * sizeof(unsigned ll));
    cudaMemcpy(cipher_gpu, cipher, numrows * numcols * sizeof(unsigned ll),
    	    cudaMemcpyHostToDevice);

    unsigned ll *cipher_transpose_gpu; 
    cudaMalloc((void**)&cipher_transpose_gpu, numrows * numcols * sizeof(unsigned ll));

    const int block_size = TRANSPOSE_BLOCK_DIM;
    const int block_num_x = numcols / block_size; // will always be 1 since numcols = 1
    const int block_num_y= ceil((double) numrows / (double) block_size);

    transposeCoalesced<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
    (cipher_gpu, cipher_transpose_gpu, numrows, numcols);

    // Get columns that we will be operating on;
	unsigned ll *indep_cipher_gpu = &cipher_transpose_gpu[independent_col*numrows];
    unsigned ll *dep_cipher_gpu = &cipher_transpose_gpu[dependent_col*numrows];

    unsigned ll *indep_data_gpu;
    unsigned ll *dep_data_gpu;
    cudaMalloc((void**)&indep_data_gpu, numrows * sizeof(ll));
    cudaMalloc((void**)&dep_data_gpu, numrows * sizeof(ll));
    thrust::device_vector<unsigned ll> indep_data_thrust(numrows);
    thrust::device_vector<unsigned ll> dep_data_thrust(numrows);
    indep_data_gpu = thrust::raw_pointer_cast(indep_data_thrust.data());
    dep_data_gpu = thrust::raw_pointer_cast(dep_data_thrust.data()); 

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
    cudaEventRecord(start);
    

	// Set # of threads and blocks:
	int blockdim = 256;
	int num_blocks = ceil((double) numrows / (double) blockdim); // operate on one col at a time

    gpu_decrypt<<<num_blocks, blockdim>>>(indep_cipher_gpu, indep_data_gpu, numrows, d, n);
    gpu_decrypt<<<num_blocks, blockdim>>>(dep_cipher_gpu, dep_data_gpu, numrows, d, n);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

    unsigned ll *dep_cipher_host = new unsigned ll[numrows]; 
    unsigned ll *indep_cipher_host = new unsigned ll[numrows]; 
	cudaMemcpy(dep_cipher_host, dep_cipher_gpu, numrows * sizeof(ll),
            cudaMemcpyDeviceToHost);
    cudaMemcpy(indep_cipher_host, indep_cipher_gpu, numrows * sizeof(ll),
            cudaMemcpyDeviceToHost);

    unsigned ll *dep_data_host = new unsigned ll[numrows]; 
    unsigned ll *indep_data_host = new unsigned ll[numrows]; 
    cudaMemcpy(dep_data_host, dep_data_gpu, numrows * sizeof(ll),
            cudaMemcpyDeviceToHost);
    cudaMemcpy(indep_data_host, indep_data_gpu, numrows * sizeof(ll),
            cudaMemcpyDeviceToHost);
    
    printf("Datalength:%d\n", datalength);
    printf("Cols:%d\n", numcols);
    printf("Rows:%d\n", numrows); 

    printf("\nCipher text:\n");
    print_1D(indep_cipher_host, numrows);
    print_1D(dep_cipher_host, numrows);

	printf("\nData after decryption and depadding on GPU\n");
    print_1D(indep_data_host, numrows);
    print_1D(dep_data_host, numrows);

	printf("\nGPU runtime: %.4f ms\n", gpu_time);

	// printf("\nSpeedup: %.4f X\n", (float)(duration.count()) / gpu_time);

    Statistics_GPU(indep_data_thrust, dep_data_thrust, numrows);

	munmap(cipher, datalength);
	close(fd);
}

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
    std::cout<<"Standard error: \t"<<stderror<<"\n";
    std::cout<<"\nT-statistic for 95 percent confidence interval: \t"<<t_statistic<<"\n";
    std::cout<<"Margin of error for this sample: \t\t"<<moe<<"\n";
    std::cout<<"95 percent confident that the true population mean lies between "<<mean-moe<<" and "<<mean+moe<<"\n";

    // Runtime:
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
    std::cout<<"Standard error of difference: \t"<<stderror<<"\n";
    std::cout<<"\nT-statistic for difference of means compared to null hypothesis: "<<t_statistic<<"\n";
    std::cout<<"Alpha value: \t\t"<<alpha<<"\n";
    std::cout<<"P-value: \t\t"<<p_value<<"\n";
    std::cout<<"We "<<(p_value >= alpha/2 ? "fail to" : "")<<" reject the null hypothesis.\n";

    // runtime:
    std::cout<<"CPU runtime: "<<cpu_time.count()*1000.<<" ms."<<std::endl;
}

// functor to calculate standard deviation using thrust, by passing in the mean and subtracting/squaring
struct std_dev_func
{
	double mean = 0.0;
	std_dev_func(double _a) : mean(_a) {}

	__host__ __device__ double operator()(const int& val) const
	{
		return (val-mean) * (val-mean);
	}
};

void Statistics_GPU(thrust::device_vector<unsigned ll> indep, thrust::device_vector<unsigned ll> dep, int numcols)
{
    GPU_One_Sample_T_Interval(dep, numcols);

    GPU_Two_Sample_T_Test(dep, indep, numcols);
}

void GPU_One_Sample_T_Interval(thrust::device_vector<unsigned ll> data, int datalength)
{
    const double confidence = 0.95;
    
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float gpu_time=0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    // calculate mean with simple reduction
    double mean = thrust::reduce(data.begin(), data.end(), (double)0, thrust::plus<double>());
    mean /= datalength;

    // calculate standard deviation with fused transform-reduce using std_dev_func functor
    double stddev = thrust::transform_reduce(data.begin(), data.end(), std_dev_func(mean), (double)0, thrust::plus<double>());
    stddev = sqrt(stddev/(datalength-1));

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_time,start,end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // standard error
    double stderror = stddev / sqrt(datalength);

    // calculate t-statistic
    students_t dist(datalength - 1);
    double t_statistic = quantile(dist, confidence/2+0.5);

    // calculate margin of error
    double moe = t_statistic * stderror;

    std::cout<<"\nOne-Sample T-Interval GPU results: \n";
    std::cout<<"Sample size: \t\t"<<datalength<<"\n";
    std::cout<<"Sample mean: \t\t"<<mean<<"\n";
    std::cout<<"Sample std dev: \t"<<stddev<<"\n";
    std::cout<<"Standard error: \t"<<stderror<<"\n";
    std::cout<<"\nT-statistic for 95 percent confidence interval: \t"<<t_statistic<<"\n";
    std::cout<<"Margin of error for this sample: \t\t"<<moe<<"\n";
    std::cout<<"95 percent confident that the true population mean lies between "<<mean-moe<<" and "<<mean+moe<<"\n";

    printf("\nGPU runtime: %.4f ms\n",gpu_time);
}

void GPU_Two_Sample_T_Test(thrust::device_vector<unsigned ll> data, thrust::device_vector<unsigned ll> categories, int datalength)
{
    // level for statistical significance
    const double alpha = 0.05;

    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float gpu_time=0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(start);


    double mean[2] = { 0, 0 };
    int length[2] = { 0, 0 };
    // int blank[2]; //dummy array

    // key-sort the two vectors such that we have separated categories
    thrust::sort_by_key(categories.begin(), categories.end(), data.begin());
    // calculate the length of each section
    length[0] = thrust::count(categories.begin(), categories.end(), 0);
    length[1] = datalength - length[0];

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_time,start,end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    printf("\nGPU sorting runtime: %.4f ms\n",gpu_time);

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    gpu_time=0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    // create two CUDA streams
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);


    // calculate both means
    mean[0] = thrust::reduce(thrust::cuda::par.on(s1), data.begin(), data.begin() + length[0], (double)0, thrust::plus<double>());
    mean[1] = thrust::reduce(thrust::cuda::par.on(s2), data.begin() + length[0], data.end(), (double)0, thrust::plus<double>());
    mean[0] /= length[0];
    mean[1] /= length[1];

    double stddev[2] = { 0, 0 };
    double variance[2] = { 0, 0 };
    // calculate both standard deviations and variances w/ transform-reduce and std_dev_func
    variance[0] = thrust::transform_reduce(thrust::cuda::par.on(s1), data.begin(), data.begin() + length[0], std_dev_func(mean[0]), (double)0, thrust::plus<double>());
    variance[1] = thrust::transform_reduce(thrust::cuda::par.on(s2), data.begin() + length[0], data.end(), std_dev_func(mean[1]), (double)0, thrust::plus<double>());
    variance[0] /= (length[0] - 1);
    variance[1] /= (length[1] - 1);
    stddev[0] = sqrt(variance[0]);
    stddev[1] = sqrt(variance[1]);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_time,start,end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceSynchronize();
    // destroy streams
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

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
    std::cout<<"\nTwo-Sample Two-Tailed T-Test GPU results: \n";
    std::cout<<"Sample size[0]: \t"<<length[0]<<"\n";
    std::cout<<"Sample mean[0]: \t"<<mean[0]<<"\n";
    std::cout<<"Sample std dev[0]: \t"<<stddev[0]<<"\n";
    std::cout<<"Sample size[1]: \t"<<length[1]<<"\n";
    std::cout<<"Sample mean[1]: \t"<<mean[1]<<"\n";
    std::cout<<"Sample std dev[1]: \t"<<stddev[1]<<"\n\n";
    std::cout<<"Difference of means: \t\t"<<diffmeans<<"\n";
    std::cout<<"Degrees of freedom: \t\t"<<df<<"\n";
    std::cout<<"Standard error of difference: \t"<<stderror<<"\n";
    std::cout<<"\nT-statistic for difference of means compared to null hypothesis: "<<t_statistic<<"\n";
    std::cout<<"Alpha value: \t\t"<<alpha<<"\n";
    std::cout<<"P-value: \t\t"<<p_value<<"\n";
    std::cout<<"We "<<(p_value >= alpha/2 ? "fail to" : "")<<" reject the null hypothesis.\n";

    printf("\nGPU analysis runtime: %.4f ms\n",gpu_time);
}

int main(int argc, char *argv[])
{
	Test_Entire_CPU(argv[1]);
	Test_Entire_GPU(argv[1]);
	return 0;
}
