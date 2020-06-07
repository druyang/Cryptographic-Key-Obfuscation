#include <cstdio>
#include <cstdlib>

#include <algorithm> 
#include <iterator>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// #include <cublas_v2.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

#define ll long long

__device__  __constant__ unsigned ll m1[16*16] = {
    3898138, 4272236, 3818297, 3809756, 4037852, 4676915, 3500886, 3619362, 4730634, 2829970, 3517506, 4087421, 4739601, 3912997, 3681388, 6359441, 3255282, 6285463, 5859578, 6306639, 5707605, 6090034, 3791157, 3023551, 6494436, 5639060, 2442072, 4931038, 3374951, 2973521, 3534381, 2006977, 4512006, 2730539, 4739724, 3385241, 4656916, 3938940, 3796672, 4762700, 6592658, 4096400, 3402646, 4623696, 6895815, 5140537, 3059772, 6976185, 4090503, 3847150, 5493941, 4084221, 5289630, 3366036, 3877302, 6044678, 5833958, 3454838, 5369570, 4076362, 4112938, 3652335, 5586920, 4455035, 3530871, 5166626, 2851545, 6459783, 3200179, 3905475, 4882530, 6386745, 4939154, 6054883, 6988990, 2800608, 4300093, 3029774, 6468896, 5294418, 4392603, 4249913, 5528383, 3092627, 5421793, 3344070, 6070234, 3644930, 4301352, 4607155, 5854072, 5832719, 5571846, 5018476, 1457339, 3594239, 3309544, 2658855, 3726900, 5894573, 5658810, 3933981, 6213873, 4204623, 4381714, 6075984, 5200611, 3378648, 3182854, 5359960, 5160722, 4496497, 6592430, 5971095, 4102753, 3446501, 2707085, 4192901, 3942405, 4480361, 5734622, 6092668, 5815420, 2193260, 3654966, 4195351, 4400242, 4859107, 4152135, 1848942, 5410256, 5432025, 4207561, 3319602, 4141769, 4403163, 2130807, 5398166, 4417568, 4774938, 1706853, 4677230, 6013745, 3612519, 4617344, 3915945, 6005133, 4442264, 5757908, 5010119, 3258015, 3542214, 3510823, 3080679, 4174442, 5279308, 3338275, 1425392, 4225674, 4111864, 5243952, 4014867, 3083188, 3784711, 3580232, 4758474, 3637293, 2206083, 2646303, 2474987, 6277880, 1158152, 3658919, 5452372, 3882484, 4304057, 3165794, 2191558, 5989865, 5078279, 6020807, 4652723, 3458259, 5856963, 4341712, 5928672, 6142960, 5079986, 4026505, 7370147, 5035209, 4660278, 3923764, 2550655, 0, 4165580, 4629051, 3069243, 3734965, 2812772, 2957817, 5182954, 4006232, 4623017, 6091379, 4084730, 3843241, 4139640, 5351058, 5051956, 6396350, 3120492, 4636888, 3984535, 6898848, 3942165, 4713424, 3750126, 6147513, 5964568, 5260514, 5227714, 4282723, 4550280, 6269824, 5316692, 6081364, 5527953, 4178586, 3084830, 4026779, 5714687, 5225745, 5787677, 4253001, 4572657, 3353132, 5175311, 3802632, 6154279, 4099225, 4429291, 3657324, 5316210, 4680313, 3756697, 5949820, 5268209, 4685384, 4284995, 3783679, 3954066, 3515489, 4653612, 5879525, 5531160
};

__device__  __constant__ unsigned ll m2[16*16] = {
    5402623, 3995627, 6743696, 3542762, 4575592, 3695284, 3918622, 4538491, 6181070, 2512172, 4698104, 4163423, 4292282, 4453561, 1477165, 4966421, 4736401, 4810828, 5344335, 4837246, 4286233, 2634311, 4147100, 1921287, 6340262, 4113615, 3795215, 4211793, 2113035, 2556419, 2938045, 5122808, 4298089, 4512884, 3691197, 5534985, 4851922, 2772637, 2236804, 6120396, 2929315, 4143837, 3787446, 3239459, 4775456, 865982, 4496660, 1703666, 1599063, 4549671, 3501246, 1249567, 3323392, 4137217, 4195107, 6887026, 4291745, 5286819, 4357805, 2681316, 3140604, 4414659, 2713437, 4684680, 4623493, 2717830, 5068866, 5504463, 4723859, 6560104, 3436816, 5727689, 1648325, 4410464, 3260375, 3527822, 3801152, 5313457, 3446586, 5860645, 4737773, 5422197, 2731218, 4057691, 4550777, 5336448, 3092465, 2570986, 4040807, 3075376, 4867819, 4488883, 5773716, 3522895, 2814470, 5650927, 4377915, 4178781, 4884170, 3743092, 2828557, 4740730, 3232547, 4097158, 4680480, 5160081, 4033217, 2703093, 1601640, 5648239, 3977663, 5292342, 1858079, 5437452, 5778958, 3868468, 4060999, 3947463, 5609988, 4810874, 5726724, 4889048, 3297829, 4738813, 6375997, 3406592, 5284049, 4546058, 5626721, 4157315, 6594087, 6505334, 7104890, 3101450, 4471290, 4693939, 1906608, 6790147, 2749917, 4280580, 4847134, 5819494, 2243550, 2962854, 1061216, 3903053, 5066876, 4876405, 4163034, 4753225, 6430668, 4729571, 4429114, 5100039, 6968806, 4146280, 4409023, 4459757, 2614041, 5123350, 4132817, 4575138, 3217081, 4959027, 3843245, 6940258, 4267578, 3587797, 3931136, 5217963, 6225758, 2514754, 4230712, 5868142, 4260445, 4583537, 4107203, 1696201, 3402111, 4265050, 5518600, 5989181, 1619745, 3704475, 4369304, 2739916, 4914357, 3048664, 5007979, 4977233, 2376675, 3990140, 1889822, 3240595, 2616037, 2789849, 4489124, 4465373, 4793428, 6241130, 674170, 3887638, 3952595, 3276410, 3403888, 3426655, 5523518, 3812274, 5021758, 4478267, 3835431, 2826298, 3810009, 3389970, 4590227, 5646501, 698792, 4943485, 3322046, 3383689, 2234016, 2160407, 2022331, 5654802, 5180190, 3407774, 3774041, 4347928, 5845572, 2056892, 6877570, 2917990, 3144028, 2784578, 4687680, 4418303, 4558507, 5042387, 3523901, 4224349, 3338556, 5510521, 2140311, 3958657, 4013730, 1295751, 3730229, 1555011, 4265546, 4450370, 2322128, 1232047, 5340683, 3901175, 5102033, 4321191
};

// __device__  __constant__  ll test1[4] = {2, 1, 2, 1};

// __device__  __constant__  ll test2[4] = {1, 1, 2, 1};

// __global__ void Matrix_Multiplication_AB_Kernel_Poorman(const unsigned ll* Ae,const unsigned ll* Be, unsigned ll* Ce,const int Am,const int An,const int Bn)
// {
// 	int i=blockIdx.x*blockDim.x+threadIdx.x;
// 	int j=blockIdx.y*blockDim.y+threadIdx.y;

// 	ll val=0;
// 	for(int k=0;k<An;k++)
// 		val+=Ae[i*An+k]*Be[k*Bn+j];
// 	Ce[i*Bn+j]=val;
// } 


__global__ void MM_Key(const unsigned ll* Ae, const unsigned ll* Be, unsigned ll* Ce,const int Am,const int An,const int Bn)
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
	Ce[c_idx] = c_shared[threadIdx.y][threadIdx.x];
}


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
}

// __host__ void print_1D( ll* data) 
// {
//     for (int i = 0; i < 16; i++) {
//         std::cout << data[i] << ' ';
//     }
// 	std::cout << std::endl;
// }


int main()
{
    // Create a handle for CUBLAS
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // int lda = 16, ldb = 16, ldc = 16;
    // int lda = 2, ldb = 2, ldc = 2;

    // const unsigned ll alf = 1;
    // const unsigned ll bet = 0;
    // const unsigned ll *alpha = &alf;
    // const unsigned ll *beta = &bet;


    // thrust::device_vector<unsigned ll> output_m(16*16);
    // unsigned ll* output_raw = thrust::raw_pointer_cast(output_m.data());

    unsigned ll* m1_raw = nullptr;
    cudaGetSymbolAddress((void**)&m1_raw, m1);
    unsigned ll* m2_raw = nullptr;
    cudaGetSymbolAddress((void**)&m2_raw, m2);

    unsigned ll* d;
    cudaMalloc((void**)&d, sizeof(unsigned ll));

    // MM_Key<<<1,dim3(16,16)>>>(m1_raw,m2_raw,output_raw,16,16,16);
    MM_Sum<<<1,dim3(16,16)>>>(m1_raw,m2_raw,d,16,16,16);


    // thrust::constant_iterator<int> iter(0);
    // int e;
    // cudaMalloc((void**)&e, sizeof(int));

    // thrust::pair<int*, unsigned ll*> new_end;
    // thrust::equal_to<int> binary_pred;
    // new_end = thrust::reduce_by_key(iter, iter + 16*16, output_raw, e, &d, binary_pred);


    unsigned ll* d_host = (unsigned ll*)malloc(sizeof(unsigned ll));
    cudaMemcpy(d_host, d, sizeof(unsigned ll), cudaMemcpyDeviceToHost);

    // thrust::device_vector< ll> output_m(2*2);
    //  ll* output_raw = thrust::raw_pointer_cast(output_m.data());

    //  ll* m1_raw = nullptr;
    // cudaGetSymbolAddress((void**)&m1_raw, test1);
    //  ll* m2_raw = nullptr;
    // cudaGetSymbolAddress((void**)&m2_raw, test2);

    // Do the actual multiplication
    // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 16, 16, 16, alpha, m1_raw, lda, m2_raw, ldb, beta, output_raw, ldc);

    // thrust::host_vector< ll> host = output_m;
    //  ll* output_host = thrust::raw_pointer_cast(host.data());

    // print_1D(output_host);

    //  ll key = ( ll)
    // unsigned ll checkkey = (unsigned ll)thrust::reduce(output_m.begin(), output_m.end(), (unsigned ll)0, thrust::plus<unsigned ll>());
    // checkkey += 0/*3066470427934761*/;
    std::cout<<*d_host<<" calculated"<<std::endl;

    std::cout<<920403722748280569-845690870767227654<<" expected"<<std::endl;
    
    // Destroy the handle
    // cublasDestroy(handle);
    // cudaDeviceSynchronize();
}
