/**
 * Basic CPU and GPU statistics to add to our project
 *
 * Usage: ./statistics [datafile] [categoryfile] [gpu|cpu]
 * where datafile is the data, and categoryfile is an equally long array of either 0 or 1
 * which indicates which sample the data at the same index belongs to (for a t-test)
 */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <boost/math/distributions/students_t.hpp>
#include <thrust/execution_policy.h>

using namespace std::chrono;
using boost::math::students_t;

void Test_CPU_Statistics(const char *dataname, const char *categoryname);
void Test_GPU_Statistics(const char *dataname, const char *categoryname);

int32_t *File_To_Array(const char *filename, int &length, int &fd);

void CPU_One_Sample_T_Interval(int32_t *data, int datalength);
void CPU_Two_Sample_T_Test(int32_t *data, int32_t *categories, int datalength);

void GPU_One_Sample_T_Interval(thrust::device_vector<int> data, int datalength);
void GPU_Two_Sample_T_Test(thrust::device_vector<int> data, thrust::device_vector<int> categories, int datalength);

int main(const int argc, const char* argv[])
{
    if (argc != 4) {
        fprintf(stderr, "Usage: ./statistics [datafile] [categoryfile] [gpu|cpu]\n");
        return -1;
    }
    FILE* fp;
    if ((fp = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Error: data not readable\n");
        return -2;
    }
    fclose(fp);
    if ((fp = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "Error: categories not readable\n");
        return -2;
    }
    fclose(fp);
    if (strcmp(argv[3], "cpu") == 0) {
        Test_CPU_Statistics(argv[1], argv[2]);
    } else if (strcmp(argv[3], "gpu") == 0) {
        Test_GPU_Statistics(argv[1], argv[2]);
    } else {
        fprintf(stderr, "Error: not gpu or cpu\n");
        return -2;
    }
    return 0;
}

int32_t *File_To_Array(const char *filename, int &length, int &fd)
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
    length = sb.st_size / 8;

    // mmap asks the OS to provision a chunk of disk storage out to contiguous (read aligned, coalesced) RAM
    // this is the reverse of using 'swap space' to cache some RAM out to disk when under memory pressure
    // we give it the fd file descriptor and the size of the file to tell the OS which chunk of disk to allocate as memory
    // and also give it certain permissions
    // this is a direct array of data so we can cast it to whatever form we like, in this case bytes
    // and then we can address the pointer as an array as we're familiar with
    return (int32_t *)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
}

void Test_CPU_Statistics(const char *dataname, const char *categoryname)
{
    auto cpu_start = system_clock::now();

    int datalength;
    int fd;
    int32_t *data = File_To_Array(dataname, datalength, fd);
    int fd2;
    int32_t *categories = File_To_Array(categoryname, datalength, fd2);

    auto cpu_end = system_clock::now();
    duration<double> cpu_time=cpu_end-cpu_start;
    std::cout<<"File mmap time: "<<cpu_time.count()*1000.<<" ms."<<std::endl;

    CPU_One_Sample_T_Interval(data, datalength);

    CPU_Two_Sample_T_Test(data, categories, datalength);

    munmap(data, datalength);
    close(fd);
    munmap(categories, datalength);
    close(fd2);
}

void CPU_One_Sample_T_Interval(int32_t *data, int datalength)
{
    // timer
    auto cpu_start = system_clock::now();

    const double confidence = 0.95;

    // calculate mean
    double mean;
    double castdata;
    for (int i = 0; i < datalength; i++) {
        castdata = (double)data[i];
        // std::cout<<(double)castdata<<"\n";
        mean += castdata / datalength;
    }

    double stddev;
    double difference;
    // calculate std deviation
    for (int i = 0; i < datalength; i++) {
        castdata = (double)data[i];
        difference = castdata - mean;
        // std::cout<<difference<<"\n";
        stddev += difference * difference / (datalength-1);
    }
    stddev = sqrt(stddev);

    auto cpu_end = system_clock::now();
    duration<double> cpu_time=cpu_end-cpu_start;

    // standard error
    double stderror = stddev / sqrt(datalength);

    // calculate t-statistic
    students_t dist(datalength - 1);
    double t_statistic = quantile(dist, confidence/2+0.5);

    // calculate margin of error
    double moe = t_statistic * stderror;


    // print out statistics
    std::cout<<"\nOne-Sample T-Interval CPU results: \n";
    std::cout<<"Sample size: \t\t"<<datalength<<"\n";
    std::cout<<"Sample mean: \t\t"<<mean<<"\n";
    std::cout<<"Sample std dev: \t"<<stddev<<"\n";
    std::cout<<"Standard error: \t"<<stderror<"\n";
    std::cout<<"\nT-statistic for 95 percent confidence interval: \t"<<t_statistic<<"\n";
    std::cout<<"Margin of error for this sample: \t\t"<<moe<<"\n";
    std::cout<<"95 percent confident that the true population mean lies between "<<mean-moe<<" and "<<mean+moe<<"\n";

    std::cout<<"CPU runtime: "<<cpu_time.count()*1000.<<" ms."<<std::endl;
}

void CPU_Two_Sample_T_Test(int32_t *data, int32_t *categories, int datalength)
{
    auto cpu_start = system_clock::now();

    // level for statistical significance
    const double alpha = 0.05;

    // calculate mean
    double mean[2] = { 0, 0 };
    int length[2] = { 0, 0 };
    double castdata;
    int index;
    for (int i = 0; i < datalength; i++) {
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
    for (int i = 0; i < datalength; i++) {
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


void Test_GPU_Statistics(const char *dataname, const char *categoryname)
{
    auto cpu_start = system_clock::now();

    int datalength;
    int fd;
    int32_t *data = File_To_Array(dataname, datalength, fd);
    int fd2;
    int32_t *categories = File_To_Array(categoryname, datalength, fd2);

    auto cpu_end = system_clock::now();
    duration<double> cpu_time=cpu_end-cpu_start;
    std::cout<<"File mmap time: "<<cpu_time.count()*1000.<<" ms."<<std::endl;

    std::vector<int> data_host(data, data + datalength);
    thrust::device_vector<int> data_thrust = data_host;

    GPU_One_Sample_T_Interval(data_thrust, datalength);

    std::vector<int> categories_host(categories, categories + datalength);
    thrust::device_vector<int> categories_thrust = categories_host;

    GPU_Two_Sample_T_Test(data_thrust, categories_thrust, datalength);

    munmap(data, datalength);
    close(fd);
    munmap(categories, datalength);
    close(fd2);
}

struct std_dev_func
{
	double mean = 0.0;
	std_dev_func(double _a) : mean(_a) {}

	__host__ __device__ double operator()(const int& val) const
	{
		return (val-mean) * (val-mean);
	}
};

void GPU_One_Sample_T_Interval(thrust::device_vector<int> data, int datalength)
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
    std::cout<<"Standard error: \t"<<stderror<"\n";
    std::cout<<"\nT-statistic for 95 percent confidence interval: \t"<<t_statistic<<"\n";
    std::cout<<"Margin of error for this sample: \t\t"<<moe<<"\n";
    std::cout<<"95 percent confident that the true population mean lies between "<<mean-moe<<" and "<<mean+moe<<"\n";

    printf("\nGPU runtime: %.4f ms\n",gpu_time);
}

void GPU_Two_Sample_T_Test(thrust::device_vector<int> data, thrust::device_vector<int> categories, int datalength)
{
    // level for statistical significance
    const double alpha = 0.05;

    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float gpu_time=0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    // create two CUDA streams
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    double mean[2] = { 0, 0 };
    int length[2] = { 0, 0 };
    // int blank[2]; //dummy array

    // key-sort the two vectors such that we have separated categories
    thrust::sort_by_key(categories.begin(), categories.end(), data.begin());
    // calculate the length of each section
    length[0] = thrust::count(categories.begin(), categories.end(), 0);
    length[1] = datalength - length[0];

    // calculate both means
    mean[0] = thrust::reduce(thrust::cuda::par.on(s1), data.begin(), data.begin() + length[0], (double)0, thrust::plus<double>());
    mean[1] = thrust::reduce(thrust::cuda::par.on(s2), data.begin() + length[0], data.end(), (double)0, thrust::plus<double>());
    // thrust::pair<int*,double*> new_end;
    // thrust::equal_to<int> binary_pred;
    // int* categories_raw = thrust::raw_pointer_cast(categories.data());
    // double* data_raw = (double *)thrust::raw_pointer_cast(data.data());
    // new_end = thrust::reduce_by_key(categories_raw, categories_raw + datalength, data_raw, blank, mean, binary_pred);
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
    std::cout<<"Standard error of difference: \t"<<stderror<"\n";
    std::cout<<"\nT-statistic for difference of means compared to null hypothesis: "<<t_statistic<<"\n";
    std::cout<<"Alpha value: \t\t"<<alpha<<"\n";
    std::cout<<"P-value: \t\t"<<p_value<<"\n";
    std::cout<<"We "<<(p_value >= alpha/2 ? "fail to" : "")<<" reject the null hypothesis.\n";

    printf("\nGPU runtime: %.4f ms\n",gpu_time);
}
