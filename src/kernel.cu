#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include <omp.h>
#include <iostream>
#include <mma.h>

#define A100
#ifdef A100
#define SM_NUM   108
#define CUDA_CORE_PER_SM        64
#define CUDA_CORE_PER_WARP      16
#else
#define SM_NUM   40
#define CUDA_CORE_PER_SM        128
#define CUDA_CORE_PER_WARP      16
#endif

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using namespace nvcuda;

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", code: " << error \
                  << ", reason: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    const cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", reason: " << status << std::endl; \
        exit(1); \
    } \
}

__global__ void blank_warmingGPU() {}

// A: row major; B: row major; C: row major;
void mulMatrixWithCpu(float* c, float* a, float* b, int m, int k, int n)
{
    int i_x = 0, i_y = 0;
#pragma omp parallel for private(i_x, i_y)
    for (int i = 0;i < m * n;i++)
    {
        i_x = i / n;  //i_x line of A,
        i_y = i % n;  //i_y column of B;
        for (int j = 0;j < k;j++)
        {
            c[i] += a[i_x * k + j] * b[j * n + i_y];
        }   
    }
}

// A: row major; B: row major; C: row major;
__global__ void mulKernel(float* c, float* a, float* b, int m, int k, int n)
{
    int i = 0;
    int j = 0;
    for(int index = blockIdx.x * blockDim.x + threadIdx.x;index < m * n;index+=gridDim.x*blockDim.x)
    {
        i = index / n;
        j = index % n;
        for (int l = 0; l < k; l++)
        {
            c[index] += a[i * k + l] * b[l * n + j];
        }
    }
}

// A: row major; B: row major; C: row major;
cudaError_t mulMatrixWithNaiveCuda(float* c, float* a, float* b, int m, int k, int n)
{
    cudaError_t cudaStatus = cudaSuccess;
    blank_warmingGPU << <1, 1 >> > ();
    // create two events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event on the default stream
    cudaEventRecord(start);
    // execute kernel
    mulKernel << <SM_NUM, 32*CUDA_CORE_PER_SM / CUDA_CORE_PER_WARP >> > (c, a, b, m, k, n);
   // record stop event on the default stream
    cudaEventRecord(stop);
    // wait until the stop event completes
    cudaEventSynchronize(stop);
    // calculate the elapsed time between two events
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time_naive_cuda is %f ms.\n\n", time);
    // clean up the two events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return cudaStatus;
}

// A: row major; B: row major; C: row major;
/* Cublas Cuda perform almost same as Cublas TC, since parameter 'cublasGemmAlgo_t' doesn't have effect on NVIDIA Ampere architecture GPUs (A100) and newer. (https://docs.nvidia.com/cuda/cublas/index.html?highlight=gemmEx#cublasgemmalgo-t) */
cudaError_t mulMatrixWithCublasCuda(float* c, float* a, float* b, int m, int k, int n)
{
    cudaError_t cudaStatus = cudaSuccess;
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;
    blank_warmingGPU << <1, 1 >> > ();
    // create two events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event on the default stream
    cudaEventRecord(start);
    // execute kernel
    // Result in row-major format
    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              n, m, k,
                              &alpha,
                              b, CUDA_R_32F, n,
                              a, CUDA_R_32F, k,
                              &beta,
                              c, CUDA_R_32F, n,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT));

    CHECK_CUBLAS(cublasDestroy(handle));
   // record stop event on the default stream
    cudaEventRecord(stop);
    // wait until the stop event completes
    cudaEventSynchronize(stop);
    // calculate the elapsed time between two events
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time_cublas_cuda is %f ms.\n\n", time);
    // clean up the two events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return cudaStatus;
}

// A: row major; B: row major; C: row major;
cudaError_t mulMatrixWithCublasTC(float* c, float* a, float* b, int m, int k, int n)
{
    cudaError_t cudaStatus = cudaSuccess;
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;
    blank_warmingGPU << <1, 1 >> > ();
    // create two events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event on the default stream
    cudaEventRecord(start);
    // execute kernel
    // Result in row-major format
    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              n, m, k,
                              &alpha,
                              b, CUDA_R_32F, n,
                              a, CUDA_R_32F, k,
                              &beta,
                              c, CUDA_R_32F, n,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CHECK_CUBLAS(cublasDestroy(handle));
   // record stop event on the default stream
    cudaEventRecord(stop);
    // wait until the stop event completes
    cudaEventSynchronize(stop);
    // calculate the elapsed time between two events
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time_cublas_tc is %f ms.\n\n", time);
    // clean up the two events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return cudaStatus;
}

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void wmmaNaiveKernel(half *C, half *A, half *B, size_t M,
                                size_t K, size_t N) {
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t warp_row = blockIdx.y * WMMA_M;
    const size_t warp_col = blockIdx.x * WMMA_N;

    if (warp_row >= M && warp_col >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;

    wmma::fill_fragment(C_frag, 0.0);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, B + i * WMMA_K + warp_col * K, K);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
}

// A: row major; B: row major; C: row major;
cudaError_t mulMatrixWithWmmaTC(half* c, half* a, half* b, size_t m, size_t k, size_t n)
{
    cudaError_t cudaStatus = cudaSuccess;
    blank_warmingGPU << <1, 1 >> > ();
    // create two events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event on the default stream
    cudaEventRecord(start);
    // execute kernel
    wmmaNaiveKernel << <SM_NUM, 32*CUDA_CORE_PER_SM / CUDA_CORE_PER_WARP >> > (c, a, b, m, k, n);
   // record stop event on the default stream
    cudaEventRecord(stop);
    // wait until the stop event completes
    cudaEventSynchronize(stop);
    // calculate the elapsed time between two events
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time_wmma_tc is %f ms.\n\n", time);
    // clean up the two events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return cudaStatus;
}