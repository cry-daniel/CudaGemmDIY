#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

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

// Kernel for matrix addition using CUDA cores
__global__ void matrixAdd(const float* A, const float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;
    if (idx < total_elements) {
        C[idx] = A[idx] + B[idx];
    }
}

void matrixMultiplyTensorCore(const float *a, const float *b, float *c, int m, int n, int k) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, k,
                              &alpha,
                              a, CUDA_R_32F, m,
                              b, CUDA_R_32F, k,
                              &beta,
                              c, CUDA_R_32F, m,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CHECK_CUBLAS(cublasDestroy(handle));
}

int main() {
    int m = 512;  // number of rows of A and C
    int n = 512;  // number of columns of B and C
    int k = 512;  // number of columns of A and rows of B

    size_t bytes = m * k * sizeof(float);

    float *h_A, *h_B, *h_C, *h_D, *h_E;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(m * n * sizeof(float));
    h_D = (float*)malloc(m * n * sizeof(float));
    h_E = (float*)malloc(m * n * sizeof(float));

    // Initialize matrices A and B
    for (int i = 0; i < m * k; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C, *d_D, *d_E;

    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_D, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_E, m * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // CUDA core matrix addition
    int blockSize = 256;
    int numBlocks = (m * k + blockSize - 1) / blockSize;
    matrixAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, m, k);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Tensor Core matrix multiplication
    matrixMultiplyTensorCore(d_A, d_B, d_D, m, n, k);

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D, d_D, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Validate results
    bool correctAdd = true;
    for (int i = 0; i < m * k; i++) {
        if (h_C[i] != 2.0f) {
            correctAdd = false;
            break;
        }
    }

    bool correctMul = true;
    for (int i = 0; i < m * n; i++) {
        if (h_D[i] != k) {
            correctMul = false;
            break;
        }
    }

    if (correctAdd) {
        std::cout << "Matrix addition is correct!" << std::endl;
    } else {
        std::cout << "Matrix addition is incorrect!" << std::endl;
    }

    if (correctMul) {
        std::cout << "Matrix multiplication is correct!" << std::endl;
    } else {
        std::cout << "Matrix multiplication is incorrect!" << std::endl;
    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_E);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));
    CHECK_CUDA(cudaFree(d_E));

    return 0;
}