#include "kernel.cu"

// #define CheckWithCPU

#define SCALE 512
#define M SCALE
#define K SCALE
#define N SCALE
#define M_MAT_A_ROW_NUM         M //how many rows in A
#define K_MAT_A_COLUMN_NUM      K //how many column in A
#define K_MAT_B_ROW_NUM         K_MAT_A_COLUMN_NUM //how many rows in B
#define N_MAT_B_COLUMN_NUM      N //how many column in B
#define M_MAT_C_ROW_NUM         M_MAT_A_ROW_NUM        //how many rows in C
#define N_MAT_C_COLUMN_NUM      N_MAT_B_COLUMN_NUM     //how many column in C
#define MATRIX_GLOBAL_SIZE      (M_MAT_C_ROW_NUM * N_MAT_C_COLUMN_NUM)

int main(){
    // host memory allocate
    float* h_a = (float*)malloc(M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM*sizeof(float));
    float* h_b = (float*)malloc(K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM*sizeof(float));
    half* h_a_half = (half*)malloc(M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM*sizeof(half));
    half* h_b_half = (half*)malloc(K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM*sizeof(half));
    float* h_c_cpu = (float*)malloc(MATRIX_GLOBAL_SIZE*sizeof(float));
    float* h_c_naive_cuda = (float*)malloc(MATRIX_GLOBAL_SIZE*sizeof(float));
    float* h_c_cublas_tc = (float*)malloc(MATRIX_GLOBAL_SIZE*sizeof(float));
    float* h_c_cublas_cuda = (float*)malloc(MATRIX_GLOBAL_SIZE*sizeof(float));
    half* h_c_wmma_tc_half = (half*)malloc(MATRIX_GLOBAL_SIZE*sizeof(half));
    srand((unsigned)time(NULL));
    for(int i=0;i< MATRIX_GLOBAL_SIZE;i++)
    {
        h_c_cpu[i] = 0.0;
        h_c_naive_cuda[i] = 0.0;
        h_c_cublas_tc[i] = 0.0;
        h_c_cublas_cuda[i] = 0.0;
        h_c_wmma_tc_half[i] = 0.0;
    }
    for (int i = 0; i < M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM; i++){
        h_a[i] = (rand() % 3)/3.0;// (111.1f / (float)i) % 2.4f;
        h_a_half[i] = half(h_a[i]);
    }
    for (int i = 0; i < K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM; i++){
        h_b[i] = (rand() % 3)/3.0;// (i / 111.0) % 3.0;
        h_b_half[i] = half(h_b[i]);
    }

    //device memory allocate
    float *d_a, *d_b, *d_c_naive_cuda, *d_c_cublas_tc, *d_c_cublas_cuda;
    half *d_a_half, *d_b_half, *d_c_wmma_tc_half;
    CHECK_CUDA(cudaMalloc(&d_a, M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM * sizeof(float)))
    CHECK_CUDA(cudaMalloc(&d_b, K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM * sizeof(float)))
    CHECK_CUDA(cudaMalloc(&d_c_naive_cuda, MATRIX_GLOBAL_SIZE * sizeof(float)))
    CHECK_CUDA(cudaMalloc(&d_c_cublas_tc, MATRIX_GLOBAL_SIZE * sizeof(float)))
    CHECK_CUDA(cudaMalloc(&d_c_cublas_cuda, MATRIX_GLOBAL_SIZE * sizeof(float)))
    CHECK_CUDA(cudaMalloc(&d_a_half, M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM * sizeof(half)))
    CHECK_CUDA(cudaMalloc(&d_b_half, K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM * sizeof(half)))
    CHECK_CUDA(cudaMalloc(&d_c_wmma_tc_half, MATRIX_GLOBAL_SIZE * sizeof(half)))

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_a_half, h_a_half, M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_half, h_b_half, K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM * sizeof(half), cudaMemcpyHostToDevice));

    CHECK_CUDA(mulMatrixWithNaiveCuda(d_c_naive_cuda, d_a, d_b, M, K, N));
    CHECK_CUDA(mulMatrixWithCublasCuda(d_c_cublas_cuda, d_a, d_b, M, K, N));
    CHECK_CUDA(mulMatrixWithCublasTC(d_c_cublas_tc, d_a, d_b, M, K, N));
    CHECK_CUDA(mulMatrixWithWmmaTC(d_c_wmma_tc_half, d_a_half, d_b_half, M, K, N));
    // CHECK_CUDA(mulMatrixWithCublasCuda(d_c_cublas_cuda, d_a, d_b, M, K, N));

    CHECK_CUDA(cudaMemcpy(h_c_naive_cuda, d_c_naive_cuda, MATRIX_GLOBAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_c_cublas_tc, d_c_cublas_tc, MATRIX_GLOBAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_c_cublas_cuda, d_c_cublas_tc, MATRIX_GLOBAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_c_wmma_tc_half, d_c_wmma_tc_half, MATRIX_GLOBAL_SIZE * sizeof(half), cudaMemcpyDeviceToHost));

#ifdef CheckWithCPU
    mulMatrixWithCpu(h_c_cpu, h_a, h_b, M, K, N);
    printf("Sample results:\n");
    for(int i=0;i<4;i++)
        // printf("cpu[%d]: %5f, tc[%d]: %5f,gpu[%d]:%5f\n", i, h_c_cpu[i], i, h_c_cublas_tc[i], i, h_c_naive_cuda[i]);
        printf("cpu[%d]: %5f, tc[%d]: %5f, gpu[%d]:%5f, wmma[%d]:%5f\n", i, h_c_cpu[i], i, h_c_cublas_tc[i], i, h_c_naive_cuda[i], i, static_cast<float>(h_c_wmma_tc_half[i]));
#endif

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CHECK_CUDA(cudaDeviceReset())

}