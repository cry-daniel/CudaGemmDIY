#include "kernel.cu"

// #define CheckWithCPU

#define SCALE 512
#define M SCALE
#define K 256
#define N 128
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
    float* h_c_cpu = (float*)malloc(MATRIX_GLOBAL_SIZE*sizeof(float));
    float* h_c_naive_cuda = (float*)malloc(MATRIX_GLOBAL_SIZE*sizeof(float));
    float* h_c_cublas_tc = (float*)malloc(MATRIX_GLOBAL_SIZE*sizeof(float));
    float* h_c_cublas_cuda = (float*)malloc(MATRIX_GLOBAL_SIZE*sizeof(float));
    srand((unsigned)time(NULL));
    for(int i=0;i< MATRIX_GLOBAL_SIZE;i++)
    {
        h_c_cpu[i] = 0.0;
        h_c_naive_cuda[i] = 0.0;
        h_c_cublas_tc[i] = 0.0;
        h_c_cublas_cuda[i] = 0.0;
    }
    for (int i = 0; i < M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM; i++)
        h_a[i] = (rand() % 3)/3.0;// (111.1f / (float)i) % 2.4f;
    for (int i = 0; i < K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM; i++)
        h_b[i] = (rand() % 3)/3.0;// (i / 111.0) % 3.0;

    //device memory allocate
    float *d_a, *d_b, *d_c_naive_cuda, *d_c_cublas_tc, *d_c_cublas_cuda;
    CHECK_CUDA(cudaMalloc(&d_a, M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM * sizeof(float)))
    CHECK_CUDA(cudaMalloc(&d_b, K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM * sizeof(float)))
    CHECK_CUDA(cudaMalloc(&d_c_naive_cuda, MATRIX_GLOBAL_SIZE * sizeof(float)))
    CHECK_CUDA(cudaMalloc(&d_c_cublas_tc, MATRIX_GLOBAL_SIZE * sizeof(float)))
    CHECK_CUDA(cudaMalloc(&d_c_cublas_cuda, MATRIX_GLOBAL_SIZE * sizeof(float)))

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M_MAT_A_ROW_NUM * K_MAT_A_COLUMN_NUM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K_MAT_B_ROW_NUM * N_MAT_B_COLUMN_NUM * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(mulMatrixWithNaiveCuda(d_c_naive_cuda, d_a, d_b, M, K, N));
    CHECK_CUDA(mulMatrixWithCublasCuda(d_c_cublas_cuda, d_a, d_b, M, K, N));
    CHECK_CUDA(mulMatrixWithCublasTC(d_c_cublas_tc, d_a, d_b, M, K, N));
    // CHECK_CUDA(mulMatrixWithCublasCuda(d_c_cublas_cuda, d_a, d_b, M, K, N));

    CHECK_CUDA(cudaMemcpy(h_c_naive_cuda, d_c_naive_cuda, MATRIX_GLOBAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_c_cublas_tc, d_c_cublas_tc, MATRIX_GLOBAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_c_cublas_cuda, d_c_cublas_tc, MATRIX_GLOBAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef CheckWithCPU
    mulMatrixWithCpu(h_c_cpu, h_a, h_b, M, K, N);
    printf("Sample results:\n");
    for(int i=0;i<4;i++)
        printf("cpu[%d]: %5f, tc[%d]: %5f,gpu[%d]:%5f\n", i, h_c_cpu[i], i, h_c_cublas_tc[i], i, h_c_naive_cuda[i]);
#endif

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CHECK_CUDA(cudaDeviceReset())

}