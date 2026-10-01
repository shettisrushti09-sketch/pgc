#include <stdio.h>
#include <cuda_runtime.h>

#define N 4000
#define TILE_WIDTH 16

__global__ void matrixMulGPU(const float *A, const float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    printf("Allocating host memory (%dx%d)...\n", N, N);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    printf("Launching CUDA Kernel on GPU...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("\n--- CUDA EXECUTION COMPLETE ---\n");
    printf("Matrix Dimension: %d x %d\n", N, N);
    printf("Block Size: %dx%d\n", TILE_WIDTH, TILE_WIDTH);
    printf("Verification C[0][0]: %.2f\n", h_C[0]);
    printf("Execution Time: %.6f seconds\n", seconds);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}