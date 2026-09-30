#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 4000

double double_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int main() {
    printf("Allocating matrices (%dx%d)...\n", N, N);
    
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    printf("Starting Sequential Matrix Multiplication...\n");
    double start_time = double_time();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    double end_time = double_time();
    double execution_time = end_time - start_time;

    printf("\n--- SEQUENTIAL EXECUTION COMPLETE ---\n");
    printf("Matrix Dimension: %d x %d\n", N, N);
    printf("Verification C[0][0]: %.2f\n", C[0]);
    printf("Execution Time: %.6f seconds\n", execution_time);

    free(A);
    free(B);
    free(C);
    return 0;
}