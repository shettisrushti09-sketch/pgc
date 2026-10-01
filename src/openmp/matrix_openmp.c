#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000

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
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    int threads = omp_get_max_threads();
    printf("Starting OpenMP Parallel Matrix Multiplication with %d threads...\n", threads);

    double start_time = omp_get_wtime();


    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    printf("\n--- OPENMP EXECUTION COMPLETE ---\n");
    printf("Matrix Dimension: %d x %d\n", N, N);
    printf("Threads Used: %d\n", threads);
    printf("Verification C[0][0]: %.2f\n", C[0]);
    printf("Execution Time: %.6f seconds\n", execution_time);

    free(A);
    free(B);
    free(C);
    return 0;
}