# Parallel & GPU Computing Lab — 4000×4000 Matrix Multiplication

## Executive Summary

This repository presents a comparative performance analysis of **4000×4000 single-precision floating-point matrix multiplication** across four high-performance computing architectures:

- **Sequential C** — single-threaded CPU baseline
- **OpenMP** — shared-memory CPU parallelism using 8 threads
- **MPI** — distributed-memory execution across a 4-node/VM cluster
- **CUDA** — GPU acceleration using an NVIDIA RTX GPU

The study evaluates execution time, relative speedup, and floating-point throughput (GFLOPS), while verifying the computed result across implementations.

For `N = 4000`, dense matrix multiplication requires approximately:

```text
2 × N³ = 2 × 4000³ = 128,000,000,000 floating-point operations
```

All implementations report:

```text
C[0][0] = 4000.00
```

---

# 1. Objectives

The objectives of this laboratory experiment are to:

1. Implement matrix multiplication using sequential C.
2. Parallelize the computation using OpenMP.
3. Distribute the workload using MPI.
4. Accelerate matrix multiplication using CUDA.
5. Benchmark the four implementations under the same workload.
6. Compare execution time, speedup, and GFLOPS.
7. Verify correctness.
8. Visualize the measured performance.

---

# 2. Computational Problem

The benchmark computes:

```text
C = A × B
```

where:

```text
A = 4000 × 4000 float matrix
B = 4000 × 4000 float matrix
C = 4000 × 4000 float matrix
```

Each element is calculated as:

```text
C[i][j] = Σ A[i][k] × B[k][j]
```

The matrices use **single-precision floating-point (`float`)** values.

---

# 3. Architectures

| Implementation | Architecture | Parallelism | Configuration |
|---|---|---|---|
| Sequential C | CPU | None | 1 thread |
| OpenMP | Shared-memory CPU | Threads | 8 CPU threads |
| MPI | Distributed memory | Processes | 4 nodes / VMs |
| CUDA | GPU | CUDA threads | NVIDIA RTX GPU |

---

# 4. Benchmark Results

## Comprehensive Performance Table

| Implementation | Execution Time (s) | Speedup | GFLOPS | Verification |
|---|---:|---:|---:|---|
| Sequential C | 348.02 | 1.00× | 0.37 | `C[0][0] = 4000.00` |
| OpenMP — 8 Threads | 132.46 | 2.63× | 0.97 | `C[0][0] = 4000.00` |
| MPI — 4 Nodes | 92.98 | 3.74× | 1.38 | `C[0][0] = 4000.00` |
| CUDA — NVIDIA RTX | 0.165 | 2109.18× | 775.74 | `C[0][0] = 4000.00` |

---

# 5. 📊 Visual Performance Comparison

## Overall Performance Comparison

![Complete Performance Comparison](images/performance_comparison_charts.png)

The consolidated performance figure provides an overall comparison of the four implementations.

It summarizes the differences in:

- Execution time
- Speedup
- GFLOPS throughput
- Architectural performance

---

## Execution Time Comparison

![Execution Time Chart](images/execution_time_chart.png)

### Measured execution times

```text
Sequential : 348.02 s
OpenMP     : 132.46 s
MPI        :  92.98 s
CUDA       :   0.165 s
```

The execution-time graph shows the reduction in wall-clock time obtained through increasing levels of parallelism.

---

## Speedup Comparison

![Speedup Chart](images/speedup_chart.png)

Speedup is calculated relative to the sequential implementation:

```text
Speedup = Sequential Time / Implementation Time
```

Measured values:

```text
Sequential :     1.00×
OpenMP     :     2.63×
MPI        :     3.74×
CUDA       :  2109.18×
```

---

## GFLOPS Throughput Comparison

![GFLOPS Throughput Chart](images/gflops_throughput_chart.png)

Measured computational throughput:

```text
Sequential :   0.37 GFLOPS
OpenMP     :   0.97 GFLOPS
MPI        :   1.38 GFLOPS
CUDA       : 775.74 GFLOPS
```

---

## Matrix Scaling Comparison

![Matrix Scaling Chart](images/matrix_scaling_chart.png)

The matrix-scaling visualization illustrates performance behavior as matrix dimensions increase.

---

# 6. Performance Analysis

## Sequential C

The sequential implementation establishes the baseline:

```text
Execution Time = 348.02 s
Speedup        = 1.00×
GFLOPS         = 0.37
```

It executes the matrix multiplication using a single CPU execution flow.

---

## OpenMP

The OpenMP implementation uses **8 CPU threads**:

```text
Execution Time = 132.46 s
Speedup        = 2.63×
GFLOPS         = 0.97
```

The workload is divided among multiple CPU threads operating in a shared-memory environment.

---

## MPI

The MPI implementation uses a **4-node/VM distributed configuration**:

```text
Execution Time = 92.98 s
Speedup        = 3.74×
GFLOPS         = 1.38
```

MPI demonstrates distributed-memory parallelism, where processes communicate using message passing.

---

## CUDA

The CUDA implementation runs on an **NVIDIA RTX GPU**:

```text
Execution Time = 0.165 s
Speedup        = 2109.18×
GFLOPS         = 775.74
```

The GPU implementation exploits massive fine-grained parallelism by executing many matrix operations concurrently.

---

# 7. 🖥️ Output Verification Screenshots

## Sequential C Output

![Sequential Output](images/sequential.png)

The sequential implementation verifies:

```text
C[0][0] = 4000.00
```

---

## OpenMP Output

![OpenMP Output](images/openmp.png)

The OpenMP implementation verifies:

```text
C[0][0] = 4000.00
```

with the benchmark configured for 8 threads.

---

## OpenMP CPU Utilization

![OpenMP htop](images/openmp_htop.png)

The `htop` screenshot provides visual evidence of CPU utilization during the OpenMP execution.

---

## MPI Cluster Connectivity

![MPI Ping](images/mpi_ping.png)

The MPI connectivity screenshot demonstrates communication between the configured nodes/VMs.

---

## MPI Send/Receive Communication

![MPI Send Receive](images/mpi_send_recv.png)

This screenshot demonstrates MPI message-passing activity.

---

## MPI Computation Result

![MPI Result](images/mpi_result.png)

The MPI computation produces:

```text
C[0][0] = 4000.00
```

---

# 8. Source Code & Execution

## 8.1 Sequential C

### Source

```text
src/sequential/
```

### Compilation

```bash
gcc -O2 src/sequential/matrix_mul.c -o sequential
```

### Execution

```bash
./sequential
```

### Expected output

```text
C[0][0] = 4000.00
Execution Time: 348.02 seconds
Performance: 0.37 GFLOPS
```

---

# 9. OpenMP

### Source

```text
src/openmp/
```

### Compilation

```bash
gcc -O2 -fopenmp src/openmp/matrix_mul.c -o openmp
```

### Configure 8 threads

```bash
export OMP_NUM_THREADS=8
```

### Execution

```bash
./openmp
```

### Expected output

```text
C[0][0] = 4000.00
Execution Time: 132.46 seconds
Performance: 0.97 GFLOPS
```

---

# 10. MPI

### Source

```text
src/mpi/
```

### Compilation

```bash
mpicc -O2 src/mpi/matrix_mul.c -o mpi_matrix_mul
```

### Four-process execution

```bash
mpirun -np 4 ./mpi_matrix_mul
```

### Multi-node execution

```bash
mpirun --hostfile hosts -np 4 ./mpi_matrix_mul
```

### Expected output

```text
C[0][0] = 4000.00
Execution Time: 92.98 seconds
Performance: 1.38 GFLOPS
```

---

# 11. CUDA

### Source

```text
src/cuda/
```

### Compilation

```bash
nvcc -O2 src/cuda/matrix_mul.cu -o cuda_matrix_mul
```

### Execution

```bash
./cuda_matrix_mul
```

### Expected output

```text
C[0][0] = 4000.00
Execution Time: 0.165 seconds
Performance: 775.74 GFLOPS
```

---

# 12. Correctness Verification

All four implementations produce the same benchmark verification value:

| Implementation | Verification |
|---|---|
| Sequential | `C[0][0] = 4000.00` |
| OpenMP | `C[0][0] = 4000.00` |
| MPI | `C[0][0] = 4000.00` |
| CUDA | `C[0][0] = 4000.00` |

This provides a basic cross-implementation correctness check.

> **Note:** Checking one matrix element is a basic verification method. A complete numerical validation would compare the entire result matrix against a trusted reference using an appropriate floating-point tolerance.

---

# 13. GFLOPS Calculation

For dense matrix multiplication:

```text
FLOPs ≈ 2 × N³
```

For `N = 4000`:

```text
FLOPs = 2 × 4000³
      = 128,000,000,000 FLOPs
      = 128 GFLOP
```

Therefore:

```text
GFLOPS = FLOPs / (Execution Time × 10⁹)
```

---

# 14. Speedup Calculation

The sequential implementation is the baseline:

```text
Speedup = T_sequential / T_implementation
```

For OpenMP:

```text
348.02 / 132.46 ≈ 2.63×
```

For MPI:

```text
348.02 / 92.98 ≈ 3.74×
```

For CUDA:

```text
348.02 / 0.165 ≈ 2109.18×
```

---

# 15. Architecture Comparison

| Feature | Sequential | OpenMP | MPI | CUDA |
|---|---|---|---|---|
| Processing Unit | CPU | CPU | Multiple CPUs/VMs | GPU |
| Memory Model | Single process | Shared memory | Distributed memory | GPU memory |
| Parallelism | None | Threads | Processes | CUDA threads |
| Configuration | 1 thread | 8 threads | 4 nodes | NVIDIA RTX |
| Communication | None | Shared memory | Message passing | Host ↔ GPU |
| Compiler | GCC | GCC | MPICC | NVCC |
| Speedup | 1.00× | 2.63× | 3.74× | 2109.18× |
| GFLOPS | 0.37 | 0.97 | 1.38 | 775.74 |

---

# 16. Benchmark Methodology

The same computational workload is evaluated across all four implementations:

```text
Matrix A : 4000 × 4000
Matrix B : 4000 × 4000
Matrix C : 4000 × 4000
Precision: float
Operation: C = A × B
```

The following metrics are recorded:

1. Execution time
2. Speedup
3. GFLOPS throughput
4. Correctness verification

The sequential implementation serves as the baseline for speedup.

---

# 17. Chart Generation

The charts can be generated using:

```bash
python3 scripts/generate_charts.py
```

Generated charts are stored under:

```text
images/
```

---

# 18. Repository Directory Structure

```text
pgc/
├── README.md
├── .gitignore
├── images/
│   ├── execution_time_chart.png
│   ├── speedup_chart.png
│   ├── gflops_throughput_chart.png
│   ├── matrix_scaling_chart.png
│   ├── performance_comparison_charts.png
│   ├── sequential.png
│   ├── openmp.png
│   ├── openmp_htop.png
│   ├── mpi_ping.png
│   ├── mpi_send_recv.png
│   └── mpi_result.png
├── scripts/
│   └── generate_charts.py
└── src/
    ├── cuda/
    ├── mpi/
    ├── openmp/
    └── sequential/
```

---

# 19. Key Findings

The benchmark demonstrates the following progression:

```text
Sequential CPU
      ↓
OpenMP Shared-Memory CPU
      ↓
MPI Distributed-Memory Computing
      ↓
CUDA GPU Acceleration
```

The measured results are:

```text
                    Time       Speedup       GFLOPS
Sequential          348.02 s    1.00×          0.37
OpenMP              132.46 s    2.63×          0.97
MPI                  92.98 s    3.74×          1.38
CUDA                  0.165 s   2109.18×     775.74
```

The results illustrate how exploiting different forms of parallelism can substantially change the performance of a computationally intensive workload.

---

# 20. Conclusion

This laboratory experiment compares four approaches to high-performance matrix multiplication: sequential CPU execution, shared-memory CPU parallelism using OpenMP, distributed-memory parallelism using MPI, and GPU acceleration using CUDA.

The benchmark demonstrates the architectural differences between these approaches and quantifies their impact using execution time, speedup, and GFLOPS.

For the tested `4000 × 4000` single-precision workload, the measured CUDA implementation achieves the highest throughput and the lowest execution time among the evaluated implementations.

---

## Technologies Used

- C
- OpenMP
- MPI
- CUDA
- GCC
- MPICC
- NVCC
- Python
- Linux / Ubuntu
- Virtual Machines
- Git & GitHub
