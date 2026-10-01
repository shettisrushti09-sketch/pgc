# Parallel & GPU Computing Lab — 4000×4000 Matrix Multiplication

## Executive Summary

This project benchmarks **4000×4000 single-precision floating-point matrix multiplication** across four high-performance computing paradigms:

1. **Sequential C** — baseline single-threaded implementation.
2. **OpenMP** — shared-memory parallel implementation using **8 CPU threads**.
3. **MPI** — distributed-memory implementation using a **4-node cluster / VM configuration**.
4. **CUDA** — GPU-accelerated implementation targeting an **NVIDIA RTX GPU**.

The objective is to compare execution time, parallel speedup, and floating-point throughput (**GFLOPS**) across different architectures.

For the benchmark configuration, the matrices contain 16 million elements each, and the matrix multiplication performs approximately \(2N^3 = 128\) billion floating-point operations for `N = 4000`. All implementations produce the same verification value:

```text
C[0][0] = 4000.00
```

The results demonstrate the performance differences between conventional sequential execution, shared-memory CPU parallelism, distributed-memory computation, and GPU acceleration.

---

## 1. Project Objectives

The primary objectives of this lab are to:

- Implement matrix multiplication using multiple computing paradigms.
- Establish a sequential C implementation as the performance baseline.
- Parallelize the workload using OpenMP.
- Distribute computation across multiple processes/nodes using MPI.
- Accelerate matrix multiplication using CUDA on an NVIDIA RTX GPU.
- Measure and compare execution time and throughput.
- Calculate relative speedup against the sequential baseline.
- Verify numerical correctness across all implementations.
- Visualize the performance differences using benchmark charts.

---

## 2. System Configuration

| Parameter | Configuration |
|---|---|
| Matrix Size | `4000 × 4000` |
| Data Type | Single-precision `float` |
| Operation | Dense Matrix Multiplication |
| Baseline | Sequential C |
| OpenMP Configuration | 8 CPU threads |
| MPI Configuration | 4-node cluster / VMs |
| GPU Configuration | NVIDIA RTX GPU |
| Primary Metrics | Execution Time, Speedup, GFLOPS |
| Verification | `C[0][0] = 4000.00` |

### Matrix Multiplication

The computation is:

```text
C = A × B
```

with each output element calculated as:

```text
C[i][j] = Σ A[i][k] × B[k][j]
```

For `N = 4000`, the theoretical floating-point operation count is approximately:

```text
2 × N³ = 2 × 4000³ = 128,000,000,000 FLOPs
```

---

## 3. Architectural Implementations

### 3.1 Sequential C — Baseline

The sequential implementation performs the complete matrix multiplication using a single CPU execution flow.

**Purpose:**

- Establish the reference execution time.
- Provide the baseline for calculating speedup.
- Verify the correctness of the matrix multiplication.

Typical compilation:

```bash
gcc -O2 src/sequential/matrix_mul.c -o sequential
```

Execution:

```bash
./sequential
```

Expected verification:

```text
C[0][0] = 4000.00
```

### Output Verification

![Sequential Output](images/sequential.png)

---

### 3.2 OpenMP — Shared-Memory Parallelism

OpenMP parallelizes the matrix multiplication across multiple CPU threads sharing the same memory space.

The benchmark uses:

```text
8 CPU Threads
```

Typical compilation:

```bash
gcc -O2 -fopenmp src/openmp/matrix_mul_openmp.c -o openmp
```

Execution:

```bash
export OMP_NUM_THREADS=8
./openmp
```

Expected verification:

```text
C[0][0] = 4000.00
```

### Output Verification

![OpenMP Output](images/openmp.png)

### CPU Thread Monitoring

![OpenMP htop](images/openmp_htop.png)

The `htop` output provides a visual confirmation of CPU-thread utilization during the OpenMP execution.

---

### 3.3 MPI — Distributed-Memory Parallelism

The MPI implementation distributes portions of the matrix multiplication workload among multiple MPI processes.

Benchmark configuration:

```text
4 Nodes / VMs
```

Typical compilation:

```bash
mpicc -O2 src/mpi/matrix_mul_mpi.c -o mpi_matmul
```

Execution:

```bash
mpirun -np 4 ./mpi_matmul
```

Depending on the cluster configuration, the hostfile can be specified with:

```bash
mpirun --hostfile hosts -np 4 ./mpi_matmul
```

Expected verification:

```text
C[0][0] = 4000.00
```

### MPI Communication Verification

![MPI Ping](images/mpi_ping.png)

![MPI Send Receive](images/mpi_send_recv.png)

### MPI Result

![MPI Result](images/mpi_result.png)

MPI demonstrates how computational workloads can be distributed across independent memory spaces and connected through message passing.

---

### 3.4 CUDA — GPU Acceleration

The CUDA implementation executes matrix multiplication on an NVIDIA RTX GPU.

Typical compilation:

```bash
nvcc -O2 src/cuda/matrix_mul_cuda.cu -o cuda_matmul
```

Execution:

```bash
./cuda_matmul
```

Expected verification:

```text
C[0][0] = 4000.00
```

The GPU implementation exploits massive parallelism by assigning matrix computation to CUDA threads organized into blocks and grids.

---

## 4. Performance Benchmarks

The following benchmark results were obtained for:

```text
Matrix Size: 4000 × 4000
Data Type:   Single-precision float
```

| Implementation | Architecture | Configuration | Execution Time (s) | Speedup | GFLOPS | Verification |
|---|---|---:|---:|---:|---:|---:|
| Sequential C | Single CPU | 1 thread | 348.02 | 1.00× | 0.37 | `4000.00` |
| OpenMP | Shared-memory CPU | 8 threads | 132.46 | 2.63× | 0.97 | `4000.00` |
| MPI | Distributed memory | 4 nodes / VMs | 92.98 | 3.74× | 1.38 | `4000.00` |
| CUDA | NVIDIA GPU | RTX GPU | 0.165 | 2109.18× | 775.74 | `4000.00` |

### Performance Interpretation

#### Sequential C

The sequential implementation requires:

```text
348.02 seconds
```

and establishes the baseline:

```text
1.00× speedup
0.37 GFLOPS
```

#### OpenMP

Using eight CPU threads reduces execution time to:

```text
132.46 seconds
```

corresponding to:

```text
2.63× speedup
0.97 GFLOPS
```

This demonstrates the benefit of shared-memory CPU parallelism, although the speedup is not perfectly linear with the number of threads.

#### MPI

The distributed implementation achieves:

```text
92.98 seconds
3.74× speedup
1.38 GFLOPS
```

The improvement comes from distributing the computational workload among multiple MPI processes/nodes. Communication and synchronization overhead remain important considerations.

#### CUDA

The CUDA implementation completes the computation in:

```text
0.165 seconds
```

with:

```text
2109.18× speedup
775.74 GFLOPS
```

The substantially higher throughput results from the massive parallel execution capability of the GPU architecture.

---

## 5. Performance Analysis

### 5.1 Execution Time

Execution time decreases progressively from the sequential implementation to OpenMP, MPI, and CUDA.

```text
Sequential  →  OpenMP  →  MPI  →  CUDA
348.02 s       132.46 s   92.98 s   0.165 s
```

The CUDA implementation provides a particularly large reduction in execution time for this matrix size.

---

### 5.2 Speedup

Speedup is calculated relative to the sequential baseline:

```text
Speedup = Sequential Execution Time / Parallel Execution Time
```

The measured speedups are:

```text
Sequential : 1.00×
OpenMP     : 2.63×
MPI        : 3.74×
CUDA       : 2109.18×
```

---

### 5.3 GFLOPS Throughput

GFLOPS represents billions of floating-point operations completed per second.

The measured throughput is:

```text
Sequential : 0.37 GFLOPS
OpenMP     : 0.97 GFLOPS
MPI        : 1.38 GFLOPS
CUDA       : 775.74 GFLOPS
```

The CUDA implementation therefore demonstrates substantially higher computational throughput for the tested workload.

---

## 6. Visualizations

### Execution Time

![Execution Time Chart](images/execution_time_chart.png)

### Speedup

![Speedup Chart](images/speedup_chart.png)

### GFLOPS Throughput

![GFLOPS Throughput Chart](images/gflops_throughput_chart.png)

### Matrix Scaling

![Matrix Scaling Chart](images/matrix_scaling_chart.png)

### Overall Performance Comparison

![Performance Comparison Charts](images/performance_comparison_charts.png)

These visualizations provide graphical comparisons of execution time, relative speedup, computational throughput, and matrix-size scaling behavior.

---

## 7. Source Code and Execution Commands

### Sequential

**Source:**

```text
src/sequential/
```

**Compile:**

```bash
gcc -O2 src/sequential/matrix_mul.c -o sequential
```

**Run:**

```bash
./sequential
```

---

### OpenMP

**Source:**

```text
src/openmp/
```

**Compile:**

```bash
gcc -O2 -fopenmp src/openmp/matrix_mul_openmp.c -o openmp
```

**Run with 8 threads:**

```bash
export OMP_NUM_THREADS=8
./openmp
```

---

### MPI

**Source:**

```text
src/mpi/
```

**Compile:**

```bash
mpicc -O2 src/mpi/matrix_mul_mpi.c -o mpi_matmul
```

**Run using four processes:**

```bash
mpirun -np 4 ./mpi_matmul
```

**Run using a hostfile:**

```bash
mpirun --hostfile hosts -np 4 ./mpi_matmul
```

---

### CUDA

**Source:**

```text
src/cuda/
```

**Compile:**

```bash
nvcc -O2 src/cuda/matrix_mul_cuda.cu -o cuda_matmul
```

**Run:**

```bash
./cuda_matmul
```

---

## 8. Correctness Verification

All four implementations produced the expected verification value:

```text
C[0][0] = 4000.00
```

This common result provides a basic correctness check that the implementations are computing the same matrix multiplication result.

| Implementation | Verification |
|---|---|
| Sequential | `C[0][0] = 4000.00` |
| OpenMP | `C[0][0] = 4000.00` |
| MPI | `C[0][0] = 4000.00` |
| CUDA | `C[0][0] = 4000.00` |

> **Note:** A single output element is a basic verification check. For production-grade numerical validation, the complete output matrix should be compared against a trusted reference using an appropriate floating-point tolerance.

---

## 9. Benchmark Methodology

To make the comparison meaningful, the same fundamental workload is evaluated across all implementations:

```text
N = 4000
A = 4000 × 4000 float matrix
B = 4000 × 4000 float matrix
C = 4000 × 4000 float matrix
```

Each implementation performs:

```text
C = A × B
```

The principal measurements are:

### Execution Time

The elapsed wall-clock time required to complete the matrix multiplication.

### Speedup

```text
Speedup = T_sequential / T_implementation
```

where:

- `T_sequential` is the sequential execution time.
- `T_implementation` is the execution time of the implementation being evaluated.

### GFLOPS

For dense matrix multiplication:

```text
FLOPs ≈ 2 × N³
```

and:

```text
GFLOPS = FLOPs / (Execution Time × 10⁹)
```

---

## 10. Comparative Discussion

The benchmark illustrates four distinct parallel-computing models.

| Paradigm | Memory Model | Main Parallelism Mechanism | Key Characteristic |
|---|---|---|---|
| Sequential C | Single memory space | None | Baseline |
| OpenMP | Shared memory | CPU threads | Low programming overhead |
| MPI | Distributed memory | Processes + message passing | Scales across nodes |
| CUDA | GPU memory hierarchy | GPU threads/blocks | Massive fine-grained parallelism |

### Sequential vs OpenMP

OpenMP improves performance by dividing loop iterations among multiple CPU threads. Because the threads share memory, explicit data communication between threads is relatively straightforward.

### OpenMP vs MPI

MPI allows computation to be distributed across independent processes and potentially different physical or virtual machines. This makes MPI suitable for cluster-based computing, but communication and synchronization introduce additional overhead.

### CPU Parallelism vs GPU Parallelism

CPU architectures provide a smaller number of powerful general-purpose cores, while GPUs provide a very large number of parallel execution units optimized for highly parallel workloads.

Matrix multiplication is particularly suitable for GPU acceleration because many output elements can be computed independently.

---

## 11. Repository Structure

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
    │   └── matrix_mul_cuda.cu
    ├── mpi/
    │   └── matrix_mul_mpi.c
    ├── openmp/
    │   └── matrix_mul_openmp.c
    └── sequential/
        └── matrix_mul.c
```

---

## 12. Chart Generation

The benchmark visualizations can be generated using:

```bash
python3 scripts/generate_charts.py
```

The script is responsible for generating the performance charts stored under:

```text
images/
```

---

## 13. Compilation Flags

| Flag / Command | Purpose |
|---|---|
| `-O2` | Enables compiler optimization |
| `-fopenmp` | Enables OpenMP support in GCC |
| `gcc` | Compiles C source code |
| `mpicc` | MPI C compiler wrapper |
| `mpirun` | Launches MPI processes |
| `nvcc` | NVIDIA CUDA compiler |
| `OMP_NUM_THREADS=8` | Sets OpenMP thread count |

---

## 14. Key Findings

The benchmark results show a clear progression in performance:

```text
Sequential
    ↓
OpenMP shared-memory parallelism
    ↓
MPI distributed-memory parallelism
    ↓
CUDA GPU acceleration
```

For the tested `4000 × 4000` single-precision workload:

- Sequential execution establishes the baseline at **348.02 s**.
- OpenMP reduces execution time to **132.46 s** using 8 CPU threads.
- MPI reduces execution time further to **92.98 s** using 4 nodes/VMs.
- CUDA completes the computation in **0.165 s** on the NVIDIA RTX GPU.
- All implementations report the verification value **`C[0][0] = 4000.00`**.

The results demonstrate how architectural characteristics and parallel execution models affect computational performance.

---

## 15. Conclusion

This lab provides a comparative study of matrix multiplication across sequential CPU execution, shared-memory CPU parallelism, distributed-memory computing, and GPU acceleration.

The experiment highlights the importance of selecting an appropriate computing architecture for a workload. Matrix multiplication contains substantial data-level parallelism, making it particularly suitable for highly parallel architectures such as GPUs.

The benchmark also demonstrates that performance should be evaluated using multiple metrics rather than execution time alone. **Execution time, speedup, throughput, scalability, communication overhead, and correctness** together provide a more complete understanding of a high-performance computing implementation.

---

## 16. Technologies Used

- **C**
- **OpenMP**
- **MPI**
- **CUDA**
- **GCC**
- **NVIDIA CUDA Toolkit**
- **Python**
- **Git & GitHub**
- **Linux / Ubuntu**
- **Virtual Machines / Cluster Environment**

---

## 17. Author / Lab Repository

This repository was developed as part of a **Parallel & GPU Computing Lab** study of high-performance matrix multiplication.

The project demonstrates practical implementation and benchmarking of:

```text
CPU → Multithreading → Distributed Computing → GPU Computing
```

