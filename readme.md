# Parallel Matrix Multiplication using Sequential, OpenMP, MPI and CUDA

## 📌 Project Overview

This project implements **matrix multiplication** using four different approaches:

1. **Sequential execution** – traditional single-threaded implementation
2. **OpenMP** – shared-memory parallel execution using multiple CPU threads
3. **MPI** – distributed-memory parallel execution using multiple processes
4. **CUDA** – GPU-based parallel execution using CUDA

The main objective is to compare the **execution time, speedup, scalability, and performance** of sequential and parallel matrix multiplication approaches.

---

## 🎯 Objectives

* Implement matrix multiplication using sequential execution.
* Parallelize matrix multiplication using **OpenMP**.
* Implement distributed matrix multiplication using **MPI**.
* Implement GPU-based matrix multiplication using **CUDA**.
* Measure and compare execution times.
* Analyze speedup and scalability.
* Compare the performance of CPU-based and GPU-based approaches.
* Visualize the obtained performance results using graphs.

---

## 📂 Project Structure

```text
1pgc-Parallel_matrix_multiplication/
│
├── images/
│   ├── execution_time_chart.png
│   ├── gflops_throughput_chart.png
│   ├── matrix_scaling_chart.png
│   ├── mpi_ping.png
│   ├── mpi_result.png
│   ├── mpi_send_recv.png
│   ├── open mp.png
│   ├── openmp_htop.png
│   ├── performance_comparison_charts.png
│   ├── sequential.png
│   └── speedup_chart.png
│
├── src/
│   ├── cuda/
│   │   └── matrix_cuda.cu
│   │
│   ├── mpi/
│   │   ├── matrix_mpi.c
│   │   └── mpi_send_recv.c
│   │
│   ├── openmp/
│   │   └── matrix_openmp.c
│   │
│   └── sequential/
│       └── matrix_sequential.c
│
└── generate_charts.py
```

---

# 🧠 Implementations

## 1. Sequential Matrix Multiplication

The sequential implementation performs matrix multiplication using a single CPU thread.

The basic operation is:

```text
C[i][j] = Σ A[i][k] × B[k][j]
```

This implementation is used as the **baseline** for comparing the performance of the parallel implementations.

### Source File

```text
src/sequential/matrix_sequential.c
```

### Execution Output

![Sequential Execution](images/sequential.png)

---

# 2. OpenMP Matrix Multiplication

OpenMP is used to parallelize matrix multiplication across multiple CPU threads.

Instead of performing all iterations sequentially, independent iterations of the matrix multiplication are distributed among multiple threads.

### Source File

```text
src/openmp/matrix_openmp.c
```

### OpenMP Execution

![OpenMP Output](images/open%20mp.png)

### CPU Utilization

The CPU utilization during OpenMP execution can be observed using `htop`.

![OpenMP htop](images/openmp_htop.png)

OpenMP demonstrates how a shared-memory system can use multiple CPU cores to reduce computation time.

---

# 3. MPI Matrix Multiplication

MPI (**Message Passing Interface**) is used to perform matrix multiplication using multiple processes.

The matrix computation can be distributed among different processes. Each process performs a portion of the computation and communicates results using MPI communication functions.

### Source Files

```text
src/mpi/matrix_mpi.c
src/mpi/mpi_send_recv.c
```

### MPI Communication

The project also demonstrates MPI point-to-point communication using send and receive operations.

![MPI Send Receive](images/mpi_send_recv.png)

### MPI Ping

MPI communication can be tested using a ping-style communication between processes.

![MPI Ping](images/mpi_ping.png)

### MPI Result

![MPI Result](images/mpi_result.png)

MPI demonstrates how computation can be distributed across multiple processes using message passing.

---

# 4. CUDA Matrix Multiplication

CUDA is used to execute matrix multiplication on a GPU.

CUDA follows a highly parallel execution model where thousands of GPU threads can execute operations concurrently.

### Source File

```text
src/cuda/matrix_cuda.cu
```

The CUDA implementation is designed to exploit the parallel processing capabilities of GPUs for matrix multiplication.

---

# 📊 Performance Comparison

The implementations are compared using several performance metrics:

* Execution time
* Speedup
* GFLOPS throughput
* Matrix-size scalability
* Overall performance

---

## ⏱️ Execution Time Comparison

Execution time measures how long each implementation takes to complete matrix multiplication.

Lower execution time indicates faster completion for the tested configuration.

![Execution Time Comparison](images/execution_time_chart.png)

---

## 🚀 Speedup Comparison

Speedup compares the performance of a parallel implementation with the sequential baseline.

The general formula is:

```text
Speedup = Sequential Execution Time / Parallel Execution Time
```

A higher speedup indicates a greater reduction in execution time compared with the sequential implementation.

![Speedup Comparison](images/speedup_chart.png)

---

## ⚡ GFLOPS Throughput

GFLOPS (**Giga Floating Point Operations Per Second**) represents the number of billions of floating-point operations performed per second.

It provides a way to compare computational throughput between different implementations.

![GFLOPS Throughput](images/gflops_throughput_chart.png)

---

## 📈 Matrix Scaling

Matrix scaling shows how the execution time/performance changes as the matrix size increases.

Larger matrices provide more computation and therefore help demonstrate how well each implementation scales.

![Matrix Scaling](images/matrix_scaling_chart.png)

---

# 📊 Overall Performance Comparison

The collected results can be summarized using the performance comparison charts.

![Performance Comparison](images/performance_comparison_charts.png)

The comparison considers the behavior of:

| Implementation | Execution Model              | Main Parallelism         |
| -------------- | ---------------------------- | ------------------------ |
| Sequential     | Single CPU thread            | None                     |
| OpenMP         | Shared-memory CPU            | Multiple CPU threads     |
| MPI            | Distributed-memory processes | Multiple processes       |
| CUDA           | GPU                          | Thousands of GPU threads |

---

# 🔍 Comparison of Approaches

| Feature                   | Sequential                 | OpenMP         | MPI                 | CUDA                             |
| ------------------------- | -------------------------- | -------------- | ------------------- | -------------------------------- |
| Processing Unit           | CPU                        | CPU            | CPU / Processes     | GPU                              |
| Parallelism               | ❌                          | ✅              | ✅                   | ✅                                |
| Memory Model              | Shared                     | Shared         | Distributed         | GPU Memory                       |
| Communication             | None                       | Thread-based   | Message passing     | Host-GPU                         |
| Scalability               | Low                        | Moderate       | High                | Very High for suitable workloads |
| Implementation Complexity | Low                        | Moderate       | Higher              | Higher                           |
| Best Use Case             | Baseline / small workloads | Multi-core CPU | Distributed systems | Highly parallel workloads        |

---

# 🧪 Experiments

The project evaluates the implementations using different matrix sizes and execution configurations.

The following performance characteristics are analyzed:

### 1. Execution Time

Measures the time required to complete matrix multiplication.

### 2. Speedup

Measures the improvement over the sequential implementation.

```text
Speedup = T_sequential / T_parallel
```

### 3. Throughput

Measured using GFLOPS to determine computational throughput.

### 4. Scalability

The matrix size is increased to observe how the implementations behave as computational workload increases.

### 5. CPU Utilization

OpenMP execution is monitored to observe CPU-core utilization.

---

# 📁 Source Code

### Sequential

```text
src/sequential/matrix_sequential.c
```

### OpenMP

```text
src/openmp/matrix_openmp.c
```

### MPI

```text
src/mpi/matrix_mpi.c
src/mpi/mpi_send_recv.c
```

### CUDA

```text
src/cuda/matrix_cuda.cu
```

---

# 📊 Chart Generation

Performance graphs are generated using:

```text
generate_charts.py
```

The generated charts include:

* Execution time comparison
* Speedup comparison
* GFLOPS throughput
* Matrix scaling
* Overall performance comparison

---

# 🏁 Conclusion

This project demonstrates four different approaches to matrix multiplication: **Sequential, OpenMP, MPI, and CUDA**.

The sequential implementation provides a baseline for performance comparison. OpenMP demonstrates shared-memory CPU parallelism, while MPI demonstrates distributed-memory computation and process-to-process communication. CUDA demonstrates GPU-based parallel computation.

The performance graphs help analyze how execution time, speedup, throughput, and scalability change across different implementations and matrix sizes.

Overall, the project provides a practical comparison of **serial execution, CPU parallelism, distributed processing, and GPU acceleration** for a computationally intensive problem such as matrix multiplication.

---

# 🛠️ Technologies Used

* **C**
* **OpenMP**
* **MPI**
* **CUDA**
* **Python**
* **Matplotlib**
* **Linux / Ubuntu**
* **Git & GitHub**

---

# 👩‍💻 Project

**Parallel Matrix Multiplication using Sequential, OpenMP, MPI and CUDA**

The project demonstrates how the same computational problem can be implemented using different parallel computing paradigms and evaluated through experimental performance analysis.
