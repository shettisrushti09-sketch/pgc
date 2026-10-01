import matplotlib.pyplot as plt
import numpy as np

# Set clean academic style
plt.style.use('default')

# High-contrast academic color palette
colors = ['#1B365D', '#008080', '#D9534F', '#2E7D32']
hatch_patterns = ['///', '\\\\\\', '...', '***']

categories = [
    "Sequential CPU\n(Single-Threaded)",
    "MPI Cluster\n(4 VM Nodes)",
    "OpenMP\n(8 CPU Threads)",
    "CUDA Acceleration\n(NVIDIA GPU)"
]

times = [348.02, 92.98, 132.46, 0.165]
speedups = [1.0, 3.74, 2.63, 2109.18]

# Calculate GFLOPS: (2 * N^3) / (Time * 10^9) for N=4000
total_flops = 2 * (4000 ** 3)
gflops = [total_flops / (t * 1e9) for t in times]

# 1. Execution Time Chart
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

bars = ax.bar(categories, times, color=colors, edgecolor='#111111', linewidth=1.2, width=0.45)
for bar, hatch in zip(bars, hatch_patterns):
    bar.set_hatch(hatch)

ax.set_yscale('log')
ax.set_ylabel('Execution Time (Seconds, Log Scale)', fontsize=11, fontweight='bold', color='#111111')
ax.set_title('Matrix Multiplication (4000x4000) Execution Time Comparison', fontsize=13, fontweight='bold', pad=15)

for bar, val in zip(bars, times):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.3, f'{val:.2f} s' if val > 1 else f'{val:.3f} s', 
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#999999', alpha=0.95))

ax.grid(True, which="both", ls="--", lw=0.6, color="#D3D3D3")
ax.set_ylim(0.05, 1200)
plt.tight_layout()
plt.savefig('../images/execution_time_chart.png', dpi=300, facecolor='white')
plt.close()

# 2. Speedup Multiplier Chart
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

bars = ax.bar(categories, speedups, color=colors, edgecolor='#111111', linewidth=1.2, width=0.45)
for bar, hatch in zip(bars, hatch_patterns):
    bar.set_hatch(hatch)

ax.set_yscale('log')
ax.set_ylabel('Speedup Multiplier vs Sequential (Log Scale)', fontsize=11, fontweight='bold', color='#111111')
ax.set_title('Speedup Multiplier Relative to Sequential Baseline', fontsize=13, fontweight='bold', pad=15)

for bar, val in zip(bars, speedups):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.3, f'{val:.2f}x', 
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#999999', alpha=0.95))

ax.grid(True, which="both", ls="--", lw=0.6, color="#D3D3D3")
ax.set_ylim(0.5, 6000)
plt.tight_layout()
plt.savefig('../images/speedup_chart.png', dpi=300, facecolor='white')
plt.close()

# 3. GFLOPS Throughput Chart
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

bars = ax.bar(categories, gflops, color=colors, edgecolor='#111111', linewidth=1.2, width=0.45)
for bar, hatch in zip(bars, hatch_patterns):
    bar.set_hatch(hatch)

ax.set_yscale('log')
ax.set_ylabel('Computational Throughput (GFLOPS, Log Scale)', fontsize=11, fontweight='bold', color='#111111')
ax.set_title('Computational Throughput Performance (GFLOPS)', fontsize=13, fontweight='bold', pad=15)

for bar, val in zip(bars, gflops):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.3, f'{val:.2f} GFLOPS', 
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#999999', alpha=0.95))

ax.grid(True, which="both", ls="--", lw=0.6, color="#D3D3D3")
ax.set_ylim(0.1, 2000)
plt.tight_layout()
plt.savefig('../images/gflops_throughput_chart.png', dpi=300, facecolor='white')
plt.close()

# 4. Matrix Scaling Chart
matrix_sizes = [1000, 2000, 3000, 4000]
seq_scaling = [5.44, 43.50, 146.82, 348.02]
omp_scaling = [2.07, 16.56, 55.88, 132.46]
mpi_scaling = [1.45, 11.62, 39.22, 92.98]
cuda_scaling = [0.003, 0.021, 0.070, 0.165]

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

ax.plot(matrix_sizes, seq_scaling, marker='o', linewidth=2.2, color='#1B365D', label='Sequential CPU')
ax.plot(matrix_sizes, omp_scaling, marker='s', linewidth=2.2, color='#D9534F', label='OpenMP (8 Threads)')
ax.plot(matrix_sizes, mpi_scaling, marker='^', linewidth=2.2, color='#008080', label='MPI Cluster (4 Nodes)')
ax.plot(matrix_sizes, cuda_scaling, marker='d', linewidth=2.2, color='#2E7D32', label='CUDA GPU')

ax.set_yscale('log')
ax.set_xlabel('Matrix Dimension Size (N x N)', fontsize=11, fontweight='bold', color='#111111')
ax.set_ylabel('Execution Time (Seconds, Log Scale)', fontsize=11, fontweight='bold', color='#111111')
ax.set_title('Execution Time Scaling Across Problem Sizes', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(matrix_sizes)
ax.set_xticklabels(['1000x1000', '2000x2000', '3000x3000', '4000x4000'], fontweight='bold')
ax.legend(frameon=True, facecolor='white', edgecolor='#999999')
ax.grid(True, which="both", ls="--", lw=0.6, color="#D3D3D3")

plt.tight_layout()
plt.savefig('../images/matrix_scaling_chart.png', dpi=300, facecolor='white')
plt.close()

# 5. Combined Chart
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

x = np.arange(len(categories))
width = 0.35

ax.bar(x - width/2, times, width, label='Execution Time (s)', color='#1B365D', edgecolor='#111111', hatch='///')
ax.bar(x + width/2, speedups, width, label='Speedup Factor (x)', color='#D9534F', edgecolor='#111111', hatch='\\\\\\')

ax.set_yscale('log')
ax.set_ylabel('Logarithmic Scale', fontsize=11, fontweight='bold', color='#111111')
ax.set_title('Combined Performance & Speedup Benchmarks', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontweight='bold')
ax.legend(frameon=True, facecolor='white', edgecolor='#999999')
ax.grid(True, which="both", ls="--", lw=0.6, color="#D3D3D3")

plt.tight_layout()
plt.savefig('../images/performance_comparison_charts.png', dpi=300, facecolor='white')
plt.close()

print("All 5 performance charts generated and saved to images/")