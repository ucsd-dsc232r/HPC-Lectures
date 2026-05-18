# Measuring Speedup in Distributed Systems

## Key Takeaways

- **Speedup** measures how much faster parallel execution is compared to sequential
- **Efficiency** tells you how well you're utilizing your parallel resources
- **Practical measurement** validates theoretical expectations (Amdahl's Law)
- **Bottleneck identification** helps optimize your distributed applications

---

## 1. Why Measure Speedup?

### Theory vs Reality

In Module 1, you learned about Amdahl's Law:

$$S(n) = \frac{1}{(1-p) + \frac{p}{n}}$$

But theoretical speedup assumes:
- Perfect parallelization of the parallel portion
- Zero communication overhead
- No resource contention

**Real-world speedup is always lower.** Measuring it tells you:
1. How efficient your implementation actually is
2. Where bottlenecks exist
3. Whether adding more resources will help

---

## 2. Speedup and Efficiency Formulas

### Speedup

$$\text{Speedup}(n) = \frac{T_1}{T_n}$$

Where:
- $T_1$ = Execution time with 1 executor
- $T_n$ = Execution time with n executors

### Efficiency

$$\text{Efficiency}(n) = \frac{\text{Speedup}(n)}{n} = \frac{T_1}{n \times T_n}$$

### Interpretation

| Efficiency | Meaning |
|------------|---------|
| 100% | Perfect parallelization (theoretical ideal) |
| 70-90% | Good parallelization |
| 50-70% | Acceptable, room for improvement |
| < 50% | Significant overhead or bottlenecks |

---

## 3. Measuring Speedup in Spark

### Step 1: Create a Timing Function

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description: str):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.2f} seconds")
```

### Step 2: Baseline Measurement (Single Executor)

```python
# Configure Spark with 1 executor
spark_single = SparkSession.builder \
    .appName("Speedup-Baseline") \
    .config("spark.executor.instances", 1) \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Load your data
df = spark_single.read.parquet("your_data.parquet")

# Time your representative operation
with timer("Single executor"):
    result = df.groupBy("category").agg(
        {"value": "mean", "count": "sum"}
    ).collect()

# Record: T_1 = X seconds
spark_single.stop()
```

### Step 3: Parallel Measurement (Multiple Executors)

```python
# Configure Spark with n executors
n_executors = 7  # For 8 cores total

spark_parallel = SparkSession.builder \
    .appName("Speedup-Parallel") \
    .config("spark.executor.instances", n_executors) \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Load the same data
df = spark_parallel.read.parquet("your_data.parquet")

# Time the same operation
with timer(f"{n_executors} executors"):
    result = df.groupBy("category").agg(
        {"value": "mean", "count": "sum"}
    ).collect()

# Record: T_n = Y seconds
```

### Step 4: Reliable Measurement Methodology

Spark runs on the JVM, and JIT (Just-In-Time) compilation makes the first execution of any code path significantly slower than subsequent runs. Additionally, Spark's lazy evaluation means that transformations are not executed until an action (like `.count()` or `.collect()`) triggers them.

**Two rules for reliable timing:**

1. **Discard the first run** — JVM warmup inflates initial measurements
2. **Force execution inside the timer** — use `.count()` or `.collect()` to ensure Spark actually runs the computation

```python
def measure_with_warmup(operation, description, n_runs=3):
    """
    Measure execution time with JVM warmup handling.

    Runs the operation n_runs times, discards the first run,
    and returns the average of the remaining runs.
    """
    times = []
    for i in range(n_runs):
        start = time.time()
        operation()  # Must include an action like .count()
        elapsed = time.time() - start
        times.append(elapsed)
        label = "(warmup - discarded)" if i == 0 else ""
        print(f"  Run {i+1}: {elapsed:.2f}s {label}")

    # Discard first run, average the rest
    valid_times = times[1:]
    avg_time = sum(valid_times) / len(valid_times)
    print(f"  Average (excluding warmup): {avg_time:.2f}s")
    return avg_time

# Usage example
def my_pipeline():
    """Wrap your pipeline in a function that includes an action."""
    df.groupBy("category").agg(
        {"value": "mean", "count": "sum"}
    ).count()  # .count() forces execution

T_1 = measure_with_warmup(my_pipeline, "Single executor")
```

**Common mistake:** Timing a chain of transformations without an action at the end. Spark will return instantly because no computation actually occurred.

```python
# WRONG: This measures nothing — Spark is lazy!
start = time.time()
result = df.groupBy("category").agg({"value": "mean"})
elapsed = time.time() - start  # ~0.01s (just built a plan)

# RIGHT: Force execution with an action
start = time.time()
result = df.groupBy("category").agg({"value": "mean"}).count()
elapsed = time.time() - start  # ~25s (actually computed)
```

---

### Step 5: Calculate and Report

```python
# Your measurements
T_1 = 120.0  # seconds with 1 executor
T_7 = 25.0   # seconds with 7 executors
n = 7

speedup = T_1 / T_7
efficiency = speedup / n

print(f"Speedup: {speedup:.2f}x")
print(f"Efficiency: {efficiency:.1%}")

# Compare to theoretical maximum
# If p = 0.9 (90% parallelizable):
p = 0.9
theoretical_max = 1 / ((1 - p) + p / n)
print(f"Theoretical max (p={p}): {theoretical_max:.2f}x")
print(f"Achieved: {speedup/theoretical_max:.1%} of theoretical")
```

---

## 4. Recording Your Results

Create a speedup table in your README.md:

```markdown
## Speedup Analysis

| Executors | Time (sec) | Speedup | Efficiency |
|-----------|------------|---------|------------|
| 1         | 120.0      | 1.00x   | 100%       |
| 3         | 45.0       | 2.67x   | 89%        |
| 7         | 25.0       | 4.80x   | 69%        |

**Analysis:**
- Achieved 4.8x speedup with 7 executors (69% efficiency)
- Efficiency drops as we add executors, suggesting communication overhead
- Based on measurements, estimated parallelizable fraction: ~85%
```

---

## 5. Estimating Parallelizable Fraction

From your measurements, you can estimate what fraction of your code is actually parallelizable:

### Rearranging Amdahl's Law

$$p = \frac{n \times (S - 1)}{S \times (n - 1)}$$

Where:
- $p$ = parallelizable fraction
- $S$ = measured speedup
- $n$ = number of executors

### Example Calculation

```python
def estimate_parallel_fraction(speedup, n_executors):
    """Estimate parallelizable fraction from measured speedup."""
    S = speedup
    n = n_executors
    p = (n * (S - 1)) / (S * (n - 1))
    return p

# From our measurements: 4.8x speedup with 7 executors
p = estimate_parallel_fraction(4.8, 7)
print(f"Estimated parallelizable fraction: {p:.1%}")
# Output: Estimated parallelizable fraction: 86.7%
```

---

## 6. Weak vs Strong Scaling

Your MS3 speedup measurements use **strong scaling** — but understanding both scaling types helps you interpret results and plan for larger datasets.

### Strong Scaling (What MS3 Measures)

Fix the dataset size, increase executors. This answers: **"How much faster can I finish this job?"**

$$\text{Speedup}_{\text{strong}}(n) = \frac{T_1}{T_n} \quad \text{(same data throughout)}$$

Strong scaling hits diminishing returns because the serial fraction (Amdahl's Law) and communication overhead grow relative to the shrinking per-executor work.

### Weak Scaling

Grow the data proportionally with executors. This answers: **"Can I handle bigger data by adding resources?"**

$$\text{Efficiency}_{\text{weak}}(n) = \frac{T_1}{T_n} \quad \text{(data per executor stays constant)}$$

Ideal weak scaling: efficiency stays at 1.0 regardless of executor count.

### Comparison Table

| Property | Strong Scaling | Weak Scaling |
|----------|---------------|--------------|
| Data size | Fixed | Grows with executors |
| Goal | Faster time-to-solution | Handle larger problems |
| Ideal result | Linear speedup | Constant runtime |
| Limited by | Amdahl's Law (serial fraction) | Communication overhead |
| MS3 measurement | Yes — this is what you report | Optional but informative |

### When Each Matters

- **Strong scaling** is what matters when you have a fixed dataset (e.g., your project's 50GB Parquet files) and want to minimize wall-clock time.
- **Weak scaling** matters when you anticipate data growth — e.g., next quarter's data will be 2x larger, can you just add 2x executors?

### Example: Interpreting Your Results

If your strong scaling efficiency drops below 50% at 7 executors, consider:
1. Your dataset may be too small for 7 executors (each executor has too little work)
2. Communication overhead dominates at this scale
3. A weak scaling test would reveal whether bigger data improves efficiency

```python
# Weak scaling test: double data with double executors
configs = [
    (1, "data_10gb.parquet"),    # Baseline: 1 executor, 10 GB
    (2, "data_20gb.parquet"),    # 2 executors, 20 GB
    (4, "data_40gb.parquet"),    # 4 executors, 40 GB
    (7, "data_70gb.parquet"),    # 7 executors, 70 GB
]
# Ideal: all configs take roughly the same time
```

---

## 7. Common Causes of Low Efficiency

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Efficiency < 50% with few executors | Data skew | Repartition, salting |
| Efficiency drops sharply as n increases | Communication overhead | Reduce shuffles, use broadcast joins |
| Efficiency varies between runs | Resource contention | Use dedicated partition, avoid peak times |
| Single stage dominates runtime | Sequential bottleneck | Parallelize that stage or accept limit |

---

## 8. GC Overhead Awareness

Spark executors run on the JVM, which uses **garbage collection (GC)** to reclaim memory from objects no longer in use. GC pauses can add significant, unpredictable variance to your timing measurements.

### Checking GC Time in Spark UI

Navigate to **Stages** → Click on a completed stage → Look at the **GC Time** column in the task table.

| GC % of Task Time | Assessment | Action |
|-------------------|------------|--------|
| < 5% | Normal | No action needed |
| 5-10% | Elevated | Monitor; may affect timing consistency |
| > 10% | Problem | Increase executor memory or reduce data per executor |

### Why GC Matters for Speedup Measurements

GC pauses are **non-deterministic** — they happen when the JVM decides memory is running low. This means:
- The same job can produce different timings across runs
- GC pauses affect some executor configurations more than others (less memory per executor = more GC)
- Your speedup numbers may be unreliable if GC overhead varies between configurations

### Diagnosing GC Issues

```python
# Enable verbose GC logging to see pause details
spark = SparkSession.builder \
    .config("spark.executor.extraJavaOptions",
            "-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps") \
    .getOrCreate()
```

After running your job, check the executor logs for lines like:
```
[GC (Allocation Failure) 2048K->512K(8192K), 0.0032 secs]
[Full GC (Ergonomics) 6144K->2048K(8192K), 0.0891 secs]
```

**Full GC** pauses (>50ms) are the ones that affect your timing. If you see many Full GC events, your executors need more memory.

### Fix: Increase Executor Memory

```python
# If GC > 10% of task time, increase memory
spark = SparkSession.builder \
    .config("spark.executor.memory", "24g")  \  # Was 16g
    .config("spark.memory.fraction", "0.8")  \  # Default 0.6; use more for caching
    .getOrCreate()
```

---

## 9. Executor Configuration Tradeoffs

The number and size of executors significantly affects both performance and speedup measurements. "7 executors x 2 cores" is **not** the same as "14 executors x 1 core."

### Few-Large vs Many-Small Executors

| Property | Few Large Executors | Many Small Executors |
|----------|--------------------|--------------------|
| Cores/executor | 4-8 | 1-2 |
| Memory/executor | 16-64 GB | 2-8 GB |
| Broadcast efficiency | Better (fewer copies) | Worse (many copies) |
| GC overhead | Lower (larger heap) | Higher (small heap, frequent GC) |
| Shuffle overhead | Lower (fewer connections) | Higher (N² connections) |
| Task parallelism | Good | Maximum |
| HDFS throughput | Good (multi-threaded reads) | Poor (1 thread per executor) |

### Practical Formula for SDSC Expanse

For a typical SDSC allocation with `--cpus-per-task=32` and `--mem=128G`:

```python
# Recommended starting configuration
total_cores = 32     # From SLURM allocation
total_memory = 128   # GB, from SLURM allocation

cores_per_executor = 4  # Sweet spot: 4-5 cores each
driver_memory = 4       # GB, reserved for driver

num_executors = (total_cores - 1) // cores_per_executor  # Reserve 1 core for driver
memory_per_executor = (total_memory - driver_memory) // num_executors

spark = SparkSession.builder \
    .config("spark.executor.instances", num_executors) \
    .config("spark.executor.cores", cores_per_executor) \
    .config("spark.executor.memory", f"{memory_per_executor}g") \
    .config("spark.driver.memory", f"{driver_memory}g") \
    .getOrCreate()

# With 32 cores, 128 GB:
# num_executors = 7, cores_per_executor = 4, memory_per_executor = 17 GB
```

### Why 4-5 Cores Per Executor?

- **1 core**: Maximum number of executors but each one is single-threaded, poor HDFS I/O, high shuffle overhead
- **4-5 cores**: Good balance — multi-threaded reads, reasonable heap size, manageable shuffle connections
- **All cores in one executor**: Maximum memory but no parallelism across executors, one giant GC heap

### Impact on Your Speedup Measurements

When measuring speedup across executor configurations, **keep cores_per_executor constant** and vary `num_executors`. This isolates the effect of parallelism from the effect of executor sizing.

```python
# Good: Consistent executor size, varying count
configs = [
    {"executors": 1, "cores_per_exec": 4, "mem_per_exec": "64g"},
    {"executors": 3, "cores_per_exec": 4, "mem_per_exec": "20g"},
    {"executors": 7, "cores_per_exec": 4, "mem_per_exec": "8g"},
]
```

See [SPARK_HPC_BEST_PRACTICES.md](../group-project/SPARK_HPC_BEST_PRACTICES.md) for detailed SDSC memory planning guidance.

---

## 10. Visualizing Speedup

Include a speedup plot in your project:

```python
import matplotlib.pyplot as plt
import numpy as np

# Your measurements
executors = [1, 3, 5, 7]
times = [120, 45, 30, 25]
speedups = [times[0] / t for t in times]

# Theoretical curves
n_range = np.linspace(1, 8, 100)
ideal = n_range  # Linear speedup
amdahl_90 = 1 / ((1 - 0.9) + 0.9 / n_range)  # 90% parallel
amdahl_80 = 1 / ((1 - 0.8) + 0.8 / n_range)  # 80% parallel

plt.figure(figsize=(10, 6))
plt.plot(n_range, ideal, '--', label='Ideal (linear)', alpha=0.5)
plt.plot(n_range, amdahl_90, '--', label='Amdahl (p=0.9)', alpha=0.5)
plt.plot(n_range, amdahl_80, '--', label='Amdahl (p=0.8)', alpha=0.5)
plt.plot(executors, speedups, 'o-', label='Measured', markersize=10)

plt.xlabel('Number of Executors')
plt.ylabel('Speedup')
plt.title('Speedup Analysis: Measured vs Theoretical')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('speedup_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. Milestone 3 Requirements

For your project, you must include:

### Required (5 points)

1. **Baseline timing** (1 point)
   - Time a representative operation with 1 executor
   - Document the specific operation timed

2. **Parallel timing** (1 point)
   - Time the same operation with your full executor configuration
   - Use consistent data and operations

3. **Speedup calculation** (1 point)
   - Calculate speedup = T_1 / T_n
   - Calculate efficiency = speedup / n

4. **Results table** (1 point)
   - Present results in a clear table format
   - Include at least 2 different executor configurations

5. **Analysis** (1 point)
   - Estimate parallelizable fraction using Amdahl's Law
   - Explain any deviations from theoretical maximum
   - Identify potential bottlenecks

### Example README Section

```markdown
## Speedup Analysis

We measured the performance of our feature engineering pipeline
(data loading → preprocessing → aggregation) across different
executor configurations.

### Methodology
- Dataset: 50GB Parquet files
- Operation: Full preprocessing pipeline
- Each measurement: Average of 3 runs

### Results

| Executors | Memory/Exec | Time (sec) | Speedup | Efficiency |
|-----------|-------------|------------|---------|------------|
| 1         | 64GB        | 342        | 1.00x   | 100%       |
| 4         | 16GB        | 98         | 3.49x   | 87%        |
| 7         | 9GB         | 62         | 5.52x   | 79%        |
| 15        | 4GB         | 48         | 7.13x   | 48%        |

### Analysis

Using the formula p = n(S-1) / S(n-1):
- With 7 executors: p = 7(5.52-1) / 5.52(6) = 0.95 (95% parallelizable)

Efficiency drops to 48% at 15 executors, indicating:
1. Shuffle overhead becomes significant
2. Memory per executor (4GB) may be insufficient
3. Amdahl's Law limits with ~5% sequential code

**Recommendation**: 7 executors provides best balance of
speedup (5.5x) and efficiency (79%) for our workload.
```

---

## Summary

### Key Formulas

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Speedup | $S = T_1 / T_n$ | How many times faster |
| Efficiency | $E = S / n$ | Resource utilization |
| Parallelizable fraction | $p = n(S-1) / S(n-1)$ | What fraction can parallelize |
| Max speedup (Amdahl) | $S_{max} = 1 / (1-p)$ | Theoretical limit |

### Checklist for Your Project

- [ ] Timed baseline (single executor)
- [ ] Timed parallel execution
- [ ] Calculated speedup and efficiency
- [ ] Created results table
- [ ] Estimated parallelizable fraction
- [ ] Analyzed deviations from theoretical
- [ ] Included speedup plot (optional but recommended)

---

*Next: Understanding Communication Costs and Shuffle Optimization*
