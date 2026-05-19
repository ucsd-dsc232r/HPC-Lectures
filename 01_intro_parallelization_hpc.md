# Introduction to Parallelization and High-Performance Computing

## Key Takeaways

- **Parallelization** is essential when single-machine performance hits fundamental limits (memory, CPU, I/O)
- **Amdahl's Law** tells us the theoretical speedup limit based on the parallelizable fraction of code
- **HPC clusters** like SDSC Expanse provide massive compute resources (93,000+ cores, 5+ petaflops)
- **Different frameworks** (MPI, Spark, Ray) serve different parallelization needs

---

## Connecting to What You Know

### From Single Machine to Cluster

You've seen in **Class01** how numpy achieves 10,000x speedup over pure Python through optimized libraries. But what happens when even numpy isn't fast enough?

You've learned in **Class02** about memory hierarchy and how cache-friendly access patterns improve performance. But what happens when your data doesn't fit in RAM?

You've used **Spark** throughout this course to process data across multiple executors. Now we'll understand the broader landscape of parallel computing and where Spark fits.

---

## 1. Why Parallelization?

### The Single-Machine Wall

Even the most powerful single machine has limits:

| Resource | Typical Limit | What Happens When Exceeded |
|----------|---------------|---------------------------|
| **RAM** | 64-256 GB | Out-of-memory errors, excessive swapping |
| **CPU Cores** | 8-64 cores | Can't parallelize beyond core count |
| **Disk I/O** | ~500 MB/s (SSD) | Bottleneck on data loading |
| **Network** | 1-10 Gbps | Slow data transfer |

### Real-World Scale

Consider processing the entire NOAA weather dataset:
- **Size**: ~1 TB of historical weather data
- **Processing**: Feature extraction, PCA, model training
- **Single machine**: Days to weeks
- **HPC cluster**: Hours

---

## 2. Types of Parallelism

### Data Parallelism

**Same operation applied to different chunks of data**

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT DATA                            │
│  [Chunk 1] [Chunk 2] [Chunk 3] [Chunk 4] [Chunk 5]      │
└─────────────────────────────────────────────────────────┘
      │          │          │          │          │
      ▼          ▼          ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker 4 │ │Worker 5 │
│ f(x)    │ │ f(x)    │ │ f(x)    │ │ f(x)    │ │ f(x)    │
└─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
      │          │          │          │          │
      ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────┐
│                   COMBINED OUTPUT                        │
└─────────────────────────────────────────────────────────┘
```

**Examples**: Spark `map()`, distributed training batches

### Task Parallelism

**Different operations running concurrently**

```
┌──────────────────────────────────────────┐
│              INPUT                        │
└──────────────────────────────────────────┘
          │
          ├──────────────┬──────────────┐
          ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Task A   │   │ Task B   │   │ Task C   │
    │ (ETL)    │   │ (Train)  │   │ (Eval)   │
    └──────────┘   └──────────┘   └──────────┘
          │              │              │
          └──────────────┴──────────────┘
                         │
                         ▼
               ┌──────────────────┐
               │   FINAL OUTPUT   │
               └──────────────────┘
```

**Examples**: Ray tasks, pipeline stages

---

## 3. Amdahl's Law

### The Theoretical Limit

**Amdahl's Law** tells us the maximum speedup achievable by parallelizing a program:

$$S(n) = \frac{1}{(1-p) + \frac{p}{n}}$$

Where:
- $S(n)$ = Speedup with $n$ processors
- $p$ = Fraction of code that can be parallelized
- $n$ = Number of processors

### Implications

| Parallelizable Fraction (p) | Max Speedup (n→∞) |
|----------------------------|-------------------|
| 50% | 2x |
| 75% | 4x |
| 90% | 10x |
| 95% | 20x |
| 99% | 100x |

**Key Insight**: Even with infinite processors, if 5% of your code is sequential, you can never achieve more than 20x speedup!

### Gustafson's Law: A More Optimistic View

Gustafson observed that as we get more processors, we typically increase problem size:

$$S(n) = n - (1-p) \cdot (n-1)$$

This "scaled speedup" is more realistic for data-intensive workloads where we process more data with more resources.

---

## 4. What is HPC?

### High-Performance Computing Defined

**HPC** = Using parallel processing to solve computationally intensive problems faster than traditional computers.

### HPC Cluster Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         HEAD NODE                                │
│  - Job scheduling (SLURM)                                       │
│  - User login                                                    │
│  - File management                                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  COMPUTE NODE 1 │ │  COMPUTE NODE 2 │ │  COMPUTE NODE N │
    │  - 128 cores    │ │  - 128 cores    │ │  - 128 cores    │
    │  - 256 GB RAM   │ │  - 256 GB RAM   │ │  - 256 GB RAM   │
    │  - Local SSD    │ │  - Local SSD    │ │  - Local SSD    │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   PARALLEL FILESYSTEM                        │
    │  (Lustre: 12 PB storage, 140 GB/s bandwidth)                │
    └─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Head/Login Nodes**: Where users connect and submit jobs
2. **Compute Nodes**: Where actual computation happens
3. **High-Speed Interconnect**: Fast communication between nodes (InfiniBand)
4. **Parallel Filesystem**: Shared storage accessible from all nodes

---

## 5. SDSC Expanse: Our HPC Platform

### System Overview

| Component | Specification |
|-----------|--------------|
| **Standard Nodes** | 728 nodes |
| **Processors** | Dual AMD EPYC 7742 (64 cores each = 128 cores/node) |
| **Memory** | 256 GB DDR4 per node |
| **Total Cores** | 93,184 compute cores |
| **Peak Performance** | 5.16 petaflops |
| **GPU Nodes** | 54 nodes with 4x NVIDIA V100 (32 GB) each |
| **Storage** | 12 PB Lustre (140 GB/s) |
| **Interconnect** | HDR InfiniBand (200 Gbps) |

### Putting It in Perspective

Your laptop vs SDSC Expanse:

| Metric | Typical Laptop | Expanse | Ratio |
|--------|---------------|---------|-------|
| CPU Cores | 8 | 93,184 | 11,648x |
| RAM | 16 GB | 186 TB | 11,625x |
| Storage | 512 GB | 12 PB | 24,576x |

### Expanse Partitions

| Partition | Max Nodes | Max Time | Use Case |
|-----------|-----------|----------|----------|
| `debug` | 2 | 30 min | Testing, development |
| `shared` | 1 | 48 hrs | Small jobs (<128 cores) |
| `compute` | 32 | 48 hrs | Large parallel jobs |
| `gpu` | 4 | 48 hrs | GPU workloads |
| `gpu-shared` | 1 | 48 hrs | Small GPU jobs |

---

## 6. Parallelization Frameworks Comparison

### When to Use What

| Framework | Best For | Programming Model | Fault Tolerance |
|-----------|----------|-------------------|-----------------|
| **MPI** | Tightly-coupled HPC, simulations | Message passing | Limited |
| **Spark** | Batch data processing, ETL | RDD/DataFrame | Automatic (lineage) |
| **Ray** | ML workloads, flexible tasks | Tasks/Actors | Automatic |

### Design Philosophy Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                           MPI                                    │
│  - Explicit communication (send/receive)                        │
│  - Maximum control and performance                              │
│  - Steep learning curve                                         │
│  - Best for: Physics simulations, scientific computing          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          SPARK                                   │
│  - Data-parallel operations (map, reduce, join)                 │
│  - Optimized for batch processing                               │
│  - Strong SQL/DataFrame support                                 │
│  - Best for: ETL, data warehousing, batch analytics             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                           RAY                                    │
│  - Simple API (@ray.remote decorator)                           │
│  - Flexible task and actor model                                │
│  - Native Python objects                                        │
│  - Best for: ML training, reinforcement learning, serving       │
└─────────────────────────────────────────────────────────────────┘
```

### Code Style Comparison

**Parallel Sum - Three Ways**

**MPI (C-style, explicit)**:
```python
# Pseudocode - actual MPI is more complex
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
local_sum = sum(my_data_chunk)
total = comm.reduce(local_sum, op=MPI.SUM, root=0)
```

**Spark (Data-parallel)**:
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
total = spark.sparkContext.parallelize(data).reduce(lambda a, b: a + b)
```

**Ray (Task-parallel)**:
```python
import ray
ray.init()

@ray.remote
def sum_chunk(chunk):
    return sum(chunk)

chunks = [data[i:i+1000] for i in range(0, len(data), 1000)]
futures = [sum_chunk.remote(chunk) for chunk in chunks]
total = sum(ray.get(futures))
```

---

## 7. The Road Ahead

### This Week's Journey

```
YOU ARE HERE
     │
     ▼
┌─────────────────────────────────────────┐
│ Module 1: Parallelization & HPC         │ ← Current
│ - Why parallel?                         │
│ - HPC architecture                      │
│ - SDSC Expanse                          │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ Module 2: SLURM & Spark on SDSC         │
│ - Job scheduling                        │
│ - Spark on HPC                          │
│ - Singularity containers                │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ Module 3: Introduction to Ray           │
│ - Tasks, Actors, Objects                │
│ - Ray Data                              │
│ - Ray Train                             │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ Module 4: Ray + Spark Integration       │
│ - RayDP                                 │
│ - Spark vs Ray comparison               │
│ - Data handoff patterns                 │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ Module 5: Ray on SLURM                  │
│ - Production deployment                 │
│ - Multi-node clusters                   │
└─────────────────────────────────────────┘
```

---

## Summary

### Key Concepts

1. **Parallelization** overcomes single-machine limits by distributing work
2. **Data parallelism** applies the same operation to different data chunks
3. **Task parallelism** runs different operations concurrently
4. **Amdahl's Law** sets the theoretical speedup limit
5. **HPC clusters** provide massive compute resources (thousands of cores)
6. **Different frameworks** (MPI, Spark, Ray) serve different use cases

### Connecting to DSC 232R

| Prior Topic | Connection |
|-------------|------------|
| Class01: Numpy performance | Even optimized code needs parallelization at scale |
| Class02: Memory hierarchy | HPC extends memory hierarchy across machines |
| Spark basics | Spark is one parallelization framework; Ray is another |

---

## Practice Problems

1. **Amdahl's Law Calculation**: If 80% of your code is parallelizable, what's the maximum speedup with 100 processors? With 1000 processors?

2. **Framework Selection**: You need to train 100 independent machine learning models on different hyperparameter configurations. Which framework would you choose and why?

3. **Resource Estimation**: You have a 50 GB dataset that requires 2x memory overhead. How many Expanse nodes (256 GB each) would you need to process it entirely in memory?

---

## Further Reading

- Patterson & Hennessy, "Computer Architecture: A Quantitative Approach", Chapter 6 (Parallel Processors)
- SDSC Expanse User Guide: https://www.sdsc.edu/systems/expanse/user_guide.html
- Ray Documentation: https://docs.ray.io/en/latest/

---

*Next: Module 2 - SLURM and SDSC Usage with Spark*
