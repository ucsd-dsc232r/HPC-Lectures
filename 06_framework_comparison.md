# Spark vs Ray: A Practical Comparison

## Learning Objectives

- Understand when to choose Spark vs Ray for different workloads
- Implement the same task in both frameworks
- Measure and compare performance characteristics
- Make informed framework selection decisions

---

## 1. Framework Philosophy

### Apache Spark

**Design Goal**: Unified analytics engine for large-scale data processing

```
Data Lake → ETL → Analytics → ML → Data Warehouse
         ←────── Spark handles all of this ──────→
```

**Strengths**:
- Optimized for batch data processing
- Strong SQL support (Spark SQL)
- Mature ecosystem (MLlib, GraphX, Structured Streaming)
- Automatic optimization (Catalyst optimizer)

**Model**: Data-parallel operations on distributed collections (RDDs, DataFrames)

### Ray

**Design Goal**: Simple, universal framework for distributed computing

```
                    ┌─ Training ─┐
Python Function → Ray Task → │  Serving  │ → Results
                    └─ Tuning  ─┘
```

**Strengths**:
- Simple Python-native API (`@ray.remote`)
- Flexible task and actor model
- Excellent for ML workloads (Ray Train, Ray Tune, Ray Serve)
- Lower latency for small tasks

**Model**: Task-parallel execution of arbitrary Python functions

---

## 2. API Comparison

### Simple Map Operation

**Spark**:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Data as RDD
data = spark.sparkContext.parallelize(range(1000000))
result = data.map(lambda x: x ** 2).collect()

# Data as DataFrame
df = spark.createDataFrame([(i,) for i in range(1000000)], ["value"])
result_df = df.select((df.value ** 2).alias("squared"))
```

**Ray**:
```python
import ray

ray.init()

@ray.remote
def square(x):
    return x ** 2

# Process in batches
futures = [square.remote(x) for x in range(1000000)]
result = ray.get(futures)  # Warning: 1M tasks is too many!

# Better: batch processing
@ray.remote
def square_batch(batch):
    return [x ** 2 for x in batch]

batches = [range(i, min(i+1000, 1000000)) for i in range(0, 1000000, 1000)]
futures = [square_batch.remote(batch) for batch in batches]
result = ray.get(futures)
```

### Key Difference

| Aspect | Spark | Ray |
|--------|-------|-----|
| Overhead per task | High (optimized for large batches) | Low (fine-grained tasks) |
| Ideal task size | 100K+ rows per partition | 1-1000 tasks typical |
| Scheduling | Batch-oriented | Fine-grained |

---

## 3. Hands-On Comparison Exercise

### Task: Distributed Aggregation

Compute the mean of a large dataset using both frameworks.

### Setup

```python
import numpy as np
import time

# Generate 100M random numbers
np.random.seed(42)
data = np.random.randn(100_000_000)

# Save to disk for fair comparison
np.save("large_array.npy", data)
```

### Spark Implementation

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean

spark = SparkSession.builder \
    .appName("Spark-Mean") \
    .config("spark.executor.instances", 7) \
    .config("spark.executor.memory", "16g") \
    .getOrCreate()

# Load and compute
start = time.time()

df = spark.createDataFrame(
    [(float(x),) for x in np.load("large_array.npy")],
    ["value"]
)
result = df.agg(mean("value")).collect()[0][0]

spark_time = time.time() - start
print(f"Spark Mean: {result:.6f} in {spark_time:.2f}s")
```

### Ray Implementation

```python
import ray
import numpy as np

ray.init()

@ray.remote
def compute_partial_sum(data_chunk):
    return np.sum(data_chunk), len(data_chunk)

start = time.time()

# Load data
data = np.load("large_array.npy")

# Split into chunks
num_chunks = 100
chunks = np.array_split(data, num_chunks)

# Distribute computation
futures = [compute_partial_sum.remote(chunk) for chunk in chunks]
results = ray.get(futures)

# Combine results
total_sum = sum(r[0] for r in results)
total_count = sum(r[1] for r in results)
result = total_sum / total_count

ray_time = time.time() - start
print(f"Ray Mean: {result:.6f} in {ray_time:.2f}s")
```

### Comparison Table Template

```markdown
## Framework Comparison Results

| Metric | Spark | Ray |
|--------|-------|-----|
| Execution Time | X.XX s | Y.YY s |
| Setup Overhead | High | Low |
| Lines of Code | N | M |
| Memory Usage | A GB | B GB |

### Analysis

[Your analysis here]
```

---

## 4. When to Choose Each Framework

### Choose Spark When...

| Scenario | Why Spark |
|----------|-----------|
| ETL pipelines | Optimized for data transformation |
| SQL queries on big data | Spark SQL with Catalyst optimizer |
| Batch processing | Designed for throughput over latency |
| Joining large datasets | Efficient distributed joins |
| Data warehousing | Native Parquet, Delta Lake support |
| Existing Hadoop ecosystem | HDFS, Hive integration |

### Choose Ray When...

| Scenario | Why Ray |
|----------|---------|
| ML model training | Ray Train with XGBoost, PyTorch, TensorFlow |
| Hyperparameter tuning | Ray Tune built-in |
| Model serving | Ray Serve for deployment |
| Reinforcement learning | RLlib |
| Stateful computations | Actor model |
| Low-latency tasks | Fine-grained scheduling |
| Custom parallelism | Flexible task model |

### Choose Both (Hybrid) When...

| Scenario | How to Combine |
|----------|----------------|
| ETL → ML Training | Spark for data prep → Ray for training |
| Feature engineering → Model | Spark SQL → Ray Train |
| Batch + Real-time | Spark Streaming + Ray Serve |

---

## 5. Practical Decision Matrix

Answer these questions to choose a framework:

```
1. Is your workload primarily data transformation (ETL)?
   └─ Yes → Spark
   └─ No → Continue

2. Do you need to run SQL queries on large datasets?
   └─ Yes → Spark
   └─ No → Continue

3. Are you training ML models with hyperparameter tuning?
   └─ Yes → Ray Train
   └─ No → Continue

4. Do you need model serving with low latency?
   └─ Yes → Ray Serve
   └─ No → Continue

5. Is your workload a mix of ETL and ML?
   └─ Yes → Both (Spark for ETL, Ray for ML)
   └─ No → Continue

6. Do you need stateful distributed objects (actors)?
   └─ Yes → Ray
   └─ No → Spark (default for batch processing)
```

---

## 6. Extra Credit Assignment: Framework Comparison (5 points)

### Requirements

Implement a data processing task using both Spark and Ray, then compare:

### Part 1: Implementation (2 points)

Choose ONE of these tasks:
- A) Compute statistics on your project dataset (mean, std, percentiles)
- B) Perform a group-by aggregation
- C) Train an XGBoost model on a subset of your data

Implement in both frameworks with equivalent functionality.

### Part 2: Performance Comparison (2 points)

Create a comparison table with:

| Metric | Spark | Ray |
|--------|-------|-----|
| Execution Time | | |
| Lines of Code | | |
| Memory Usage (peak) | | |
| Ease of Implementation (1-5) | | |

Include at least 3 runs to account for variance.

### Part 3: Analysis (1 point)

Answer these questions:
1. Which framework was faster? By how much?
2. Which was easier to implement? Why?
3. For your specific use case, which would you choose? Justify.

### Submission Format

Add a section to your README.md:

```markdown
## Framework Comparison (Extra Credit)

### Task Description
[Describe what you implemented]

### Spark Implementation
```python
# Your Spark code here
```

### Ray Implementation
```python
# Your Ray code here
```

### Results

| Metric | Spark | Ray |
|--------|-------|-----|
| Time (avg of 3 runs) | X.XX s | Y.YY s |
| Memory (peak) | A GB | B GB |
| Lines of Code | N | M |

### Analysis

[Your 3-part analysis here]
```

---

## 7. Common Pitfalls

### Spark Pitfalls

| Pitfall | Solution |
|---------|----------|
| Too many small tasks | Coalesce partitions |
| collect() on large data | Write to Parquet instead |
| Driver OOM | Keep driver memory small |
| Shuffle explosion | Use broadcast joins |

### Ray Pitfalls

| Pitfall | Solution |
|---------|----------|
| Too many fine-grained tasks | Batch your work |
| Not using ray.put() for large objects | Use object store |
| Forgetting ray.get() | Results stay in cluster |
| Actor deadlock | Use async methods |

---

## 8. Integration Patterns

### Pattern 1: Spark ETL → Ray Training

```python
# Step 1: Prepare data with Spark
spark_df = spark.read.parquet("raw_data")
processed = spark_df.filter(...).select(...).groupBy(...)
processed.write.parquet("processed_data")

# Step 2: Train with Ray
import ray
from ray.train.xgboost import XGBoostTrainer

trainer = XGBoostTrainer(
    label_column="target",
    datasets={"train": ray.data.read_parquet("processed_data")}
)
result = trainer.fit()
```

### Pattern 2: Using RayDP (Spark on Ray)

```python
import raydp

# Start Spark on Ray
spark = raydp.init_spark(
    app_name="RayDP",
    num_executors=4,
    executor_cores=2,
    executor_memory="4GB"
)

# Use Spark DataFrame API
df = spark.read.parquet("data.parquet")
processed = df.groupBy("category").count()

# Convert to Ray Dataset for ML
ray_ds = raydp.spark.spark_dataframe_to_ray_dataset(processed)

# Continue with Ray Train
```

---

## Summary

### Quick Reference

| Need | Use |
|------|-----|
| SQL on big data | Spark SQL |
| ETL pipelines | Spark DataFrame |
| ML training | Ray Train |
| Hyperparameter tuning | Ray Tune |
| Model serving | Ray Serve |
| Complex DAGs | Either (Spark for data, Ray for compute) |
| Low-latency tasks | Ray |
| Joining large tables | Spark |

### Key Takeaway

**Spark** excels at data processing at scale.
**Ray** excels at distributed Python and ML workloads.
**Together** they cover the full data science lifecycle.

---

*This module is part of DSC 232R: Big Data Analysis Using Spark at UCSD.*
