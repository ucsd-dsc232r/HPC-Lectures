# Understanding Communication Costs in Spark

## Key Takeaways

- **Shuffles** are the most expensive operations in Spark
- **Data movement** across executors dominates distributed computing costs
- **Minimizing shuffles** is often more impactful than adding resources
- **Broadcast joins** can eliminate shuffles for small tables

---

## 1. What is a Shuffle?

A **shuffle** occurs when Spark needs to redistribute data across executors. This happens when the output of one stage requires data from multiple input partitions.

### Shuffle Anatomy

```
BEFORE SHUFFLE (Data distributed by input partitions)
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Executor 1  │  │ Executor 2  │  │ Executor 3  │
│ Partition 1 │  │ Partition 2 │  │ Partition 3 │
│ key: A,B,C  │  │ key: A,B,D  │  │ key: B,C,D  │
└─────────────┘  └─────────────┘  └─────────────┘
       │               │               │
       └───────────────┼───────────────┘
                       │
              ┌────────┴────────┐
              │     SHUFFLE     │
              │  (Network I/O)  │
              │  (Disk I/O)     │
              └────────┬────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Executor 1  │  │ Executor 2  │  │ Executor 3  │
│ All key: A  │  │ All key: B  │  │ All key:C,D │
└─────────────┘  └─────────────┘  └─────────────┘

AFTER SHUFFLE (Data distributed by key)
```

### The Cost of Shuffles

1. **Disk I/O**: Write intermediate data to local disk
2. **Network I/O**: Transfer data between executors
3. **Disk I/O again**: Read shuffled data from remote locations
4. **Serialization**: Convert objects to bytes and back
5. **Memory pressure**: Buffer data during transfer

---

## 2. Operations That Cause Shuffles

### Always Shuffle

| Operation | Why It Shuffles | Example |
|-----------|-----------------|---------|
| `groupBy()` | All rows with same key must be on same executor | `df.groupBy("category").count()` |
| `reduceByKey()` | Aggregation requires co-located keys | `rdd.reduceByKey(lambda a,b: a+b)` |
| `join()` | Matching keys must be co-located | `df1.join(df2, "key")` |
| `distinct()` | Must check all partitions for duplicates | `df.select("category").distinct()` |
| `repartition(n)` | Explicitly redistributes data | `df.repartition(100)` |
| `orderBy()` / `sort()` | Global ordering requires data exchange | `df.orderBy("timestamp")` |

### Conditional Shuffle

| Operation | Shuffles When... | Avoids Shuffle When... |
|-----------|------------------|------------------------|
| `coalesce(n)` | n > current partitions | n ≤ current partitions |
| `join()` | Both tables are large | One table is broadcast |
| `union()` | Never shuffles | Always append-only |

---

## 3. Identifying Shuffles in Spark UI

### Stage Boundaries = Shuffles

In the Spark UI, each **stage boundary** represents a shuffle:

```
Job 0
├── Stage 0: Read Parquet (no shuffle)
│   └── 50 tasks
├── Stage 1: GroupBy + Agg (SHUFFLE)
│   └── 200 tasks
└── Stage 2: Write Parquet (no shuffle)
    └── 200 tasks
```

### Shuffle Metrics to Monitor

Navigate to **Stages** → Click on a stage → **Shuffle Read/Write**

| Metric | What It Tells You |
|--------|-------------------|
| **Shuffle Write** | Data written to disk before network transfer |
| **Shuffle Read** | Data read from remote executors |
| **Shuffle Spill (Memory)** | Data that couldn't fit in memory |
| **Shuffle Spill (Disk)** | Spilled data written to disk |

### Red Flags

- **Shuffle Read >> Shuffle Write**: Data explosion during shuffle
- **Shuffle Spill > 0**: Memory pressure, consider more memory or fewer partitions
- **Single task much slower**: Data skew in shuffle

---

## 4. Data Skew Diagnosis and Salting Fix

**Data skew** is the most common cause of one slow task dragging down your entire stage. It occurs when some keys have far more rows than others, causing one executor to process a disproportionate share of the data.

### Detecting Skew in Spark UI

Navigate to **Stages** → Click on the slow stage → **Summary Metrics**

Look at the **Duration** row:

| Metric | Healthy | Skewed |
|--------|---------|--------|
| Min vs Max task duration | Max < 2x Min | Max > 3x Min |
| Median vs Max task duration | Similar | Max >> Median |
| Shuffle Read Size (Max vs Median) | Similar | Max >> Median |

**Rule of thumb:** If the Max task duration is more than 3x the Median, you have data skew.

### Why Skew Hurts Performance

```
Without skew:     With skew:
Task 1: ████ 10s  Task 1: ████████████████ 40s  ← Hot key
Task 2: ████ 10s  Task 2: ██ 5s
Task 3: ████ 10s  Task 3: ██ 5s
Task 4: ████ 10s  Task 4: ██ 5s
                   Tasks 2-4 WAIT for Task 1!
Total: 10s        Total: 40s (4x slower)
```

### The Salting Fix

**Salting** breaks hot keys into multiple sub-keys, spreading the load across executors:

```python
from pyspark.sql.functions import col, lit, concat, rand, floor

# Step 1: Identify skewed keys (optional diagnostic)
key_counts = df.groupBy("join_key").count().orderBy(col("count").desc())
key_counts.show(10)  # See which keys dominate

# Step 2: Salt the skewed DataFrame
num_salts = 10  # Split hot keys into 10 buckets

df_salted = df.withColumn(
    "salted_key",
    concat(col("join_key"), lit("_"), floor(rand() * num_salts).cast("int"))
)

# Step 3: Expand the other DataFrame to match all salt values
from pyspark.sql.functions import explode, array
import pyspark.sql.functions as F

salt_values = [lit(str(i)) for i in range(num_salts)]
other_df_salted = other_df.withColumn(
    "salt", explode(array([lit(i) for i in range(num_salts)]))
).withColumn(
    "salted_key",
    concat(col("join_key"), lit("_"), col("salt").cast("int"))
)

# Step 4: Join on salted key (skew is distributed!)
result = df_salted.join(other_df_salted, "salted_key")

# Step 5: Drop the salt columns
result = result.drop("salted_key", "salt")
```

### When Salting Helps vs Doesn't

| Scenario | Salting Helps? | Why |
|----------|---------------|-----|
| Few keys with many rows (e.g., "USA" in country column) | Yes | Distributes hot keys across executors |
| All keys have roughly equal rows | No | No skew to fix |
| Inherently large partitions (data is just big) | No | Need more partitions, not salting |
| Skew in `groupBy` aggregation | Yes | Salt, partial aggregate, then recombine |

For hands-on practice detecting skew in the Spark UI, see [Module 08: Spark UI Debugging Lab](08_spark_ui_debugging.md).

---

## 5. Minimizing Shuffle Costs

### Strategy 1: Use Broadcast Joins

When one table is small enough to fit in memory:

```python
from pyspark.sql.functions import broadcast

# BAD: Both tables shuffle
result = large_df.join(small_df, "key")

# GOOD: Small table broadcast to all executors
result = large_df.join(broadcast(small_df), "key")
```

**When to broadcast:**
- Small table < 10MB (default threshold)
- Can increase with `spark.sql.autoBroadcastJoinThreshold`

```python
# Increase broadcast threshold to 100MB
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 100 * 1024 * 1024)
```

### Join Strategy Analysis with explain()

Spark automatically selects a join strategy based on table sizes and statistics. Understanding which strategy was chosen helps you diagnose slow joins.

#### Join Strategy Reference

| Strategy | When Used | Shuffle? | Performance |
|----------|-----------|----------|-------------|
| **BroadcastHashJoin** | One side < broadcast threshold (10MB default) | No | Fastest |
| **SortMergeJoin** | Both sides large, equi-join | Yes (both sides) | Good for large tables |
| **ShuffledHashJoin** | One side much smaller (but > broadcast threshold) | Yes (both sides) | Moderate |
| **BroadcastNestedLoopJoin** | Non-equi join, one side small | No | Slow for large data |

#### Using explain() to See the Join Strategy

```python
# Check which join strategy Spark chose
result = large_df.join(medium_df, "key")
result.explain()
# Look for: BroadcastHashJoin, SortMergeJoin, or ShuffledHashJoin

# More detailed plan
result.explain(mode="extended")
```

#### Demo: Comparing Join Plans

```python
# Default: Spark auto-selects (likely BroadcastHashJoin for small table)
plan_auto = large_df.join(small_df, "key")
print("Auto-selected plan:")
plan_auto.explain()

# Force SortMergeJoin by disabling broadcast
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
plan_smj = large_df.join(small_df, "key")
print("Forced SortMergeJoin plan:")
plan_smj.explain()

# Restore default
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024)
```

In the explain output, look for the `Exchange` nodes — each one is a shuffle. `BroadcastHashJoin` will show `BroadcastExchange` (sending the small table) but no `Exchange` for the large table. `SortMergeJoin` will show `Exchange hashpartitioning` for both sides.

### Strategy 2: Reduce Before Shuffle

Aggregate or filter before operations that shuffle:

```python
# BAD: Shuffle all data, then filter
result = df.groupBy("category").agg({"value": "sum"}) \
           .filter(col("sum(value)") > 1000)

# BETTER: Filter early (if possible)
result = df.filter(col("value") > 0) \
           .groupBy("category").agg({"value": "sum"})
```

### Strategy 3: Use Coalesce Instead of Repartition

```python
# BAD: Full shuffle to reduce partitions
df.repartition(10).write.parquet("output")

# GOOD: No shuffle, just combine partitions locally
df.coalesce(10).write.parquet("output")
```

**Note**: `coalesce` can only reduce partitions, not increase them.

### Strategy 4: Partition by Key for Repeated Joins

If you join on the same key repeatedly:

```python
# Partition both DataFrames by join key
df1_partitioned = df1.repartition(100, "join_key")
df2_partitioned = df2.repartition(100, "join_key")

# Cache the partitioned versions
df1_partitioned.cache()
df2_partitioned.cache()

# Subsequent joins on "join_key" won't shuffle
result1 = df1_partitioned.join(df2_partitioned, "join_key")
result2 = df1_partitioned.join(df3_partitioned, "join_key")  # If df3 also partitioned
```

### Strategy 5: Avoid Unnecessary Sorts

```python
# BAD: Sort before groupBy (sort will be lost)
df.orderBy("timestamp").groupBy("category").count()

# GOOD: Just groupBy (no wasted sort)
df.groupBy("category").count()

# Only sort when needed for output
df.groupBy("category").count().orderBy("count", ascending=False)
```

---

## 6. Communication Cost Analysis

### Measuring Shuffle Cost

```python
def analyze_shuffle_cost(df, operation_name):
    """Analyze shuffle costs for a DataFrame operation."""

    # Get the query execution plan
    df.explain(mode="cost")

    # After execution, check Spark UI or use:
    spark.sparkContext.statusTracker()
```

### Shuffle Cost Estimation

For a `groupBy` operation:

$$\text{Shuffle Cost} \approx \text{Data Size} \times \text{Shuffle Ratio} \times \text{Network Overhead}$$

Where:
- **Data Size**: Size of columns involved in groupBy
- **Shuffle Ratio**: Typically 1.0-3.0 (data can expand with many groups)
- **Network Overhead**: ~1.5-2.0x for serialization and protocol

### Example Analysis

```python
# Estimate shuffle cost
data_size_gb = 50
columns_in_groupby = 3
total_columns = 20
shuffle_ratio = 1.5
network_overhead = 2.0

# Only columns used in groupBy + aggregation are shuffled
relevant_data = data_size_gb * (columns_in_groupby / total_columns)
estimated_shuffle = relevant_data * shuffle_ratio * network_overhead

print(f"Estimated shuffle: {estimated_shuffle:.1f} GB")
# With 1 Gbps network: {estimated_shuffle * 8:.0f} seconds minimum
```

---

## 7. Partition Strategy

### Too Few Partitions

- Underutilized parallelism
- Memory pressure on each executor
- Long-running tasks

### Too Many Partitions

- Scheduling overhead
- Small file problem
- Excessive shuffle overhead

### Rule of Thumb

```python
# Good starting point
num_partitions = num_executors * cores_per_executor * 2

# For shuffle operations
spark.conf.set("spark.sql.shuffle.partitions", num_partitions)
```

### Adaptive Query Execution (AQE) Deep Dive

AQE (Spark 3.0+) optimizes query execution at runtime based on actual data statistics, rather than relying on potentially stale or inaccurate pre-execution estimates. It is enabled by default in Spark 3.2+.

```python
spark.conf.set("spark.sql.adaptive.enabled", True)
```

#### AQE's Three Capabilities

**1. Auto-Coalesce Small Partitions**

After a shuffle, some partitions may be very small (e.g., 1 MB when the target is 64 MB). AQE automatically merges these small partitions to reduce scheduling overhead.

```python
# Control the target partition size after coalescing
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
# Default: 64 MB. Increase for fewer, larger partitions.
```

**2. Auto-Optimize Skew Joins**

AQE detects skewed partitions during a SortMergeJoin and automatically splits the oversized partition into smaller pieces, duplicating the corresponding partition from the other side.

```python
# Enable skew join optimization
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)

# A partition is "skewed" if it's larger than this threshold
# AND larger than the median partition × skewedPartitionFactor
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", 5)
```

This is essentially automatic salting — AQE handles it for you during SortMergeJoin operations.

**3. Runtime Join Strategy Switching**

If AQE discovers at runtime that one side of a join is actually small enough to broadcast (even though the pre-execution estimate said it was large), it switches from SortMergeJoin to BroadcastHashJoin.

```python
# AQE will switch to broadcast if a side is smaller than this after filtering
spark.conf.set("spark.sql.adaptive.autoBroadcastJoinThreshold", "10MB")
```

#### When AQE Helps vs Doesn't

| Scenario | AQE Helps? | Why |
|----------|-----------|-----|
| Many small post-shuffle partitions | Yes | Coalesces into fewer, larger partitions |
| Skewed SortMergeJoin | Yes | Auto-splits skewed partitions |
| Filter reduces table size dramatically | Yes | May switch to BroadcastHashJoin at runtime |
| Skewed groupBy aggregation | No | AQE skew handling only applies to joins |
| Data too large for broadcast in any case | No | No strategy switch possible |

#### Recommended AQE Configuration for Projects

```python
# Enable all AQE features (good defaults for most workloads)
spark.conf.set("spark.sql.adaptive.enabled", True)
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", True)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
```

---

## 8. Practical Exercise

### Task: Identify and Reduce Shuffles

Given this code:

```python
# Original code - how many shuffles?
result = df1.join(df2, "customer_id") \
            .join(df3, "product_id") \
            .groupBy("category") \
            .agg({"amount": "sum"}) \
            .orderBy("sum(amount)")
```

**Questions:**
1. How many shuffles occur?
2. Which operations cause shuffles?
3. How would you optimize this?

**Answer:**
1. 4 shuffles (join1, join2, groupBy, orderBy)
2. Both joins shuffle both tables, groupBy shuffles, orderBy shuffles
3. Optimizations:
   - Broadcast small lookup tables (df2, df3 if small)
   - Filter early if possible
   - Consider if orderBy is necessary

```python
# Optimized
result = df1.join(broadcast(df2), "customer_id") \
            .join(broadcast(df3), "product_id") \
            .groupBy("category") \
            .agg({"amount": "sum"}) \
            .orderBy("sum(amount)")
# Now: 2 shuffles (groupBy, orderBy) - eliminated 2 join shuffles
```

---

## 9. Spill Diagnosis and Fix

When Spark doesn't have enough memory to hold intermediate data during shuffles or aggregations, it **spills** data to disk. Spilling is a major performance killer — disk I/O is orders of magnitude slower than memory access.

### Finding Spill in Spark UI

Navigate to **Stages** → Click on a stage → **Summary Metrics** table

Look for these columns:
- **Spill (Memory)**: Amount of data that was serialized to spill
- **Spill (Disk)**: Amount of data actually written to disk

| Spill Amount | Severity | Impact |
|-------------|----------|--------|
| 0 | None | Ideal |
| < 1 GB | Minor | Slight slowdown, usually tolerable |
| 1-10 GB | Moderate | Noticeable performance degradation |
| > 10 GB | Severe | Major performance hit; likely OOM risk |

### Causes of Spill

1. **Insufficient executor memory**: The most common cause. Each executor doesn't have enough memory for its share of the data.
2. **Too few partitions**: Fewer partitions = more data per partition = more memory pressure per task.
3. **Data skew**: One partition is much larger than others, exceeding the memory available for that task.
4. **Large aggregation state**: `groupBy` with many groups or large intermediate results.

### Fixes (In Order of Priority)

**Fix 1: Increase shuffle partitions** (free — no more resources needed)

```python
# Default is 200; increase for large datasets
spark.conf.set("spark.sql.shuffle.partitions", 400)
# More partitions = less data per partition = less memory pressure
```

**Fix 2: Increase executor memory** (requires more SLURM allocation)

```python
spark = SparkSession.builder \
    .config("spark.executor.memory", "32g") \  # Was 16g
    .getOrCreate()
```

**Fix 3: Increase memory fraction for execution** (trade cache space for execution space)

```python
# Default: 0.6 (60% of executor memory for execution + storage)
# Increase if spilling but not using much caching
spark.conf.set("spark.memory.fraction", "0.8")
```

**Fix 4: Apply salting** (if spill is caused by data skew)

See [Section 4: Data Skew Diagnosis and Salting Fix](#4-data-skew-diagnosis-and-salting-fix) above.

### Monitoring Spill During Execution

```python
# After running a job, check stage details programmatically
# (Spark UI is easier, but this works in scripts)
sc = spark.sparkContext
for stage_info in sc.statusTracker().getActiveStageIds():
    print(f"Stage {stage_info}: check Spark UI for spill metrics")
```

For hands-on practice identifying spill metrics in the Spark UI, see [Module 08: Spark UI Debugging Lab](08_spark_ui_debugging.md).

---

## 10. Quick Reference

### Shuffle Operations Cheat Sheet

| Goal | Shuffle Operation | No-Shuffle Alternative |
|------|-------------------|----------------------|
| Reduce partitions | `repartition(n)` | `coalesce(n)` |
| Join with small table | `join()` | `broadcast()` join |
| Count distinct | `distinct().count()` | `approx_count_distinct()` |
| Get top N | `orderBy().limit(n)` | Window function if possible |
| Remove duplicates | `dropDuplicates()` | `dropDuplicates(subset)` on key |

### Configuration for Large Shuffles

```python
# Increase shuffle buffer
spark.conf.set("spark.shuffle.file.buffer", "64k")

# Increase memory fraction for shuffle
spark.conf.set("spark.shuffle.memoryFraction", "0.4")

# Compress shuffle data (CPU vs network tradeoff)
spark.conf.set("spark.shuffle.compress", True)

# Use external shuffle service (production)
spark.conf.set("spark.shuffle.service.enabled", True)
```

---

## Summary

### Key Points

1. **Shuffles are expensive** - disk I/O, network I/O, serialization
2. **Stage boundaries = shuffles** - count stages to count shuffles
3. **Broadcast small tables** - eliminate join shuffles
4. **Filter early** - reduce data before shuffles
5. **Use coalesce** - not repartition when reducing partitions
6. **Monitor Spark UI** - check shuffle read/write metrics

### Impact on Your Project

In your README.md, include:
- Number of shuffles in your pipeline
- Shuffle data volume (from Spark UI)
- Any optimizations you made to reduce shuffles

---

*This module is part of DSC 232R: Big Data Analysis Using Spark at UCSD.*
