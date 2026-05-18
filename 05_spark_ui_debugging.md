# Spark UI Debugging Lab

## Learning Objectives

By the end of this lab, you will be able to:
- Navigate the Spark UI to identify performance bottlenecks
- Interpret stage timelines and task distributions
- Identify data skew and shuffle problems
- Use the SQL tab for query plan analysis
- Take meaningful screenshots for your project documentation

---

## 1. Accessing the Spark UI

### On SDSC Expanse (JupyterLab)

When you create a SparkSession, the Spark UI is available at:

```python
# After creating SparkSession
print(f"Spark UI: {spark.sparkContext.uiWebUrl}")
```

The UI is typically at: `http://localhost:4040`

### Key Tabs Overview

| Tab | What It Shows | When to Use |
|-----|--------------|-------------|
| **Jobs** | Overall job progress | Check job completion |
| **Stages** | Detailed stage info | Identify slow stages |
| **Storage** | Cached RDDs/DataFrames | Verify caching |
| **Environment** | Spark configuration | Debug config issues |
| **Executors** | Per-executor metrics | Identify executor problems |
| **SQL** | Query plans | Understand query execution |

---

## 2. Exercise 1: Understanding Job and Stage Structure

### Setup Code

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum as spark_sum

# Create session with enough executors to see distribution
spark = SparkSession.builder \
    .appName("SparkUI-Lab") \
    .config("spark.executor.instances", 4) \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Create sample data
data = [(i, f"category_{i % 10}", i * 1.5, i % 100)
        for i in range(1000000)]
df = spark.createDataFrame(data, ["id", "category", "value", "group_id"])
```

### Run This Query

```python
# This query will create multiple stages
result = df.filter(col("value") > 100) \
           .groupBy("category") \
           .agg(
               count("*").alias("count"),
               avg("value").alias("avg_value")
           ) \
           .orderBy("count", ascending=False)

result.show()
```

### What to Observe in Spark UI

**Jobs Tab:**
1. Click on the job that just ran
2. Note how many stages it has
3. Each stage boundary = a shuffle

**Stages Tab:**
1. Find the stage with the longest duration
2. Click on it to see task details
3. Look at the "Event Timeline"

### Screenshot 1: Stage Summary

Capture a screenshot showing:
- All stages for your job
- Duration of each stage
- Shuffle read/write for each stage

**Question:** Which stage took the longest? Why?

---

## 3. Exercise 2: Identifying Data Skew

### Create Skewed Data

```python
import random

# Create intentionally skewed data
# 90% of rows have category_0, 10% distributed among others
skewed_data = []
for i in range(1000000):
    if random.random() < 0.9:
        category = "category_0"  # Hot key
    else:
        category = f"category_{random.randint(1, 99)}"
    skewed_data.append((i, category, i * 1.5))

skewed_df = spark.createDataFrame(skewed_data, ["id", "category", "value"])
```

### Run Aggregation on Skewed Data

```python
# This will show skew in the Spark UI
result = skewed_df.groupBy("category") \
                  .agg(count("*").alias("count")) \
                  .orderBy("count", ascending=False)

result.show()
```

### Identifying Skew in Spark UI

**Stages Tab → Click on GroupBy Stage → Summary Metrics:**

Look for these signs of skew:
- **Task Duration**: Max >> Median
- **Shuffle Read Size**: Max >> Median
- **GC Time**: High on specific tasks

**Event Timeline:**
- One task bar much longer than others
- Stragglers that hold up the whole stage

### Screenshot 2: Data Skew Evidence

Capture a screenshot showing:
- Task duration distribution (histogram or timeline)
- Summary metrics with Max vs Median comparison

**Question:** What is the ratio of Max to Median task duration? What does this indicate?

---

## 4. Exercise 3: Comparing Join Strategies

### Setup Tables

```python
# Large table (1M rows)
large_df = spark.createDataFrame(
    [(i, f"value_{i}", i % 1000) for i in range(1000000)],
    ["id", "data", "lookup_key"]
)

# Small lookup table (1000 rows)
small_df = spark.createDataFrame(
    [(i, f"lookup_{i}") for i in range(1000)],
    ["lookup_key", "lookup_value"]
)
```

### Compare Two Join Approaches

```python
# Approach A: Regular join (will shuffle)
result_a = large_df.join(small_df, "lookup_key")
result_a.count()  # Force execution
```

```python
from pyspark.sql.functions import broadcast

# Approach B: Broadcast join (no shuffle on large table)
result_b = large_df.join(broadcast(small_df), "lookup_key")
result_b.count()  # Force execution
```

### Analyzing in SQL Tab

**SQL Tab → Click on the query:**

For each approach, examine:
1. **Physical Plan**: Look for "BroadcastHashJoin" vs "SortMergeJoin"
2. **Stage count**: Broadcast should have fewer stages
3. **Shuffle bytes**: Broadcast should shuffle less

### Screenshot 3: Query Plan Comparison

Capture screenshots showing:
- The physical plan for the regular join
- The physical plan for the broadcast join
- Shuffle metrics for both

**Question:** How much data was shuffled in each approach?

---

## 5. Exercise 4: Memory and Spill Analysis

### Create Memory Pressure

```python
# Wide DataFrame that may cause memory pressure
wide_data = [(i, *[f"col_{j}_{i}" for j in range(50)])
             for i in range(500000)]
columns = ["id"] + [f"col_{j}" for j in range(50)]

wide_df = spark.createDataFrame(wide_data, columns)

# Aggregation that requires significant memory
result = wide_df.groupBy("col_0") \
                .agg(*[count(f"col_{j}").alias(f"count_{j}")
                       for j in range(1, 50)])

result.show()
```

### Checking for Memory Spill

**Stages Tab → Click on Stage → Summary Metrics:**

Look for:
- **Spill (Memory)**: Data that couldn't fit in memory
- **Spill (Disk)**: Spilled data written to disk

**Executors Tab:**
- Memory usage per executor
- Disk spill per executor

### Screenshot 4: Memory Metrics

Capture a screenshot showing:
- Memory usage across executors
- Any spill metrics (Memory and Disk)

**Question:** Did any spilling occur? What would you change to reduce it?

---

## 6. Exercise 5: Executor Analysis

### Executors Tab Deep Dive

Navigate to the **Executors** tab and examine:

| Metric | What to Look For |
|--------|-----------------|
| **RDD Blocks** | Cached data distribution |
| **Storage Memory** | Memory used for caching |
| **Disk Used** | Spill to disk |
| **Active Tasks** | Should be balanced |
| **Failed Tasks** | Should be 0 |
| **Task Time** | Should be similar across executors |

### Healthy vs Unhealthy Patterns

**Healthy:**
- Similar task time across executors
- Low or no GC time
- No failed tasks
- Balanced storage memory usage

**Unhealthy:**
- One executor with much higher task time (skew)
- High GC time (memory pressure)
- Failed tasks (errors)
- Unbalanced memory usage

### Screenshot 5: Executor Summary

Capture a screenshot of the Executors tab showing:
- All active executors
- Task distribution
- Memory usage

---

## 7. Debugging Checklist

Use this checklist when debugging Spark performance:

### Stage-Level Analysis
- [ ] Identify the slowest stage
- [ ] Check shuffle read/write size
- [ ] Look for spill (memory/disk)
- [ ] Compare task durations (Max vs Median)

### Task-Level Analysis
- [ ] Check for stragglers (long-running tasks)
- [ ] Look at task distribution across executors
- [ ] Check GC time per task
- [ ] Identify data skew patterns

### Query Plan Analysis
- [ ] Review physical plan in SQL tab
- [ ] Identify join strategies (Broadcast vs SortMerge)
- [ ] Count number of shuffles (stage boundaries)
- [ ] Look for unnecessary operations

### Executor Analysis
- [ ] Verify all executors are active
- [ ] Check for failed tasks
- [ ] Monitor memory usage
- [ ] Look for imbalanced workload

---

## 8. Required Screenshots for Your Project

For Milestone 3, include these Spark UI screenshots in your README:

### Screenshot A: Active Executors
- Shows multiple executors running
- Demonstrates distributed execution
- Location: Executors tab

### Screenshot B: Stage Metrics
- Shows your main processing stage
- Includes shuffle read/write
- Highlights any skew or spill

### Screenshot C: Query Plan (Optional but Recommended)
- Shows the physical plan for your main query
- Identifies join strategies
- From SQL tab

### How to Include in README

```markdown
## Spark UI Verification

### Executor Configuration
![Active Executors](images/spark_ui_executors.png)

Our job ran with 7 executors, each with 18GB memory.
All executors were actively processing tasks.

### Stage Performance
![Stage Metrics](images/spark_ui_stages.png)

The groupBy stage processed 50GB with:
- Shuffle write: 12GB
- Shuffle read: 12GB
- Max task duration: 45s
- Median task duration: 38s

### Analysis
The Max/Median ratio of 1.18 indicates minimal data skew.
No memory spill occurred, confirming our executor memory
configuration is appropriate for the workload.
```

---

## 9. Common Issues and Solutions

| Spark UI Symptom | Likely Cause | Solution |
|------------------|--------------|----------|
| Only 1 executor active | Local mode or config error | Check `spark.executor.instances` |
| High GC time | Memory pressure | Increase `spark.executor.memory` |
| Large shuffle spill | Not enough memory for shuffle | Increase `spark.memory.fraction` |
| Single slow task | Data skew | Salting, repartition, or filter |
| Many failed tasks | OOM errors | Reduce data per partition |
| No tasks running | Waiting for resources | Check SLURM allocation |

---

## Summary

### Key Skills Learned

1. **Navigate Spark UI** tabs effectively
2. **Identify bottlenecks** in stages and tasks
3. **Detect data skew** through task duration analysis
4. **Compare join strategies** using query plans
5. **Monitor memory** and identify spill issues
6. **Document findings** with meaningful screenshots

### For Your Project

Include Spark UI analysis in your README showing:
- Confirmation of distributed execution (multiple executors)
- Performance metrics for your main stages
- Any issues identified and how you addressed them

---

*This lab is part of DSC 232R: Big Data Analysis Using Spark at UCSD.*
