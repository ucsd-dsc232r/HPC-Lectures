# Ray + Spark Integration

## Key Takeaways

- **RayDP** runs Spark on Ray, unifying data processing and ML training
- **Side-by-side** patterns let each framework do what it does best
- **Data handoff** strategies enable smooth transitions between frameworks
- Choose integration approach based on your workload characteristics

---

## Connecting to What You Know

Throughout DSC 232R, you've built expertise with Spark:
- DataFrames and SQL for data processing
- PCA for dimensionality reduction
- XGBoost for gradient boosting

Now we'll connect that knowledge with Ray to create powerful ML pipelines.

---

## 1. Integration Approaches

### Three Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                  RAY + SPARK INTEGRATION                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ APPROACH 1: RayDP (Spark on Ray)                                │
│                                                                  │
│   Ray Cluster                                                   │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  SparkSession ─────────────────────────┐                │  │
│   │       │                                 │                │  │
│   │       ▼                                 ▼                │  │
│   │  Spark Executors    ───────────>    Ray Workers        │  │
│   │  (on Ray actors)                    (for ML)            │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│   Best for: Unified resource management, Spark SQL + Ray ML    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ APPROACH 2: Side-by-Side                                        │
│                                                                  │
│   ┌───────────────┐              ┌───────────────┐             │
│   │ Spark Cluster │              │ Ray Cluster   │             │
│   │               │              │               │             │
│   │  ETL Jobs     │──(Parquet)──>│  ML Training │             │
│   │  SQL Queries  │              │  Inference   │             │
│   └───────────────┘              └───────────────┘             │
│                                                                  │
│   Best for: Separate workloads, existing infrastructure        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ APPROACH 3: Data Handoff                                        │
│                                                                  │
│   Spark DataFrame                                               │
│        │                                                         │
│        ▼                                                         │
│   spark_df.toPandas()  ──or──  spark_df.write.parquet()        │
│        │                              │                          │
│        ▼                              ▼                          │
│   ray.data.from_pandas()      ray.data.read_parquet()          │
│        │                              │                          │
│        └──────────────┬───────────────┘                          │
│                       ▼                                          │
│               Ray Dataset                                        │
│                                                                  │
│   Best for: One-time conversion, memory-constrained pipelines  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. RayDP: Spark on Ray

### What is RayDP?

RayDP runs Spark executors as Ray actors, enabling:
- Unified resource management
- Shared object store between Spark and Ray
- Seamless data transfer without serialization

### Basic Setup

```python
import ray
import raydp

# Initialize Ray
ray.init()

# Create Spark session on Ray
spark = raydp.init_spark(
    app_name="RayDP_Example",
    num_executors=4,
    executor_cores=4,
    executor_memory="8GB",
)

# Use Spark as normal
df = spark.read.parquet("/data/weather/")
df_processed = df.filter(df.temperature > 0)

# Convert to Ray Dataset (zero-copy!)
ds = ray.data.from_spark(df_processed)

# Use Ray for ML
# ... training code ...

# Cleanup
raydp.stop_spark()
ray.shutdown()
```

### Memory Sharing

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAY OBJECT STORE                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   SHARED MEMORY                              ││
│  │                                                              ││
│  │   ┌───────────┐    ┌───────────┐    ┌───────────┐          ││
│  │   │ Spark DF  │───>│Ray Dataset│───>│ ML Model  │          ││
│  │   │ Partition │    │  Block    │    │  Training │          ││
│  │   └───────────┘    └───────────┘    └───────────┘          ││
│  │                                                              ││
│  │   Zero-copy data transfer via Apache Arrow                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Side-by-Side Pattern

### When to Use

- You have existing Spark infrastructure
- Workloads have clear separation (ETL vs ML)
- Want to minimize changes to existing code

### Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    ETL PIPELINE (Spark)                        │
│                                                                │
│  Raw Data ──> Clean ──> Transform ──> Feature Eng ──> Parquet │
│                                                                │
└───────────────────────────────────────────────────────────────┘
                              │
                              │ (Files on shared storage)
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                    ML PIPELINE (Ray)                           │
│                                                                │
│  Parquet ──> Ray Data ──> Preprocessing ──> Training ──> Model │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# ============== SPARK PIPELINE ==============
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("ETL").getOrCreate()

# ETL with Spark
df = spark.read.csv("/raw/weather/")
df_clean = df.dropna().filter(df.quality_flag == "OK")
df_features = df_clean.withColumn(
    "temp_celsius", (F.col("temp_kelvin") - 273.15)
)

# Save for Ray
df_features.write.parquet("/processed/weather/", mode="overwrite")
spark.stop()

# ============== RAY PIPELINE ==============
import ray
from ray.train.xgboost import XGBoostTrainer

ray.init()

# Load processed data
ds = ray.data.read_parquet("/processed/weather/")

# ML preprocessing
ds = ds.map_batches(normalize_features)

# Train
train_ds, test_ds = ds.train_test_split(0.2)
trainer = XGBoostTrainer(...)
result = trainer.fit()

ray.shutdown()
```

---

## 4. Data Handoff Strategies

### Strategy 1: Parquet Files (Recommended)

```python
# Spark writes
spark_df.write.parquet("/path/to/data/")

# Ray reads
ray_ds = ray.data.read_parquet("/path/to/data/")
```

**Pros**: Columnar format, compression, schema preservation
**Cons**: Disk I/O overhead

### Strategy 2: In-Memory (Small Data)

```python
# Collect to driver (careful with large data!)
pandas_df = spark_df.toPandas()

# Create Ray Dataset
ray_ds = ray.data.from_pandas(pandas_df)
```

**Pros**: No disk I/O
**Cons**: Memory limited, driver bottleneck

### Strategy 3: RayDP (Zero-Copy)

```python
# Direct conversion (requires RayDP)
ray_ds = ray.data.from_spark(spark_df)
```

**Pros**: Zero-copy, efficient
**Cons**: Requires RayDP setup

### Choosing a Strategy

| Strategy | Data Size | Latency | Memory Use | Setup |
|----------|-----------|---------|------------|-------|
| Parquet | Any | Higher | Low | Simple |
| In-Memory | < 10 GB | Low | High | Simple |
| RayDP | Any | Lowest | Medium | Complex |

---

## 5. Weather Data Pipeline Example

### Complete Integrated Pipeline

```python
import ray
import raydp
from pyspark.sql import functions as F
from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig

# Initialize Ray and Spark
ray.init()
spark = raydp.init_spark(
    app_name="WeatherML",
    num_executors=4,
    executor_cores=4,
    executor_memory="8GB"
)

# ========== SPARK: ETL ==========
# Load raw data
raw_df = spark.read.parquet("/data/raw/weather/")

# Clean and transform
cleaned_df = raw_df \
    .filter(F.col("quality_flag").isin(["OK", "GOOD"])) \
    .dropna(subset=["temperature", "humidity"]) \
    .withColumn("temp_celsius", F.col("temperature") - 273.15) \
    .withColumn("month", F.month("timestamp")) \
    .withColumn("hour", F.hour("timestamp"))

# Aggregate features
station_df = cleaned_df.groupBy("station_id", "date") \
    .agg(
        F.avg("temp_celsius").alias("avg_temp"),
        F.max("temp_celsius").alias("max_temp"),
        F.min("temp_celsius").alias("min_temp"),
        F.avg("humidity").alias("avg_humidity"),
        F.sum("precipitation").alias("total_precip")
    )

print(f"Processed {station_df.count()} station-day records")

# ========== HANDOFF: Spark to Ray ==========
# Zero-copy transfer
ds = ray.data.from_spark(station_df)

# ========== RAY: ML Preprocessing ==========
def engineer_features(batch):
    """Add ML features."""
    batch["temp_range"] = batch["max_temp"] - batch["min_temp"]
    batch["humidity_temp_ratio"] = batch["avg_humidity"] / (batch["avg_temp"] + 1)
    return batch

ds = ds.map_batches(engineer_features, batch_format="pandas")

# ========== RAY: Training ==========
train_ds, test_ds = ds.train_test_split(test_size=0.2)

trainer = XGBoostTrainer(
    label_column="total_precip",
    params={
        "objective": "reg:squarederror",
        "max_depth": 8,
        "eta": 0.1,
    },
    datasets={"train": train_ds, "valid": test_ds},
    scaling_config=ScalingConfig(num_workers=4),
)

result = trainer.fit()
print(f"RMSE: {result.metrics['valid-rmse']:.4f}")

# Cleanup
raydp.stop_spark()
ray.shutdown()
```

---

## 6. Performance Comparison

### Benchmark: Weather Data Processing

| Operation | Spark Only | Ray Only | Integrated |
|-----------|------------|----------|------------|
| Load 100GB Parquet | 15s | 18s | 15s |
| Complex SQL Joins | 45s | N/A | 45s (Spark) |
| Feature Engineering | 30s | 20s | 20s (Ray) |
| XGBoost Training | 120s* | 60s | 60s (Ray) |
| **Total** | **210s** | **N/A** | **140s** |

*Spark MLlib XGBoost via spark-xgboost

### Key Insights

1. **Spark wins at SQL**: Complex joins, aggregations, window functions
2. **Ray wins at ML**: Training, hyperparameter tuning, inference
3. **Integration wins overall**: Use each where it excels

---

## 7. Common Patterns

### Pattern 1: ETL → Train → Serve

```
Spark ETL ──> Parquet ──> Ray Train ──> Ray Serve
                              │
                              └──> Model Checkpoint
```

### Pattern 2: Streaming ETL → Batch Training

```
Spark Streaming ──> Delta/Parquet (hourly)
                          │
                          └──> Ray Train (daily batch)
```

### Pattern 3: Feature Store Integration

```
Spark ETL ──> Feature Store (Feast, etc.)
                    │
                    └──> Ray reads features for training/inference
```

---

## Summary

### Integration Approaches

| Approach | Use When | Complexity |
|----------|----------|------------|
| RayDP | Unified cluster, shared memory | Medium |
| Side-by-Side | Separate workloads | Low |
| Data Handoff | One-time conversion | Low |

### Best Practices

1. **Use Spark for**:
   - Complex SQL queries
   - Large-scale joins
   - Data cleaning and validation
   - Existing ETL pipelines

2. **Use Ray for**:
   - ML model training
   - Hyperparameter tuning
   - Model inference
   - Custom Python operations

3. **Data format**:
   - Parquet for interchange
   - Arrow for in-memory transfer
   - Consider compression for large data

### Connecting to DSC 232R

| Course Topic | Integration Pattern |
|--------------|---------------------|
| Weather PCA (Class09-10) | Spark preprocessing → Ray PCA |
| XGBoost (Class13-15) | Spark features → Ray Train XGBoost |
| Future projects | Spark ETL → Ray ML pipeline |

---

## Practice Problems

1. **Architecture Design**: You have 500 GB of log data that needs SQL-based aggregation followed by anomaly detection ML. Design the integration architecture.

2. **Handoff Strategy**: Your Spark DataFrame has 50 million rows. Which handoff strategy would you use for Ray ML training?

3. **Resource Planning**: You're running RayDP on a cluster with 256 GB RAM total. How would you split resources between Spark executors and Ray workers?

---

*Next: Module 5 - Ray on SLURM*
