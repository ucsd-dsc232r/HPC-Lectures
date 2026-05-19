# Introduction to Ray

## Key Takeaways

- **Ray** is a distributed computing framework designed for ML workloads
- **Tasks** (`@ray.remote`) parallelize functions across workers
- **Actors** maintain state across multiple method calls
- **Ray Data** provides scalable data processing for ML pipelines
- **Ray Train** distributes model training across multiple workers

---

## Connecting to What You Know

### From Spark to Ray

You've used Spark's RDDs and DataFrames for distributed data processing. Ray offers a different paradigm:

| Concept | Spark | Ray |
|---------|-------|-----|
| Basic unit | RDD/DataFrame | Task/Actor |
| State | Stateless transformations | Stateful Actors |
| Data model | Distributed collections | Object references |
| Primary use | Batch data processing | ML workloads |

### The ML Gap

Spark excels at ETL and batch processing, but ML workloads often need:
- **Flexible task graphs** (not just map-reduce)
- **Stateful computation** (model parameters)
- **Low latency** (real-time inference)
- **Native Python objects** (numpy arrays, tensors)

Ray fills this gap.

---

## 1. Ray Core Concepts

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAY CLUSTER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                       HEAD NODE                              ││
│  │  - Global Control Store (GCS)                               ││
│  │  - Driver process                                            ││
│  │  - Dashboard (port 8265)                                     ││
│  └─────────────────────────────────────────────────────────────┘│
│                               │                                  │
│              ┌────────────────┼────────────────┐                │
│              │                │                │                │
│              ▼                ▼                ▼                │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  WORKER NODE  │  │  WORKER NODE  │  │  WORKER NODE  │       │
│  │               │  │               │  │               │       │
│  │  ┌─────────┐  │  │  ┌─────────┐  │  │  ┌─────────┐  │       │
│  │  │ Worker  │  │  │  │ Worker  │  │  │  │ Worker  │  │       │
│  │  │ Process │  │  │  │ Process │  │  │  │ Process │  │       │
│  │  └─────────┘  │  │  └─────────┘  │  │  └─────────┘  │       │
│  │               │  │               │  │               │       │
│  │  ┌─────────┐  │  │  ┌─────────┐  │  │  ┌─────────┐  │       │
│  │  │ Object  │  │  │  │ Object  │  │  │  │ Object  │  │       │
│  │  │ Store   │  │  │  │ Store   │  │  │  │ Store   │  │       │
│  │  └─────────┘  │  │  └─────────┘  │  │  └─────────┘  │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Driver**: Your Python script that calls `ray.init()`
2. **Workers**: Processes that execute tasks and actors
3. **Object Store**: Shared memory for data (uses Apache Arrow)
4. **GCS**: Manages cluster state and scheduling

---

## 2. Ray Tasks

### The `@ray.remote` Decorator

Transform any Python function into a distributed task:

```python
import ray

ray.init()

# Regular Python function
def square(x):
    return x * x

# Ray remote function (task)
@ray.remote
def square_remote(x):
    return x * x

# Local execution
result_local = square(4)  # Returns 16 immediately

# Remote execution
future = square_remote.remote(4)  # Returns ObjectRef immediately
result = ray.get(future)  # Blocks until result ready, returns 16
```

### Key Points

1. **`.remote()`** submits the task to the cluster
2. **Returns immediately** with an `ObjectRef` (future/promise)
3. **`ray.get()`** blocks until result is ready
4. **Parallel execution**: Multiple `.remote()` calls run in parallel

### Task Dependencies

Ray automatically handles task dependencies:

```python
@ray.remote
def add(a, b):
    return a + b

@ray.remote
def multiply(a, b):
    return a * b

# Create task graph
x = add.remote(1, 2)      # Task 1: 1 + 2 = 3
y = add.remote(3, 4)      # Task 2: 3 + 4 = 7 (parallel with Task 1)
z = multiply.remote(x, y)  # Task 3: waits for x and y, then 3 * 7 = 21

result = ray.get(z)  # 21
```

```
Task Graph:
     add(1,2)         add(3,4)
         │                │
         ▼                ▼
         x                y
         │                │
         └───────┬────────┘
                 │
                 ▼
          multiply(x, y)
                 │
                 ▼
                 z
```

### Specifying Resources

Control resource allocation per task:

```python
@ray.remote(num_cpus=2, num_gpus=1)
def train_model(data):
    # This task requires 2 CPUs and 1 GPU
    return model.fit(data)

# Override at call time
future = train_model.options(num_cpus=4).remote(data)
```

---

## 3. Ray Actors

### Stateful Distributed Objects

While tasks are stateless, actors maintain state:

```python
@ray.remote
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

    def get_count(self):
        return self.count

# Create actor instance
counter = Counter.remote()  # Returns ActorHandle

# Call methods
ray.get(counter.increment.remote())  # 1
ray.get(counter.increment.remote())  # 2
ray.get(counter.get_count.remote())  # 2
```

### Actor Use Cases

| Use Case | Example |
|----------|---------|
| ML Model | Store model weights, provide inference |
| Database Connection | Maintain persistent connection |
| Counter/Metrics | Track distributed statistics |
| Simulation State | Game/physics state across updates |

### Actor Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    ACTOR LIFECYCLE                           │
└─────────────────────────────────────────────────────────────┘

1. Creation
   actor = MyActor.remote(args)
   └── Ray schedules actor on a worker
   └── __init__() runs once

2. Method Calls
   future = actor.method.remote(args)
   └── Queued and executed sequentially
   └── Maintains state between calls

3. Destruction
   ray.kill(actor)
   └── Or actor goes out of scope
   └── Resources released
```

---

## 4. Object Store

### Shared Memory for Data

Ray's object store enables zero-copy data sharing:

```python
import numpy as np

# Put data in object store
large_array = np.random.rand(10_000_000)
ref = ray.put(large_array)  # Returns ObjectRef

# Multiple tasks can read without copying
@ray.remote
def process(data_ref):
    data = ray.get(data_ref)  # Zero-copy read
    return data.mean()

futures = [process.remote(ref) for _ in range(10)]
results = ray.get(futures)  # All tasks share same data
```

### Memory Management

```
┌─────────────────────────────────────────────────────────────┐
│                     OBJECT STORE                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              SHARED MEMORY (mmap)                     │   │
│  │                                                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │   │
│  │  │ Object  │  │ Object  │  │ Object  │              │   │
│  │  │   A     │  │   B     │  │   C     │              │   │
│  │  └─────────┘  └─────────┘  └─────────┘              │   │
│  │                                                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Benefits:                                                   │
│  - Zero-copy reads between processes                        │
│  - Automatic reference counting                             │
│  - Spill to disk when full                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Ray Data

### Scalable Data Processing

Ray Data provides distributed datasets optimized for ML:

```python
import ray

# Create dataset from various sources
ds = ray.data.read_parquet("s3://bucket/data/")
ds = ray.data.read_csv("/path/to/files/*.csv")
ds = ray.data.from_pandas(pandas_df)
ds = ray.data.from_numpy(numpy_array)

# Transformations (lazy, like Spark)
ds = ds.map(lambda row: {"x": row["x"] * 2})
ds = ds.filter(lambda row: row["x"] > 0)
ds = ds.flat_map(lambda row: [row, row])  # 1 → many

# Actions (trigger computation)
ds.count()
ds.take(10)
ds.to_pandas()
```

### Ray Data vs Spark

| Feature | Spark | Ray Data |
|---------|-------|----------|
| Primary format | DataFrame | Dataset (row-based) |
| Execution | JVM-based | Native Python |
| ML integration | Spark MLlib | Ray Train, PyTorch, TF |
| Streaming | Structured Streaming | Streaming ingestion |
| Best for | ETL, SQL analytics | ML preprocessing |

### Preprocessing Pipeline

```python
import ray
from ray.data.preprocessors import StandardScaler, OneHotEncoder

# Load data
ds = ray.data.read_parquet("weather_data/")

# Define preprocessing
def extract_features(batch):
    """Process a batch of rows."""
    batch["temp_normalized"] = (batch["temperature"] - 273.15)  # Kelvin to Celsius
    batch["month"] = batch["date"].dt.month
    return batch

# Apply transformations
ds = ds.map_batches(extract_features)

# Use built-in preprocessors
scaler = StandardScaler(columns=["temp_normalized"])
ds = scaler.fit_transform(ds)

# Convert to training format
train_ds, test_ds = ds.train_test_split(test_size=0.2)
```

---

## 6. Ray Train

### Distributed Training Framework

Ray Train simplifies distributed ML training:

```python
from ray import train
from ray.train import ScalingConfig
from ray.train.xgboost import XGBoostTrainer

# Define training configuration
trainer = XGBoostTrainer(
    label_column="target",
    params={
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.1,
    },
    datasets={"train": train_ds, "valid": test_ds},
    scaling_config=ScalingConfig(
        num_workers=4,
        use_gpu=False,
    ),
)

# Run distributed training
result = trainer.fit()
print(f"Best RMSE: {result.metrics['valid-rmse']:.4f}")
```

### Supported Frameworks

| Framework | Trainer Class |
|-----------|--------------|
| XGBoost | `XGBoostTrainer` |
| LightGBM | `LightGBMTrainer` |
| PyTorch | `TorchTrainer` |
| TensorFlow | `TensorflowTrainer` |
| Scikit-learn | `SklearnTrainer` |
| Hugging Face | `TransformersTrainer` |

### Custom PyTorch Training

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import torch

def train_func(config):
    """Training function executed on each worker."""
    # Get distributed data
    train_ds = train.get_dataset_shard("train")

    # Create model (automatically distributed)
    model = torch.nn.Linear(10, 1)
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        for batch in train_ds.iter_torch_batches(batch_size=32):
            optimizer.zero_grad()
            outputs = model(batch["features"])
            loss = torch.nn.functional.mse_loss(outputs, batch["labels"])
            loss.backward()
            optimizer.step()

        # Report metrics
        train.report({"loss": loss.item(), "epoch": epoch})

trainer = TorchTrainer(
    train_func,
    train_loop_config={"lr": 0.001, "epochs": 10},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    datasets={"train": train_ds},
)

result = trainer.fit()
```

---

## 7. Comparing Ray and Spark

### When to Use Each

```
┌─────────────────────────────────────────────────────────────┐
│                  DECISION FLOWCHART                          │
└─────────────────────────────────────────────────────────────┘

What's your primary task?
         │
         ├── ETL / Data Warehousing?
         │   └── Use SPARK
         │       - Strong SQL support
         │       - Optimized joins/aggregations
         │       - Mature ecosystem
         │
         ├── ML Training / Inference?
         │   └── Use RAY
         │       - Native Python/PyTorch/TF
         │       - Stateful actors for models
         │       - Flexible task graphs
         │
         └── Both?
             └── Use BOTH!
                 - Spark for ETL
                 - Ray for ML
                 - RayDP for integration
```

### Performance Characteristics

| Workload | Spark | Ray |
|----------|-------|-----|
| SQL queries | Excellent | Good |
| Shuffle-heavy joins | Excellent | Good |
| Stateless transforms | Excellent | Excellent |
| Stateful computation | Limited | Excellent |
| Real-time inference | Poor | Excellent |
| Custom ML training | Limited | Excellent |
| Python object support | Poor | Excellent |

---

## Summary

### Ray Core

1. **Tasks** (`@ray.remote` on functions): Stateless parallel execution
2. **Actors** (`@ray.remote` on classes): Stateful distributed objects
3. **Object Store**: Shared memory with zero-copy reads

### Ray Data

1. **Datasets**: Distributed, row-based collections
2. **Transformations**: map, filter, flat_map, map_batches
3. **Integration**: Direct pipeline to Ray Train

### Ray Train

1. **Trainers**: Pre-built for XGBoost, PyTorch, TensorFlow
2. **ScalingConfig**: Control workers and resources
3. **Checkpointing**: Automatic model saving

### Connecting to DSC 232R

| Prior Topic | Ray Connection |
|-------------|----------------|
| Spark map/reduce | Ray tasks provide similar parallelism |
| Spark DataFrames | Ray Data offers ML-optimized alternative |
| XGBoost (Class13-15) | Ray Train scales XGBoost across cluster |
| Weather analysis | Same data, different processing model |

---

## Practice Problems

1. **Task vs Actor**: When would you use a Ray Actor instead of Ray Tasks?

2. **Memory Efficiency**: You have a 10 GB numpy array that 100 tasks need to read. How would you minimize memory usage?

3. **Training Scale**: You want to train XGBoost on 100 GB of data with 4 workers. Write the code to set up the trainer.

---

*Next: Module 4 - Ray + Spark Integration*
