# Ray on SLURM

## Key Takeaways

- **Ray clusters** can span multiple SLURM nodes for massive scale
- **Head node + worker nodes** architecture matches SLURM job allocations
- **Singularity containers** ensure consistent Ray environments
- **Multi-node Ray** enables training on datasets larger than single-node memory

---

## Connecting to What You Know

In Module 2, you learned to run Spark on SDSC Expanse using SLURM. Now we'll deploy Ray clusters the same way, enabling:
- Distributed ML training across multiple nodes
- Scaling beyond single-node memory limits
- Integration with existing HPC workflows

---

## 1. Ray Cluster Architecture on SLURM

### Single-Node vs Multi-Node

```
SINGLE NODE (Modules 3-4)
┌─────────────────────────────────────────────────────────────────┐
│                        COMPUTE NODE                              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     RAY CLUSTER                              ││
│  │                                                              ││
│  │   Head Node Process  ───────>  Worker Processes             ││
│  │   (ray.init())              (automatic)                     ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Limited to: 1 node's RAM (256 GB), 1 node's cores (128)       │
└─────────────────────────────────────────────────────────────────┘

MULTI-NODE (This Module)
┌─────────────────────────────────────────────────────────────────┐
│                     SLURM JOB ALLOCATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                       HEAD NODE                              ││
│  │  - Ray GCS (Global Control Store)                           ││
│  │  - Dashboard (port 8265)                                    ││
│  │  - Driver process                                            ││
│  │  IP: $SLURM_NODELIST[0]                                     ││
│  └──────────────────────────────┬──────────────────────────────┘│
│                                 │                                │
│              ┌──────────────────┼──────────────────┐            │
│              │                  │                  │            │
│              ▼                  ▼                  ▼            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  WORKER NODE  │  │  WORKER NODE  │  │  WORKER NODE  │       │
│  │               │  │               │  │               │       │
│  │  Ray Workers  │  │  Ray Workers  │  │  Ray Workers  │       │
│  │  Object Store │  │  Object Store │  │  Object Store │       │
│  │               │  │               │  │               │       │
│  │  128 cores    │  │  128 cores    │  │  128 cores    │       │
│  │  256 GB RAM   │  │  256 GB RAM   │  │  256 GB RAM   │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
│                                                                  │
│  Total: 4 nodes × 128 cores = 512 cores, 1 TB RAM              │
└─────────────────────────────────────────────────────────────────┘
```

### Why Multi-Node?

| Scenario | Single Node (256 GB) | 4 Nodes (1 TB) |
|----------|---------------------|----------------|
| 100 GB dataset | Works | Overkill |
| 500 GB dataset | Out of memory | Works |
| Train 100 models | ~100 at once | ~400 at once |
| XGBoost on 1B rows | Slow | 4x faster |

---

## 2. SLURM Script for Multi-Node Ray

### Basic Multi-Node Script

```bash
#!/bin/bash
#SBATCH --job-name=ray_cluster
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --account=uci150
#SBATCH --output=logs/ray_%j.out
#SBATCH --error=logs/ray_%j.err

# =============================================================================
# Multi-Node Ray Cluster on SLURM
# =============================================================================

# Load modules
module load singularitypro

# Container path
CONTAINER=/expanse/lustre/projects/uci150/$USER/ray_spark_dsc232r.sif

# Get head node info
HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w $HEAD_NODE hostname -i)
RAY_PORT=6379

echo "============================================="
echo "Ray Cluster Configuration"
echo "============================================="
echo "Head Node: $HEAD_NODE ($HEAD_NODE_IP)"
echo "Worker Nodes: $(scontrol show hostnames $SLURM_NODELIST | tail -n +2 | tr '\n' ' ')"
echo "Total Nodes: $SLURM_NNODES"
echo "CPUs per Node: $SLURM_CPUS_PER_TASK"
echo "============================================="

# Start Ray head node
echo "Starting Ray head node..."
srun --nodes=1 --ntasks=1 -w $HEAD_NODE \
    singularity exec --bind /expanse $CONTAINER \
    ray start --head --port=$RAY_PORT \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --block &

# Wait for head node to start
sleep 10

# Start Ray worker nodes
echo "Starting Ray worker nodes..."
for NODE in $(scontrol show hostnames $SLURM_NODELIST | tail -n +2); do
    srun --nodes=1 --ntasks=1 -w $NODE \
        singularity exec --bind /expanse $CONTAINER \
        ray start --address=$HEAD_NODE_IP:$RAY_PORT \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --block &
done

# Wait for cluster to form
sleep 20

# Run main script
echo "Running main application..."
singularity exec --bind /expanse $CONTAINER \
    python main_script.py --ray-address=$HEAD_NODE_IP:$RAY_PORT

# Cleanup
ray stop
```

---

## 3. Connecting to the Ray Cluster

### In Your Python Script

```python
import ray
import os

def connect_to_slurm_ray():
    """Connect to Ray cluster started by SLURM script."""

    # Get head node address from environment or argument
    ray_address = os.environ.get('RAY_ADDRESS', 'auto')

    # Connect to existing cluster
    ray.init(address=ray_address)

    # Verify cluster
    print(f"Ray version: {ray.__version__}")
    print(f"Cluster resources: {ray.cluster_resources()}")
    print(f"Available resources: {ray.available_resources()}")

    return ray

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ray-address', default='auto')
    args = parser.parse_args()

    os.environ['RAY_ADDRESS'] = args.ray_address
    connect_to_slurm_ray()
```

### Verifying the Cluster

```python
# Check cluster size
@ray.remote
def get_node_info():
    import socket
    return {
        'hostname': socket.gethostname(),
        'pid': os.getpid(),
    }

# Get info from all nodes
futures = [get_node_info.remote() for _ in range(10)]
results = ray.get(futures)

# Count unique hostnames
unique_hosts = set(r['hostname'] for r in results)
print(f"Tasks ran on {len(unique_hosts)} unique nodes")
```

---

## 4. Ray Data on Multi-Node Cluster

### Loading Large Datasets

```python
import ray

# Connect to cluster
ray.init(address='auto')

# Load 500 GB dataset (distributed across all nodes)
ds = ray.data.read_parquet(
    "/expanse/lustre/projects/uci150/shared/large_dataset/",
    parallelism=200,  # Match total cores
)

print(f"Dataset blocks: {ds.num_blocks()}")
print(f"Estimated size: {ds.size_bytes() / 1e9:.1f} GB")

# Processing happens across all nodes
ds_processed = ds.map_batches(
    preprocess_function,
    batch_format="pandas",
    num_cpus=1,
)
```

### Memory Management

```python
# Configure object store for large data
ray.init(
    address='auto',
    object_store_memory=100 * 1024 * 1024 * 1024,  # 100 GB per node
)

# Spill to disk if needed
ray.init(
    address='auto',
    _system_config={
        "object_spilling_config": {
            "type": "filesystem",
            "params": {
                "directory_path": "/scratch/$USER/ray_spill"
            }
        }
    }
)
```

---

## 5. Distributed Training on Multi-Node

### Ray Train with Multiple Nodes

```python
from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig

# Load data
train_ds = ray.data.read_parquet("/data/train/")
valid_ds = ray.data.read_parquet("/data/valid/")

# Configure distributed training
trainer = XGBoostTrainer(
    label_column="target",
    params={
        "objective": "reg:squarederror",
        "max_depth": 8,
        "eta": 0.1,
    },
    datasets={"train": train_ds, "valid": valid_ds},
    scaling_config=ScalingConfig(
        num_workers=16,  # Spread across nodes
        use_gpu=False,
        resources_per_worker={"CPU": 8},
    ),
)

result = trainer.fit()
print(f"Training completed on {16 * 8} = 128 cores")
```

### PyTorch Distributed Training

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import torch

def train_func(config):
    """Training function for each worker."""
    import ray.train as train

    # Get distributed data shard
    train_ds = train.get_dataset_shard("train")

    # Model (automatically distributed)
    model = MyModel()
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(config["epochs"]):
        for batch in train_ds.iter_torch_batches(batch_size=32):
            # Training step
            loss = train_step(model, batch, optimizer)

        # Report metrics
        train.report({"loss": loss.item()})

trainer = TorchTrainer(
    train_func,
    train_loop_config={"epochs": 10},
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,  # If GPU nodes
        resources_per_worker={"GPU": 1},
    ),
    datasets={"train": train_dataset},
)

result = trainer.fit()
```

---

## 6. Complete Multi-Node Example

### Job Script: `ray_multinode.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=ray_weather_ml
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --account=uci150
#SBATCH --output=logs/ray_weather_%j.out
#SBATCH --error=logs/ray_weather_%j.err

# Setup
module load singularitypro
CONTAINER=/expanse/lustre/projects/uci150/$USER/ray_spark_dsc232r.sif

# Get nodes
HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w $HEAD_NODE hostname -i)

# Start Ray cluster
echo "Starting Ray cluster..."
srun --nodes=1 --ntasks=1 -w $HEAD_NODE \
    singularity exec $CONTAINER ray start --head --port=6379 --num-cpus=64 --block &
sleep 10

for NODE in $(scontrol show hostnames $SLURM_NODELIST | tail -n +2); do
    srun --nodes=1 --ntasks=1 -w $NODE \
        singularity exec $CONTAINER ray start --address=$HEAD_IP:6379 --num-cpus=64 --block &
done
sleep 20

# Run training
singularity exec --bind /expanse $CONTAINER \
    python weather_training.py --ray-address=$HEAD_IP:6379

ray stop
```

### Python Script: `weather_training.py`

```python
#!/usr/bin/env python3
"""
Multi-node weather prediction training on SDSC Expanse.
"""

import ray
import argparse
import os
from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ray-address', default='auto')
    args = parser.parse_args()

    # Connect to cluster
    ray.init(address=args.ray_address)
    print(f"Connected to Ray cluster")
    print(f"Resources: {ray.cluster_resources()}")

    # Load data
    data_path = "/expanse/lustre/projects/uci150/shared/weather/"
    train_ds = ray.data.read_parquet(f"{data_path}/train/")
    valid_ds = ray.data.read_parquet(f"{data_path}/valid/")

    print(f"Training samples: {train_ds.count():,}")
    print(f"Validation samples: {valid_ds.count():,}")

    # Feature preprocessing
    def preprocess(batch):
        batch["temp_range"] = batch["max_temp"] - batch["min_temp"]
        batch["pressure_norm"] = (batch["pressure"] - 1013.25) / 20
        return batch

    train_ds = train_ds.map_batches(preprocess, batch_format="pandas")
    valid_ds = valid_ds.map_batches(preprocess, batch_format="pandas")

    # Configure trainer for multi-node
    n_workers = int(ray.cluster_resources().get("CPU", 64)) // 8

    trainer = XGBoostTrainer(
        label_column="precipitation",
        num_boost_round=200,
        params={
            "objective": "reg:squarederror",
            "eval_metric": ["rmse", "mae"],
            "max_depth": 10,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        datasets={"train": train_ds, "valid": valid_ds},
        scaling_config=ScalingConfig(
            num_workers=n_workers,
            use_gpu=False,
            resources_per_worker={"CPU": 8},
        ),
    )

    print(f"Training with {n_workers} workers...")
    result = trainer.fit()

    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"RMSE: {result.metrics.get('valid-rmse', 'N/A'):.4f}")
    print(f"MAE: {result.metrics.get('valid-mae', 'N/A'):.4f}")
    print(f"Checkpoint: {result.checkpoint}")

    ray.shutdown()

if __name__ == "__main__":
    main()
```

---

## 7. Troubleshooting Multi-Node Ray

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Workers can't connect | "Unable to connect to GCS" | Check firewall, use correct IP |
| Port conflict | "Address already in use" | Use unique port per job |
| Memory issues | Workers killed | Increase --mem, enable spilling |
| Slow startup | Cluster takes minutes | Reduce worker count, check network |

### Debugging Commands

```bash
# Check Ray status (on head node)
ray status

# View Ray logs
ls /tmp/ray/session_latest/logs/

# Monitor resource usage
ray memory

# Kill stuck Ray processes
ray stop --force
```

### Network Configuration

```python
# Specify ports explicitly
ray.init(
    address='auto',
    _node_ip_address=os.environ.get('HEAD_IP'),
    dashboard_port=8265,
    object_manager_port=8076,
)
```

---

## Summary

### Multi-Node Ray on SLURM

1. **SLURM allocates nodes** → Ray creates cluster across them
2. **Head node** runs GCS, dashboard, driver
3. **Worker nodes** run Ray workers and object stores
4. **Singularity** ensures consistent environment

### When to Use Multi-Node

| Scenario | Nodes Needed |
|----------|--------------|
| Development, small data | 1 |
| Medium datasets (50-200 GB) | 2-4 |
| Large datasets (200+ GB) | 4-8 |
| Massive parallel training | 8+ |

### Best Practices

1. **Start small**: Test on 1-2 nodes first
2. **Use debug partition**: Fast iteration
3. **Monitor resources**: `ray status`, SLURM logs
4. **Enable spilling**: For datasets near memory limit
5. **Scale workers**: More workers = more parallelism

### Connecting to DSC 232R

| Prior Topic | Multi-Node Connection |
|-------------|----------------------|
| Spark on SDSC | Same SLURM patterns, different framework |
| XGBoost | Now scales beyond single node |
| Weather analysis | Can process full dataset in memory |

---

## Practice Problems

1. **Resource Planning**: You have 1 TB of training data. How many Expanse nodes would you request?

2. **Script Writing**: Write a SLURM script for 8-node Ray cluster with GPU support.

3. **Troubleshooting**: Your 4-node job runs but only uses 1 node. What might be wrong?

---

## Further Reading

- Ray Cluster Documentation: https://docs.ray.io/en/latest/cluster/getting-started.html
- SDSC Expanse User Guide: https://www.sdsc.edu/support/user_guides/expanse.html
- Ray on SLURM: https://docs.ray.io/en/latest/cluster/slurm.html

---

*End of Module 5 - Ray on SLURM*
