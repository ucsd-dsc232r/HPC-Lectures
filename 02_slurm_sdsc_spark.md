# SLURM and Spark on SDSC Expanse

## Key Takeaways

- **SLURM** is the job scheduler that manages compute resources on HPC clusters
- **sbatch** submits batch jobs; **srun** runs interactive commands within allocations
- **Spark on HPC** requires careful configuration of memory, cores, and executors
- **Singularity containers** provide consistent software environments across nodes

---

## Connecting to What You Know

### From Local to Cluster

You've run Spark locally throughout this course with `SparkSession.builder.master("local[*]")`. On a cluster:
- Resources must be **requested** through a scheduler
- Jobs **wait in queue** until resources are available
- Multiple users **share** the system

### From Docker to Singularity

If you've used Docker, Singularity is similar but designed for HPC:
- **No root privileges** required to run containers
- **Direct access** to host filesystems
- **MPI-compatible** for distributed computing

---

## 1. Connecting to SDSC Expanse

### SSH Access

```bash
# First time: Set up SSH key authentication
ssh-keygen -t rsa -b 4096 -C "your_email@ucsd.edu"
ssh-copy-id username@login.expanse.sdsc.edu

# Connect to Expanse
ssh username@login.expanse.sdsc.edu
```

### File System Layout

```
/home/<username>/                    # Home directory (100 GB quota)
├── .bashrc                          # Shell configuration
├── .jupyter/                        # Jupyter settings
└── projects/                        # Symbolic links to projects

/expanse/lustre/projects/uci150/     # Project directory (shared)
├── <username>/                      # Your allocation
│   ├── data/                        # Datasets
│   ├── containers/                  # Singularity images
│   └── outputs/                     # Job outputs

/scratch/<username>/<job_id>/        # Temporary job storage
                                     # Auto-deleted after 30 days
```

### Important Paths

| Path | Purpose | Quota | Notes |
|------|---------|-------|-------|
| `/home` | Personal files | 100 GB | Backed up |
| `/expanse/lustre/projects` | Project data | Per allocation | High bandwidth |
| `/scratch` | Temporary files | 10 TB | Auto-purged |

---

## 2. Understanding SLURM

### What is SLURM?

**SLURM** (Simple Linux Utility for Resource Management) is a job scheduler that:
- Allocates compute resources to users
- Manages job queues and priorities
- Tracks resource usage and accounting

### SLURM Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SLURM CONTROLLER                          │
│  - Manages job queue                                             │
│  - Allocates resources                                           │
│  - Tracks accounting                                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ COMPUTE     │     │ COMPUTE     │     │ COMPUTE     │
    │ NODE 1      │     │ NODE 2      │     │ NODE N      │
    │             │     │             │     │             │
    │ slurmd      │     │ slurmd      │     │ slurmd      │
    │ daemon      │     │ daemon      │     │ daemon      │
    └─────────────┘     └─────────────┘     └─────────────┘
```

### Key SLURM Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `sbatch` | Submit batch job | `sbatch job.slurm` |
| `srun` | Run command in allocation | `srun python script.py` |
| `squeue` | View job queue | `squeue -u $USER` |
| `scancel` | Cancel job | `scancel 12345` |
| `sinfo` | View partition info | `sinfo -p shared` |
| `sacct` | View job accounting | `sacct -j 12345` |

---

## 3. Writing SLURM Job Scripts

### Basic Structure

```bash
#!/bin/bash
#SBATCH --job-name=my_spark_job      # Job name
#SBATCH --partition=shared           # Queue/partition
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Tasks per node
#SBATCH --cpus-per-task=8            # CPUs per task
#SBATCH --mem=32G                    # Memory per node
#SBATCH --time=01:00:00              # Time limit (HH:MM:SS)
#SBATCH --account=uci150             # Allocation account
#SBATCH --output=output_%j.log       # Standard output (%j = job ID)
#SBATCH --error=error_%j.log         # Standard error

# Your commands here
module load singularitypro
singularity exec container.sif python my_script.py
```

### Common SBATCH Options

| Option | Description | Common Values |
|--------|-------------|---------------|
| `--partition` | Queue to submit to | `debug`, `shared`, `compute`, `gpu` |
| `--nodes` | Number of nodes | 1-32 (partition dependent) |
| `--ntasks` | Total number of tasks | Typically 1 for Python |
| `--cpus-per-task` | CPU cores per task | 1-128 |
| `--mem` | Memory per node | e.g., `32G`, `64G`, `256G` |
| `--time` | Wall clock limit | `HH:MM:SS` format |
| `--gres` | Generic resources | `gpu:1` for GPU jobs |

### Partition Selection Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                    CHOOSING A PARTITION                          │
└─────────────────────────────────────────────────────────────────┘

Testing/Development?
     │
     ├── YES → Use 'debug' partition
     │         - Max 2 nodes, 30 minutes
     │         - Fast queue, good for testing
     │
     └── NO → How many cores do you need?
              │
              ├── ≤128 cores (1 node) → Use 'shared' partition
              │   - Share node with other users
              │   - Most efficient for small jobs
              │
              └── >128 cores → Use 'compute' partition
                  - Full nodes dedicated to your job
                  - Max 32 nodes, 48 hours

Need GPUs?
     │
     ├── 1-2 GPUs → Use 'gpu-shared'
     │
     └── 3+ GPUs → Use 'gpu'
```

---

## 4. Spark Configuration for SDSC

### Memory Hierarchy on Expanse

```
┌─────────────────────────────────────────────────────────────────┐
│                         NODE (256 GB)                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 SPARK DRIVER (e.g., 32 GB)                 │ │
│  │  - Coordinates execution                                    │ │
│  │  - Collects results                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  EXECUTOR 1  │ │  EXECUTOR 2  │ │  EXECUTOR N  │            │
│  │  (e.g., 48G) │ │  (e.g., 48G) │ │  (e.g., 48G) │            │
│  │              │ │              │ │              │            │
│  │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │            │
│  │ │  Cores   │ │ │ │  Cores   │ │ │ │  Cores   │ │            │
│  │ │  (8)     │ │ │ │  (8)     │ │ │ │  (8)     │ │            │
│  │ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              OVERHEAD / OS (remaining memory)               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Spark Configuration Template

```python
from pyspark.sql import SparkSession
import os

# Calculate resources based on SLURM allocation
cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
mem_gb = int(os.environ.get('SLURM_MEM_PER_NODE', '32000').replace('G', '').replace('M', '')) // 1000

# Reserve memory for driver and overhead
driver_mem = max(4, mem_gb // 8)  # ~12.5% for driver
executor_mem = mem_gb - driver_mem - 2  # 2 GB overhead

spark = SparkSession.builder \
    .appName("SDSC_Spark_Job") \
    .master("local[*]") \
    .config("spark.driver.memory", f"{driver_mem}g") \
    .config("spark.executor.memory", f"{executor_mem}g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.sql.shuffle.partitions", str(cpus * 2)) \
    .config("spark.default.parallelism", str(cpus * 2)) \
    .getOrCreate()
```

### Configuration Guidelines

| Parameter | Guideline | Reason |
|-----------|-----------|--------|
| `spark.driver.memory` | 10-15% of total | Driver aggregates results |
| `spark.executor.memory` | 75-80% of total | Main processing memory |
| `spark.sql.shuffle.partitions` | 2-3x cores | Parallelism for shuffles |
| `spark.default.parallelism` | 2-3x cores | Default RDD partitions |
| `spark.driver.maxResultSize` | 2-4 GB | Prevent OOM on collect() |

---

## 5. Singularity Containers

### Why Containers on HPC?

1. **Reproducibility**: Same environment everywhere
2. **Dependency Management**: Complex software stacks packaged together
3. **Portability**: Move jobs between clusters
4. **Version Control**: Track environment changes

### Using Pre-built Containers

```bash
# Pull from Docker Hub (converts automatically)
singularity pull docker://continuumio/miniconda3:latest

# Pull from Singularity Hub
singularity pull shub://singularityhub/hello-world

# Use our course container
singularity pull /expanse/lustre/projects/uci150/shared/ray_spark_dsc232r.sif
```

### Running Commands in Containers

```bash
# Interactive shell
singularity shell container.sif

# Execute single command
singularity exec container.sif python --version

# Run with bound paths (access host filesystem)
singularity exec --bind /expanse container.sif python script.py

# Run with environment variables
singularity exec --env PYTHONPATH=/app container.sif python script.py
```

### Binding Paths

```bash
# Bind Expanse project directory
singularity exec \
    --bind /expanse/lustre/projects/uci150 \
    --bind /scratch/$USER/$SLURM_JOB_ID:/scratch \
    container.sif python my_script.py

# Multiple binds
singularity exec \
    --bind /data/input:/input:ro \      # Read-only
    --bind /data/output:/output \        # Read-write
    container.sif python process.py
```

---

## 6. Complete Spark Job Example

### Job Script: `spark_weather.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=spark_weather
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --account=uci150
#SBATCH --output=logs/weather_%j.out
#SBATCH --error=logs/weather_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Start time: $(date)"

# Load Singularity
module load singularitypro

# Set paths
CONTAINER=/expanse/lustre/projects/uci150/$USER/ray_spark_dsc232r.sif
SCRIPT=/expanse/lustre/projects/uci150/$USER/scripts/weather_analysis.py
DATA=/expanse/lustre/projects/uci150/shared/weather_data

# Run the job
singularity exec \
    --bind /expanse \
    $CONTAINER \
    python $SCRIPT \
        --input $DATA \
        --output /scratch/$USER/$SLURM_JOB_ID/results \
        --cpus $SLURM_CPUS_PER_TASK

echo "End time: $(date)"
```

### Python Script: `weather_analysis.py`

```python
#!/usr/bin/env python3
"""
Weather data analysis on SDSC Expanse
Run with: sbatch spark_weather.slurm
"""

import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input data path')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--cpus', type=int, default=8, help='Number of CPUs')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("WeatherAnalysis") \
        .master(f"local[{args.cpus}]") \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "96g") \
        .config("spark.sql.shuffle.partitions", str(args.cpus * 2)) \
        .getOrCreate()

    print(f"Spark version: {spark.version}")
    print(f"Processing data from: {args.input}")

    # Load and process data
    df = spark.read.parquet(args.input)
    print(f"Total records: {df.count():,}")

    # Aggregate by station and year
    summary = df.groupBy("station_id", F.year("date").alias("year")) \
        .agg(
            F.avg("temperature").alias("avg_temp"),
            F.max("temperature").alias("max_temp"),
            F.min("temperature").alias("min_temp"),
            F.sum("precipitation").alias("total_precip")
        )

    # Save results
    output_path = os.path.join(args.output, "weather_summary")
    summary.write.mode("overwrite").parquet(output_path)
    print(f"Results saved to: {output_path}")

    spark.stop()

if __name__ == "__main__":
    main()
```

---

## 7. Job Monitoring and Debugging

### Monitoring Commands

```bash
# Check your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# View job output in real-time
tail -f logs/weather_12345.out

# Check job efficiency after completion
seff <job_id>

# View job accounting
sacct -j <job_id> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS
```

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Out of memory | Job killed, exit code 137 | Increase `--mem` or optimize code |
| Timeout | Job killed at time limit | Increase `--time` or parallelize more |
| Module not found | ImportError in logs | Check container has package installed |
| File not found | FileNotFoundError | Check `--bind` paths in Singularity |
| Job pending | PENDING state for hours | Check partition limits, try `debug` |

### Debugging Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEBUGGING WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 1. Test Locally     │
                    │    (small data)     │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 2. Test in debug    │
                    │    partition        │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 3. Check logs       │
                    │    (.out and .err)  │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 4. Use seff for     │
                    │    resource usage   │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 5. Adjust resources │
                    │    and resubmit     │
                    └─────────────────────┘
```

---

## 8. Interactive Sessions

### Starting an Interactive Job

```bash
# Quick interactive session for testing
srun --partition=debug --pty --nodes=1 --ntasks=1 \
     --cpus-per-task=4 --mem=16G --time=00:30:00 \
     --account=uci150 /bin/bash

# Interactive session with Singularity
srun --partition=debug --pty --nodes=1 --ntasks=1 \
     --cpus-per-task=8 --mem=32G --time=00:30:00 \
     --account=uci150 \
     singularity shell --bind /expanse ray_spark_dsc232r.sif
```

### Jupyter on Expanse

```bash
# Step 1: Start Jupyter in a job
# jupyter_job.slurm
#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=uci150
#SBATCH --output=jupyter_%j.log

module load singularitypro

# Get the node hostname
NODE=$(hostname)
PORT=8888

echo "============================================="
echo "Jupyter running on: $NODE:$PORT"
echo "To connect, run on your local machine:"
echo "  ssh -L $PORT:$NODE:$PORT $USER@login.expanse.sdsc.edu"
echo "Then open: http://localhost:$PORT"
echo "============================================="

singularity exec --bind /expanse ray_spark_dsc232r.sif \
    jupyter lab --no-browser --port=$PORT --ip=0.0.0.0

# Step 2: Create SSH tunnel from your laptop
ssh -L 8888:exp-15-01:8888 username@login.expanse.sdsc.edu

# Step 3: Open browser to http://localhost:8888
```

---

## Summary

### Key Concepts

1. **SLURM** schedules jobs on HPC clusters using `sbatch` and `srun`
2. **Partitions** control resource limits and queue priorities
3. **Singularity** provides portable, reproducible environments
4. **Spark configuration** must match SLURM resource allocation
5. **Monitoring** with `squeue`, `seff`, and log files is essential

### Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPARK ON SDSC WORKFLOW                        │
└─────────────────────────────────────────────────────────────────┘

1. Prepare Code
   └── Test locally with small data

2. Write SLURM Script
   └── Set appropriate resources

3. Submit Job
   └── sbatch my_job.slurm

4. Monitor
   └── squeue -u $USER
   └── tail -f output.log

5. Debug (if needed)
   └── Check .err file
   └── Use seff for resource usage

6. Iterate
   └── Adjust resources
   └── Resubmit
```

### Connecting to DSC 232R

| Prior Topic | Connection |
|-------------|------------|
| Spark DataFrames | Same API, just different deployment |
| Local[*] mode | Replace with SLURM-allocated resources |
| Class09-10: PCA | Run on larger datasets with HPC |
| Class13-15: XGBoost | Train on full datasets |

---

## Practice Problems

1. **Resource Estimation**: You have a 100 GB Parquet dataset. Estimate the memory and cores needed to process it efficiently on Expanse.

2. **Job Script**: Write a SLURM script that requests 64 cores and 256 GB memory for a 4-hour Spark job.

3. **Debugging**: Your job failed with exit code 137. What happened and how would you fix it?

---

*Next: Module 3 - Introduction to Ray*
