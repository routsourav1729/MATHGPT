#!/bin/bash

#SBATCH --job-name=gpt2-training
#SBATCH --error=slurm_logs/gpt2.%J.err  
#SBATCH --output=gpt2-training-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=2

# --- Request 2 GPUs for DDP training ---
#SBATCH --gres=gpu:2
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=60:00:00

# --- Performance optimization ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "========================================================"
echo "SLURM JOB: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "========================================================"

cd /home/agipml/sourav.rout/ALL_FILES/project/mathgpt/src

echo ">>> Starting GPT-2 training with DDP..."

# Run with torchrun for distributed training
torchrun --nproc_per_node=2 train.py > training.out 2>&1

echo ">>> GPT-2 training finished."