#!/bin/bash
#SBATCH --job-name=eval_llama
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.out
#SBATCH --array=0-3
#SBATCH --time=24:00:00


# Conda activation
source ~/.bashrc
conda activate s1

# Array job setup

i=$SLURM_ARRAY_TASK_ID
begin=$((i*25))
end=$(((i+1)*25))

echo "Running task ID: $i, iterating from $begin to $end"


python infer.py --config_file ./configs/attn_a16.yaml --start_index $begin --stop_index $end
