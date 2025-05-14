#!/bin/bash
#SBATCH --job-name=eval_llama_naive
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.out
#SBATCH --array=0-9
#SBATCH --time=24:00:00


# Conda activation
source ~/.bashrc
conda activate s1

# Array job setup

i=$SLURM_ARRAY_TASK_ID
begin=$((i*10))
end=$(((i+1)*10))

echo "Running task ID: $i, iterating from $begin to $end"


python infer.py --config_file ./configs/eval_naive.yaml --start_index $begin --stop_index $end --save_dir "./out/naive_noicl/run1/" --no_statutes
#python infer.py --config_file ./configs/eval_naive.yaml --start_index $begin --stop_index $end --save_dir "./out/naive/run4/"
#python infer.py --config_file ./configs/eval_naive.yaml --start_index $begin --stop_index $end --save_dir "./out/naive/run5/"
