#!/bin/bash
#SBATCH --job-name=eval_llama
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.out
#SBATCH --array=0-4
#SBATCH --time=24:00:00


# Conda activation
source ~/.bashrc
conda activate s1

# Array job setup

i=$SLURM_ARRAY_TASK_ID
begin=$((i*20))
end=$(((i+1)*20))

echo "Running task ID: $i, iterating from $begin to $end"


python infer.py --config_file ./configs/eval.yaml --start_index $begin --stop_index $end --checkpoint_path "./checkpoints/all_rank_16_alpha_16_epoch_3/" --save_dir "./out/all_r16_e3f/run1/" 
python infer.py --config_file ./configs/eval.yaml --start_index $begin --stop_index $end --checkpoint_path "./checkpoints/all_rank_16_alpha_16_epoch_3/" --save_dir "./out/all_r16_e3f/run2/" 
python infer.py --config_file ./configs/eval.yaml --start_index $begin --stop_index $end --checkpoint_path "./checkpoints/all_rank_16_alpha_16_epoch_3/" --save_dir "./out/all_r16_e3f/run3/" 
python infer.py --config_file ./configs/eval.yaml --start_index $begin --stop_index $end --checkpoint_path "./checkpoints/all_rank_16_alpha_16_epoch_3/" --save_dir "./out/all_r16_e3f/run4/" 
python infer.py --config_file ./configs/eval.yaml --start_index $begin --stop_index $end --checkpoint_path "./checkpoints/all_rank_16_alpha_16_epoch_3/" --save_dir "./out/all_r16_e3f/run5/" 
p
