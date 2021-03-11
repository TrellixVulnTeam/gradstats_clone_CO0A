#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -o out_%j.out

srun /shared/scaling_without_tuning/training_scripts/train_mrcnn_fp16.sgdw.giou.measure.sh

