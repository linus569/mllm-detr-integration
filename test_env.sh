#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --output=test_env.out
#SBATCH --error=test_env.err
#SBATCH --partition=universe
#SBATCH --qos=master-queuesave
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

ml python/anaconda3
source ~/.bashrc
conda init
conda activate vlm-detection2
python -c "import albumentations; print('Albumentations is working!')"
