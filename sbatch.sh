#!/bin/bash
 
#SBATCH --job-name=finetuning-projection
#SBATCH --output=/u/home/salzmann/Documents/dev/master-thesis/logs/sbatch_training.out
#SBATCH --error=/u/home/salzmann/Documents/dev/master-thesis/logs/sbatch_training.err
#SBATCH --mail-user=linus.salzmann@tum.de
#SBATCH --mail-type=ALL
#SBATCH --partition=universe
#SBATCH --qos=master-queuesave
#SBATCH --time=0-10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16 
#SBATCH --mem=48G

##gpu:a100:1
##cpus: 16 oder max 24
##--partition=universe,asteroids universe a40 and a100, asteroidx rtx 3090 (workstations)
 
# Load python module
ml python/anaconda3
conda init 
# kann auch außerhalb des Skripts gesetzt werden

# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that. Not necessary if you always run this script from a clean terminal
conda deactivate
 
# If the following does not work, try 'source activate <env-name>'
#conda activate vlm-detection
source activate vlm-detection

# Cache data to local /tmp directory (optional)
#rsync -r /vol/aimspace/projects/<dataset> /tmp
 
# Run the program
python /u/home/salzmann/Documents/dev/master-thesis/src/train.py +experiment=train_finetune_projection