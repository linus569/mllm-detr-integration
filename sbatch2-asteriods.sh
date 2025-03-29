#!/bin/bash

# Get job name from first argument (default if not provided)
JOB_NAME=${1:-"finetuning-projection"}
shift  # Remove first argument from argument list

# Create a temporary sbatch script with the dynamic job name
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" << EOL
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=logs/sbatch_${JOB_NAME}_%j.out
#SBATCH --error=logs/sbatch_${JOB_NAME}_%j.err
#SBATCH --mail-user=linus.salzmann@tum.de
#SBATCH --mail-type=ALL
#SBATCH --partition=asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

# asteroids,universe

# Load python module
ml python/anaconda3
conda init

# Activate corresponding environment
#conda deactivate
source activate vlm-detection2
conda list

# cache to /tmp with rsync

# Run the program with remaining arguments
python src/train.py +experiment=train_finetune_projection $@
EOL

# Submit the job
sbatch "$TMP_SCRIPT"

# Clean up
rm "$TMP_SCRIPT"
