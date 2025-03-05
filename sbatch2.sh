#!/bin/bash

# Get job name from first argument (default if not provided)
JOB_NAME=${1:-"finetuning-projection"}
shift  # Remove first argument from argument list

# Create a temporary sbatch script with the dynamic job name
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" << EOL
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=/u/home/salzmann/Documents/dev/master-thesis/logs/sbatch_${JOB_NAME}_%j.out
#SBATCH --error=/u/home/salzmann/Documents/dev/master-thesis/logs/sbatch_${JOB_NAME}_%j.err
#SBATCH --mail-user=linus.salzmann@tum.de
#SBATCH --mail-type=ALL
#SBATCH --partition=universe
#SBATCH --qos=master-queuesave
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

# Load python module
ml python/anaconda3
conda init 

# Activate corresponding environment
conda deactivate
source activate vlm-detection2

# Run the program with remaining arguments
python /u/home/salzmann/Documents/dev/master-thesis/src/train.py $@
EOL

# Submit the job
sbatch "$TMP_SCRIPT"

# Clean up
rm "$TMP_SCRIPT"