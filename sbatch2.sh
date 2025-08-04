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
#SBATCH --partition=universe
#SBATCH --qos=master-queuesave
#SBATCH --time=0-100:00:00 # 100:00:00, 48, 150:
#SBATCH --gres=gpu:1 # :a100:
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

# asteriods,universe

# Load python module
ml python/anaconda3
conda init

# Activate corresponding environment
#conda deactivate
source activate vlm-detection-new
conda list

# cache to /tmp with rsync
# rsync --progress precomputed_img_siglip_bs2_bfloat16.hdf5 /tmp/precomputed_img_siglip_bs2_bfloat16.hdf5 # not faster, even slower
# catch precomputed data file precomputed_img_siglip_bs2_bfloat16.hdf5
# rsync --progress precomputed_img_siglip_bs2_bfloat16.hdf5 /tmp/precomputed_img_siglip_bs2_bfloat16.hdf5

# Run the program with remaining arguments
# python src/precompute.py +experiment=train_finetune_projection $@
python src/train.py +experiment=train_full $@
# python src/train.py +experiment=train_stage_1_detr $@
# python src/train.py +experiment=train_stage_2_detr_llm $@
EOL

# Submit the job
sbatch "$TMP_SCRIPT"

# Clean up
rm "$TMP_SCRIPT"
