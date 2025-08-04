<!-- # master-thesis
python 3.12
python -m venv venv
source venv/bin/activate 

`python src/train.py +experiment=train_local_test add_special_tokens=false`
`python src/train.py +experiment=train_local_test add_special_tokens=false train=false load_checkpoint=../checkpoints-trained/last_model_legendary-cloud-125.pt checkpoint_dir=.`
-->

# Master Thesis: Enhancing Localization in Multimodal Large Language Models with Specialized Detection Components

## Key Features

- Integration of DETR object detection with LLMs
- Feedback mechanism from detection to language model
- Support for multiple image encoders
- Precomputation of image features for faster training
- Evaluation metrics for localization quality

## Development Tools
- Hydra for configuration management
- Weights & Biases for experiment tracking
- PyTorch for deep learning
- Transformers library for model implementations


## Setup

```bash
# Create and activate a conda environment
conda env create -f environment.yml
conda activate vlm-detection

# Set up Weights & Biases for experiment tracking
wandb init

# Load Dataset
./dataset/coco.sh
```
## Training, Evaluation and Precomputation



```bash
# Precompute Image Features
python src/precompute.py +experiment=train_full freeze_model=True batch_size=128

# Training
python src/train.py +experiment=train_full detr_type=full_detr feedback_detr_to_llm=True train_detr=True

# Evaluate
python src/train.py +experiment=train_full detr_type=full_detr feedback_detr_to_llm=True train=False load_checkpoint=PATH_TO_CHECKPOINT
```

## Project Structure

### Model Architecture

- `src/model/model.py` - Main VisionLanguageModel implementation
- `src/model/detr_integration.py` - DETR detector integration modules
- `src/model/image_encoder.py` - Image encoder implementations (SigLIP, etc.)
- `src/model/partial_frozen_embeddings.py` - Custom embedding implementations

### Dataset & Processing

- `src/dataset/dataset.py` - COCO dataset loaders
- `src/dataset/processor.py` - Data transformation and tokenization
- `src/dataset/processor_fasterrcnn.py` - FasterRCNN-specific preprocessing

### Training & Evaluation

- `src/train.py` - Main training and evaluation script
- `src/precompute.py` - Precomputation of image features script
- `src/inference_example.ipynb` - Inference example notebook

### Configuration

- Experiment Configs
  - `conf/experiment/train_full.yaml` - Full training configuration
  - `conf/experiment/train_local_test.yaml` - Local testing configuration

- Dataset Configs
  - `conf/dataset/coco_train.yaml`, `dataset/coco_val.yaml`

- Model Configs
  - `conf/model/` - Model architecture configurations
  - `conf/image_encoder/` - Image encoder configurations


### Utilities

- `utils/config.py` - Configuration classes
- `utils/train_utils.py` - Training helpers
- `utils/train_metrics.py` - Evaluation metrics

### Data & Resources

`data` - Dataset storage
`data/coco.sh` - Script for downloading COCO dataset
`checkpoints` - Model checkpoint storage

