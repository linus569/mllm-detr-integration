import os
import sys

import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from llava.mm_utils import process_images
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.config import ExperimentConfig

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset.processor import Processor
from model.model import VisionLanguageModel
from utils.config import DatasetConfig
from utils.train_utils import build_train_dataloader

OmegaConf.register_new_resolver(
    "ifel", lambda flag, val_true, val_false: val_true if flag else val_false
)


def config():
    """Fixture to load and return the Hydra configuration."""
    cs = ConfigStore.instance()
    cs.store(name="ExperimentConfig", node=ExperimentConfig)
    cs.store(name="DatasetConfig", group="dataset", node=DatasetConfig)

    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name="train",
            overrides=["+experiment=train_local_test", "main_dir='.'"],
        )
        return config


def process_image_patches(model, images):
    """Process image patches and combine their features."""
    B, N, C, H, W = images.shape  # Batch, NumPatches, Channels, Height, Width

    # Reshape to process all patches at once
    images = images.view(B * N, C, H, W)

    # TODO: image should be of type list on top so image encoder encodes all images

    # Extract features for all patches
    features = model.image_encoder(images)  # Shape: [B*N, num_tokens, hidden_size]

    # Project features
    features = model.projector(features)  # Shape: [B*N, num_tokens, hidden_size]

    # Reshape back to separate batch and patches
    B_N, T, D = features.shape  # BatchPatches, Tokens, Hidden
    features = features.view(B, N * T, D)  # Combine patch tokens for each batch

    return features


def test_dataset_processor_model(config):
    # test implementation of dataset, processor, and integration with model
    train_data_dir = "data/coco/images/train2017"
    train_annotation_file = "data/coco/annotations/instances_train2017.json"
    model_name = "lmms-lab/llava-onevision-qwen2-0.5b-si"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = VisionLanguageModel(config=config)
    # print(model)
    print(model.model.config)
    print(model.image_encoder.config)
    # print(model.image_size) # image_size moved from model to processor

    # print("projector model", model.text_encoder.projector)
    # print("projector", model.projector)

    # print number of image-tokens when forward in vision tower
    # print(
    #     "vision_tower output shape:",
    #     model.image_encoder.forward(torch.randn(1, 3, 384, 384)).shape,
    # )
    # print(
    #     "num_image_token of vision_tower:",
    #     model.image_encoder.vision_tower.vision_model.embeddings.position_embedding.weight.shape[0],
    # )

    dataloader = build_train_dataloader(config, model)

    batch = next(iter(dataloader))
    images = batch["images"]
    input_ids = batch["input_ids"]
    att_mask = batch["attention_mask"]

    # Process images with llava library
    test_encoder_manually = False
    llava = False
    if test_encoder_manually and llava:
        process_img = process_images(
            images, model.image_encoder.image_processor, model.model.config
        )
        encoded_images = []
        for batch_imgs in process_img:
            encoded_images.append(model.image_encoder(batch_imgs))

        # convert list to tensor
        images = torch.stack(encoded_images)
    elif test_encoder_manually:
        encoded_images = model.image_encoder.forward(images)
        print(encoded_images.shape)

    # # Process image patches
    # if images.dim() == 5:  # We have patches
    #     image_features = process_image_patches(model, images)
    # else:  # Regular image processing
    #     image_features = model.image_encoder(images)
    #     image_features = self.projector(image_features)

    # encoded_images = model.image_encoder.forward(images)
    # print("encoded_images shape:", encoded_images.shape)

    labels = input_ids.clone()

    model_output = model(
        input_ids=input_ids, attention_mask=att_mask, images=images, labels=labels
    )

    print(model_output)


def test_token_size(config):
    model_name = "lmms-lab/llava-onevision-qwen2-0.5b-si"
    config.batch_size = 1
    config.max_tokens = None
    config.pad_to_multiple_of = None

    model = VisionLanguageModel(config=config)
    dataloader = build_train_dataloader(config, model)

    token_sizes = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        token_size = batch["input_ids"].shape[1]
        token_sizes.append(token_size)

    # Calculate statistics
    max_size = max(token_sizes)
    min_size = min(token_sizes)
    avg_size = sum(token_sizes) / len(token_sizes)

    print(f"Token size statistics:")
    print(f"Max: {max_size}")
    print(f"Min: {min_size}")
    print(f"Average: {avg_size:.2f}")
    print(f"Number of samples: {len(token_sizes)}")


if __name__ == "__main__":
    # get command line argument to test dataset-processor or token-size
    # python src/test_implementation.py token-size
    test_name = sys.argv[1] if len(sys.argv) > 1 else ""

    config = config()

    print("Test:", test_name)
    test_token_size(config)

    # if test_name == "token-size":
    #    test_token_size()
    # else:
    #    test_dataset_processor_model()
