"""
Test to verify that the number of image features calculated in the processor
matches the actual number produced by the image encoder.
"""

import logging
import math
import os
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader, Subset

# Add the project root to the Python path to make imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset.dataset import COCODataset
from dataset.processor import Processor
from model.model import VisionLanguageModel
from utils.config import DatasetConfig, ExperimentConfig
from utils.train_utils import build_train_dataloader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="ExperimentConfig", node=ExperimentConfig)
cs.store(name="DatasetConfig", group="dataset", node=DatasetConfig)


@pytest.fixture
def config():
    """Return a configuration using hydra."""
    with initialize(version_base="1.3", config_path="../../conf"):
        # Load configuration from the default.yaml file
        cfg = compose(
            config_name="train",
            overrides=["+experiment=train_local_test", "main_dir='.'", "image_encoder=resnet50"],
        )
        return ExperimentConfig(**cfg)


@pytest.fixture
def processor(config):
    """Return initialized processor."""
    return Processor.from_config(config, add_special_tokens=True)


@pytest.fixture
def model(config, processor):
    """Return initialized model."""
    model = VisionLanguageModel(
        config=config,
        image_token_index=processor.image_token_index,
        num_new_tokens=len(processor.special_tokens),
        tokenizer_size=len(processor.tokenizer),
        initializers=processor.special_tokens_initializer,
        do_init=True,
    )
    model.eval()
    return model


@pytest.mark.parametrize("batch_size", [2])
def test_image_features_count(config, processor, model, batch_size):
    """
    Test that the number of image features calculated in the processor
    matches the actual output from the image encoder.

    Args:
        config: Test configuration
        processor: Initialized processor
        model: Initialized model
        dataset: Test dataset
        batch_size: Number of samples to process at once
    """
    # Take a small subset for testing
    config.batch_size = batch_size
    num_samples = None#5
    dataloader = build_train_dataloader(config, processor, subset_size=num_samples)

    # Test each batch
    mismatches = 0
    total = 0

    for batch in dataloader:
        total += len(batch["images"])
        images = batch["images"]
        image_sizes = batch["image_sizes"]

        with torch.no_grad():
            if "resnet50" in config.image_encoder.name:
                # For ResNet50, we need image sizes
                image_features = model._get_image_features(images, image_sizes)
            else:
                image_features = model._get_image_features(images, None)

            # currently just without patches
            actual_features = image_features.shape[0] * image_features.shape[1]

            # Calculate expected number of features based on processor logic
            if "resnet50" in config.image_encoder.name:
                # For ResNet50, features depend on image dimensions
                img_h, img_w = images.shape[2], images.shape[3]
                expected_features = batch_size * (math.ceil(img_w / 32) * math.ceil(img_h / 32))
            else:
                # For other encoders, use configured number of tokens
                expected_features = batch_size * processor.num_image_tokens

            # Compare
            assert actual_features == expected_features, (
                f"Expected {expected_features} features but got {actual_features}. "
                f"Image shape: {images.shape}, Features shape: {image_features.shape}, Image sizes: {image_sizes}"
            )
            # log.info(
            #     f"Correct feature count {actual_features}. "
            #     f"Images shape: {images.shape}. "
            #     f"Image sizes: {image_sizes}"
            # )
            if total % 1000 == 0:
                log.info(
                    f"Processed {total} samples. "
                    f"Expected features: {expected_features}, Actual features: {actual_features}"
                )

    # Assert no mismatches
    assert (
        mismatches == 0
    ), f"Found {mismatches}/{total} mismatches in image feature count"
    log.info(f"All {total} samples have correct image feature counts")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
