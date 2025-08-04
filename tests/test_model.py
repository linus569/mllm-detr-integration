import os
import sys

import pytest
import torch

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset.processor import Processor
from model.model import VisionLanguageModel
from utils.config import ExperimentConfig


@pytest.fixture
def model_config(mocker):
    config = ExperimentConfig()
    mocker.patch.object(config, "model_name", "lmms-lab/llava-onevision-qwen2-0.5b-si")
    mocker.patch.object(config, "image_size", (384, 384))
    mocker.patch.object(config, "image_token", "<image>")
    mocker.patch.object(config, "num_image_tokens", 729)
    mocker.patch.object(config, "max_tokens", 2048)
    mocker.patch.object(config, "pad_to_multiple_of", 8)
    mocker.patch.object(config, "num_coordinate_bins", 10)
    return config


@pytest.fixture
def processor(model_config):
    return Processor.from_config(model_config, add_special_tokens=True)


@pytest.fixture
def model(model_config, processor):
    return VisionLanguageModel(
        config=model_config,
        image_token_index=processor.image_token_index,
        num_new_tokens=len(processor.special_tokens),
        initializers=processor.special_tokens_initializer,
        do_init=True,
    )


def test_model_forward(model):
    batch_size = 2
    seq_length = 2000
    img_size = 384

    # Create sample inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    images = torch.rand(batch_size, 3, img_size, img_size)
    labels = torch.randint(0, 1000, (batch_size, seq_length))

    # Add 729x image_tokens to input_ids
    num_img_tokens = 729
    image_token_id = model.image_token_index
    # Add image tokens at the beginning of the sequence
    image_tokens = torch.full((batch_size, num_img_tokens), image_token_id)
    # Generate random starting positions for image tokens
    start_positions = torch.randint(0, seq_length - num_img_tokens, (batch_size,))
    for i in range(batch_size):
        input_ids[i, start_positions[i] : start_positions[i] + num_img_tokens] = (
            image_tokens[i]
        )

    # Test forward pass without labels
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
    assert outputs.logits.shape == (batch_size, seq_length, model.vocab_size)

    # Test forward pass with labels
    outputs_with_labels = model(
        input_ids=input_ids, attention_mask=attention_mask, images=images, labels=labels
    )
    assert outputs_with_labels.logits.shape == (
        batch_size,
        seq_length,
        model.vocab_size,
    )
    assert outputs_with_labels.loss is not None
    assert isinstance(outputs_with_labels.loss, torch.Tensor)
    assert outputs_with_labels.loss.dim() == 0  # Loss should be a scalar


def test_model_forward_patches(model):
    batch_size = 2
    seq_length = 6000
    img_size = 384

    # Test with image patches
    # Create sample inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    labels = torch.randint(0, 1000, (batch_size, seq_length))
    image_patches = torch.rand(batch_size, 4, 3, img_size, img_size)  # 4 patches

    # Add 729x4patches image_tokens to input_ids
    num_img_patch_tokens = 729 * 4
    image_token_id = model.image_token_index
    # Add image tokens at the beginning of the sequence
    image_tokens = torch.full((batch_size, num_img_patch_tokens), image_token_id)
    # Generate random starting positions for image tokens
    start_positions = torch.randint(0, seq_length - num_img_patch_tokens, (batch_size,))
    for i in range(batch_size):
        input_ids[i, start_positions[i] : start_positions[i] + num_img_patch_tokens] = (
            image_tokens[i]
        )

    outputs_patches = model(
        input_ids=input_ids, attention_mask=attention_mask, images=image_patches
    )
    assert outputs_patches.logits.shape == (
        batch_size,
        seq_length,
        model.vocab_size,
    )
