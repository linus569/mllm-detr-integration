import os
import sys

import pytest
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset.processor import Processor
from utils.config import DatasetConfig, ExperimentConfig
from utils.train_utils import build_dataloader

OmegaConf.register_new_resolver(
    "ifel", lambda flag, val_true, val_false: val_true if flag else val_false
)


@pytest.fixture
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


@pytest.fixture
def processor(config):
    """Fixture to create and return the tokenizer."""
    return Processor.from_config(config, add_special_tokens=True)


@pytest.fixture
def dataloader(config, processor):
    """Fixture to create and return the dataloader."""
    return build_dataloader(
        processor=processor,
        dataset_config=config.train_dataset,
        batch_size=2,
        is_train=True,
        num_workers=config.num_workers,
        subset_size=10,
    )


def test_padding_position(dataloader, processor):
    """
    Test that padding tokens are correctly positioned on the right side
    of the input sequences.
    """
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]
    padding_token_id = processor.tokenizer.pad_token_id

    for ids in input_ids:
        ids_list = ids.tolist()
        if padding_token_id in ids_list:
            padding_start = ids_list.index(padding_token_id)
            all_padding = all(
                token == padding_token_id for token in ids_list[padding_start:]
            )
            assert all_padding, "Padding tokens should be contiguous on the right side"


def test_batch_size(dataloader):
    """Test that the dataloader produces batches of the expected size."""
    batch = next(iter(dataloader))
    assert batch["input_ids"].shape[0] == 2, "Batch size should be 2"


def test_input_ids_shape(dataloader):
    """Test that input_ids have the expected shape."""
    batch = next(iter(dataloader))
    assert len(batch["input_ids"].shape) == 2, "input_ids should be 2-dimensional"
