import os
import sys

import pytest
import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset.processor import Processor
from utils.config import DatasetConfig, ExperimentConfig

OmegaConf.register_new_resolver(
    "ifel", lambda flag, val_true, val_false: val_true if flag else val_false
)

NUM_COORDINATE_BINS = 10


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
def processor(config: ExperimentConfig):
    config.num_coordinate_bins = NUM_COORDINATE_BINS
    return Processor.from_config(config, add_special_tokens=True)

def test_parse_coordinate_to_special_token(processor):
    # Test valid input
    valid_input = [0.1, 0.2, 0.3, 0.4]
    result = processor._parse_coordinate_to_special_token(valid_input)
    assert result == [1, 3, 4, 8]

    # Test input with invalid values
    invalid_input = [0.1, 0.2, "invalid", 0.4]
    with pytest.raises(ValueError):
        processor._parse_coordinate_to_special_token(invalid_input)

    # Test input with invalid number of coordinates
    invalid_input = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError):
        processor._parse_coordinate_to_special_token(invalid_input)

def test_parse_model_output_valid(processor):
    # Test valid input with multiple objects
    valid_input = "[{'class': 'dog', 'bbox': ['<coord_1>',<coord_3>, '<coord_4>', '<coord_8>']}, {'class': 'cat', 'bbox': [0.5, 0.6, 0.7, 0.8]}]"
    result = processor._parse_model_output(valid_input)
    print(result)

    bin_size = 1 / (NUM_COORDINATE_BINS - 1)

    assert result is not None
    assert len(result) == 2
    assert result[0]["class"] == "dog"
    assert result[0]["bbox"] == [1 * bin_size, 3 * bin_size, 4 * bin_size, 8 * bin_size]
    assert result[1]["class"] == "cat"
    assert result[1]["bbox"] == [0.5, 0.6, 0.7, 0.8]


def test_parse_model_output_invalid_format(processor):
    # Test invalid JSON format
    invalid_input = "not a json"
    assert processor._parse_model_output(invalid_input) == []

    # Test missing required fields
    missing_fields = "[{'class': 'dog'}]"
    assert processor._parse_model_output(missing_fields) == []


def test_parse_model_output_invalid_bbox(processor):
    # Test invalid bbox format (wrong number of coordinates)
    invalid_bbox = "[{'class': 'dog', 'bbox': [0.1, 0.2, 0.3]}]"
    assert processor._parse_model_output(invalid_bbox) == []

    # Test invalid bbox values (non-numeric)
    invalid_bbox_values = "[{'class': 'dog', 'bbox': ['invalid', 0.2, 0.3, 0.4]}]"
    assert processor._parse_model_output(invalid_bbox_values) == []


def test_parse_model_output_with_text_around(processor):
    # Test input with additional text around the JSON
    text_around = "Here is the output: [{'class': 'dog', 'bbox': [0.1, 0.2, 0.3, 0.4]}] End of output"
    result = processor._parse_model_output(text_around)
    assert result is not None
    assert len(result) == 1
    assert result[0]["class"] == "dog"
    assert result[0]["bbox"] == [0.1, 0.2, 0.3, 0.4]


class MockDataset:
    def __init__(self):
        self.cat_name_to_id = {"dog": 0, "cat": 1, "person": 2}


@pytest.fixture
def mock_dataset():
    return MockDataset()


def test_parse_model_output_to_boxes_valid(mock_dataset, processor):
    device = "cpu"
    text = ["[{'class': 'dog', 'bbox': [<coord_1>, <coord_2>, <coord_3>, <coord_4>]}]"]

    sequences = processor.tokenizer(text).input_ids

    _, result = processor.postprocess_json_batch(sequences, mock_dataset, device)
    assert result is not None
    assert isinstance(result, list)
    assert "boxes" in result[0]
    assert "labels" in result[0]
    assert "scores" in result[0]

    assert result[0]["boxes"].shape == (1, 4)
    assert result[0]["labels"].shape == (1,)
    assert result[0]["scores"].shape == (1,)

    # Check if bbox is properly unnormalized (multiplied by 384)
    bin_size = (1 / (NUM_COORDINATE_BINS - 1)) * 384
    expected_box = torch.tensor(
        [[1 * bin_size, 2 * bin_size, 3 * bin_size, 4 * bin_size]]
    )
    assert torch.allclose(result[0]["boxes"], expected_box)
    assert result[0]["labels"][0] == 0  # dog id
    assert result[0]["scores"][0] == 1.0


def test_parse_model_output_to_boxes_multiple_objects(mock_dataset, processor):
    device = "cpu"
    text = [
        "[{'class': 'dog', 'bbox': [0.1, 0.2, 0.3, 0.4]}, {'class': 'cat', 'bbox': [0.9, 1.0, 1.1, 1.2]}]"
    ]
    sequences = processor.tokenizer(text).input_ids
    _, result = processor.postprocess_json_batch(sequences, mock_dataset, device)

    assert result[0]["boxes"].shape == (2, 4)
    assert result[0]["labels"].shape == (2,)
    assert result[0]["scores"].shape == (2,)

    assert result[0]["labels"].tolist() == [0, 1]  # [dog_id, cat_id]
    assert all(score == 1.0 for score in result[0]["scores"])


def test_parse_model_output_to_boxes_batch(mock_dataset, processor):
    device = "cpu"
    text = [
        "[{'class': 'dog', 'bbox': [0.1, 0.2, 0.3, 0.4]}]",
        "[{'class': 'cat', 'bbox': [0.9, 1.0, 1.1, 1.2]}]",
    ]
    sequences = processor.tokenizer(text).input_ids
    _, result = processor.postprocess_json_batch(sequences, mock_dataset, device)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2

    assert result[0]["boxes"].shape == (1, 4)
    assert result[0]["labels"].shape == (1,)
    assert result[0]["scores"].shape == (1,)
    assert result[0]["labels"][0] == 0  # dog id

    assert result[1]["boxes"].shape == (1, 4)
    assert result[1]["labels"].shape == (1,)
    assert result[1]["scores"].shape == (1,)
    assert result[1]["labels"][0] == 1  # cat id

    # Check if bbox is properly unnormalized (multiplied by 384)
    expected_box_0 = torch.tensor([38.4, 76.8, 115.2, 153.6])
    expected_box_1 = torch.tensor([345.6, 384.0, 384, 384])
    assert torch.allclose(result[0]["boxes"], expected_box_0)
    assert torch.allclose(result[1]["boxes"], expected_box_1)
    assert result[0]["scores"][0] == 1.0
    assert result[1]["scores"][0] == 1.0
    assert result[0]["labels"][0] == 0  # dog id
    assert result[1]["labels"][0] == 1  # cat id

@pytest.mark.parametrize("text", ["not a valid json", "[]"])
def test_parse_model_output_to_boxes_invalid_input(mock_dataset, processor, text):
    device = "cpu"

    sequences = processor.tokenizer(text).input_ids
    _, result = processor.postprocess_json_batch(sequences, mock_dataset, device)

    assert result[0]["boxes"].shape == (0, 4)
    assert result[0]["labels"].shape == (0,)
    assert result[0]["scores"].shape == (0,)


def test_parse_model_output_to_boxes_unknown_class(mock_dataset, processor):
    device = "cpu"
    text = ["[{'class': 'unknown', 'bbox': [0.1, 0.2, 0.3, 0.4]}]"]
    
    sequences = processor.tokenizer(text).input_ids
    _, result = processor.postprocess_json_batch(sequences, mock_dataset, device)

    assert result[0]["labels"][0] == -1  # Unknown class should get id -1
