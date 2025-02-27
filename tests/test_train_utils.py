import os
import sys

import pytest
import torch

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils.train_utils import parse_model_output, parse_model_output_to_boxes


def test_parse_model_output_valid():
    # Test valid input with multiple objects
    valid_input = "[{'class': 'dog', 'bbox': [0.1, 0.2, 0.3, 0.4]}, {'class': 'cat', 'bbox': [0.5, 0.6, 0.7, 0.8]}]"
    result = parse_model_output(valid_input)
    print(result[0]["class"])
    assert result is not None
    assert len(result) == 2
    assert result[0]["class"] == "dog"
    assert result[0]["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert result[1]["class"] == "cat"
    assert result[1]["bbox"] == [0.5, 0.6, 0.7, 0.8]


def test_parse_model_output_invalid_format():
    # Test invalid JSON format
    invalid_input = "not a json"
    assert parse_model_output(invalid_input) is None

    # Test missing required fields
    missing_fields = "[{'class': 'dog'}]"
    assert parse_model_output(missing_fields) is None


def test_parse_model_output_invalid_bbox():
    # Test invalid bbox format (wrong number of coordinates)
    invalid_bbox = "[{'class': 'dog', 'bbox': [0.1, 0.2, 0.3]}]"
    assert parse_model_output(invalid_bbox) is None

    # Test invalid bbox values (non-numeric)
    invalid_bbox_values = "[{'class': 'dog', 'bbox': ['invalid', 0.2, 0.3, 0.4]}]"
    assert parse_model_output(invalid_bbox_values) is None


def test_parse_model_output_with_text_around():
    # Test input with additional text around the JSON
    text_around = "Here is the output: [{'class': 'dog', 'bbox': [0.1, 0.2, 0.3, 0.4]}] End of output"
    result = parse_model_output(text_around)
    assert result is not None
    assert len(result) == 1
    assert result[0]["class"] == ["dog"]
    assert result[0]["bbox"] == [[0.1, 0.2, 0.3, 0.4]]


class MockDataset:
    def __init__(self):
        self.cat_name_to_id = {"dog": 0, "cat": 1, "person": 2}


@pytest.fixture
def mock_dataset():
    return MockDataset()


def test_parse_model_output_to_boxes_valid(mock_dataset):
    device = "cpu"
    text = ["[{'class': 'dog', 'bbox': [0.1, 0.2, 0.3, 0.4]}]"]

    result = parse_model_output_to_boxes(text, mock_dataset, device)

    assert result is not None
    assert isinstance(result, list)
    assert "boxes" in result[0]
    assert "labels" in result[0]
    assert "scores" in result[0]

    assert result[0]["boxes"].shape == (1, 4)
    assert result[0]["labels"].shape == (1,)
    assert result[0]["scores"].shape == (1,)

    # Check if bbox is properly unnormalized (multiplied by 384)
    expected_box = torch.tensor([[38.4, 76.8, 115.2, 153.6]])
    assert torch.allclose(result[0]["boxes"], expected_box)
    assert result[0]["labels"][0] == 0  # dog id
    assert result[0]["scores"][0] == 1.0


def test_parse_model_output_to_boxes_multiple_objects(mock_dataset):
    device = "cpu"
    text = [
        "[{'class': 'dog', 'bbox': [0.1, 0.2, 0.3, 0.4]}, {'class': 'cat', 'bbox': [0.9, 1.0, 1.1, 1.2]}]"
    ]

    result = parse_model_output_to_boxes(text, mock_dataset, device)

    assert result[0]["boxes"].shape == (2, 4)
    assert result[0]["labels"].shape == (2,)
    assert result[0]["scores"].shape == (2,)

    assert result[0]["labels"].tolist() == [0, 1]  # [dog_id, cat_id]
    assert all(score == 1.0 for score in result[0]["scores"])


def test_parse_model_output_to_boxes_batch(mock_dataset):
    device = "cpu"
    text = [
        "[{'class': 'dog', 'bbox': [0.1, 0.2, 0.3, 0.4]}]",
        "[{'class': 'cat', 'bbox': [0.9, 1.0, 1.1, 1.2]}]",
    ]

    result = parse_model_output_to_boxes(text, mock_dataset, device)

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
    expected_box_1 = torch.tensor([345.6, 384.0, 422.4, 460.8])
    assert torch.allclose(result[0]["boxes"], expected_box_0)
    assert torch.allclose(result[1]["boxes"], expected_box_1)
    assert result[0]["scores"][0] == 1.0
    assert result[1]["scores"][0] == 1.0
    assert result[0]["labels"][0] == 0  # dog id
    assert result[1]["labels"][0] == 1  # cat id


def test_parse_model_output_to_boxes_invalid_input(mock_dataset):
    device = "cpu"
    invalid_text = ["not a valid json"]

    result = parse_model_output_to_boxes(invalid_text, mock_dataset, device)

    assert result[0]["boxes"].shape == (0, 4)
    assert result[0]["labels"].shape == (0,)
    assert result[0]["scores"].shape == (0,)


def test_parse_model_output_to_boxes_empty_predictions(mock_dataset):
    device = "cpu"
    empty_text = ["[]"]

    result = parse_model_output_to_boxes(empty_text, mock_dataset, device)

    assert result[0]["boxes"].shape == (0, 4)
    assert result[0]["labels"].shape == (0,)
    assert result[0]["scores"].shape == (0,)


def test_parse_model_output_to_boxes_unknown_class(mock_dataset):
    device = "cpu"
    text = ["[{'class': 'unknown', 'bbox': [0.1, 0.2, 0.3, 0.4]}]"]

    result = parse_model_output_to_boxes(text, mock_dataset, device)

    assert result[0]["labels"][0] == -1  # Unknown class should get id -1
