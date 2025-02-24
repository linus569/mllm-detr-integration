import dataclasses
import json
import logging
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import torch
from omegaconf import MISSING, OmegaConf
from transformers import StoppingCriteria

from dataset.dataset import DatasetConfig, build_dataloader

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    name: str = MISSING
    seed: int = MISSING

    train_dataset: DatasetConfig = MISSING
    val_dataset: DatasetConfig = MISSING
    test_dataset: DatasetConfig = MISSING

    main_dir: str = MISSING
    model_name: str = "lmms-lab/llava-onevision-qwen2-0.5b-si"

    checkpoint_dir: str = "checkpoints"

    train: bool = True
    evaluate: bool = True
    eval_mode: str = "val"
    # eval_tasks: Dict[str, Any] = field(default_factory=dict)

    num_samples: Optional[int] = None
    val_num_samples: Optional[int] = None
    max_tokens: int = MISSING

    batch_size: int = MISSING
    epochs: int = MISSING
    # max_steps: Optional[int] = None
    # max_epochs: Optional[int] = None
    lr: float = MISSING
    # min_lr: float = MISSING
    # warmup_lr: Optional[float] = MISSING
    warmup_ratio: float = MISSING
    weight_decay: Optional[float] = MISSING
    gradient_accumulation_steps: int = MISSING
    max_grad_norm: Optional[float] = MISSING
    # grad_clip_norm: Optional[float] = MISSING
    # early_stopping_patience: Optional[int] = None

    # metric: Optional[str] = None
    # metric_mode: str = 'max'

    val_freq: Optional[int] = None
    val_ep: Optional[int] = None
    print_freq: int = MISSING
    num_workers: int = MISSING
    device: str = MISSING
    debug: bool = False
    # compile: bool = True
    save_components: List[str] = field(default_factory=list)


def build_train_dataloader(config: ExperimentConfig, model, subset_size=None):
    return build_dataloader(
        config=config,
        dataset_config=config.train_dataset,
        batch_size=config.batch_size,
        tokenizer=model.tokenizer,
        is_train=True,
        num_workers=config.num_workers,
        # image_size=config.transform.image_size,
        num_image_tokens=model.num_img_tokens,
        subset_size=subset_size,
        # use_random_subset=True,
    )


def build_val_dataloader(
    config: ExperimentConfig, model, subset_size=None, use_random_subset=True
):
    return build_dataloader(
        config=config,
        dataset_config=config.val_dataset,
        batch_size=config.batch_size,
        tokenizer=model.tokenizer,
        is_train=False,
        num_workers=config.num_workers,
        # image_size=config.transform.image_size,
        num_image_tokens=model.num_img_tokens,
        subset_size=subset_size,
        use_random_subset=use_random_subset,
    )


def build_test_dataloader(
    config: ExperimentConfig, model, subset_size=None, use_random_subset=True
):
    return build_dataloader(
        config=config,
        dataset_config=config.train_dataset,
        batch_size=config.batch_size,
        tokenizer=model.tokenizer,
        is_train=False,
        num_workers=config.num_workers,
        # image_size=config.transform.image_size,
        num_image_tokens=model.num_img_tokens,
        subset_size=subset_size,
        use_random_subset=use_random_subset,
    )


def seed_everything(seed):
    pass


def save_training_checkpoint(todo):
    pass


def parse_model_output(text):
    try:
        # Replace single quotes with double quotes
        json_text = text.replace("'", '"')
        # Remove any text before first [ and after last ]
        json_text = json_text[json_text.find("[") : json_text.rfind("]") + 1]
        # Fix missing quotes around keys
        json_text = re.sub(r"(\w+):", r'"\1":', json_text)

        # Try to parse the JSON
        objects = json.loads(json_text)

        # Validate format of each object
        for obj in objects:
            if not isinstance(obj, dict):
                log.error(f"Failed ouput parsing: {obj} is not a dictionary")
                return None
            if "class" not in obj or "bbox" not in obj:
                log.error(f"Failed ouput parsing: {obj} is missing 'class' or 'bbox' keys")
                return None
            if not isinstance(obj["bbox"], list) or not isinstance(obj["class"], list):
                log.error(f"Failed ouput parsing: {obj} 'bbox' or 'class' is not a list")
                return None
            if not all((isinstance(bbox, (int, float)) and len(bbox) == 4 for bbox in bboxes) for bboxes in obj["bbox"]):
                log.error(f"Failed ouput parsing: {obj} 'bbox' elements are not a list of 4 numbers")
                return None

        return objects
    except json.JSONDecodeError as e:
        log.error(
            f"Failed to parse model output text '{text}' (converted to json text '{json_text}') with error {e}"
        )
        return None


def parse_model_output_to_boxes(text, dataset, index, device):
    # TODO: improve code
    # parse generated ouput to extract boundingboxes
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    failed_conversion = 0

    # Get original dataset from dataset if Subset is used
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset

    text_to_parse = text[index].strip()
    predictions = parse_model_output(text_to_parse)

    if predictions is not None:
        if isinstance(predictions, list):
            predictions = predictions[0]

        classes = predictions.get("class")
        bboxes = predictions.get("bbox")

        # check if same length 
        if len(classes) != len(bboxes):
            log.error(f"Failed ouput parsing: {predictions} 'bbox' and 'class' lists have different lengths")
            failed_conversion += 1
        else:
            for class_id, bbox in zip(classes, bboxes):
                try:
                    # Convert bbox to tensor
                    if len(bbox) == 4:
                        pred_boxes.append(
                            torch.tensor(bbox, dtype=torch.float32).to(device)
                        )
                        pred_labels.append(class_id)
                        pred_scores.append(1.0)
                except (ValueError, TypeError) as e:
                    failed_conversion += 1
    else:
        failed_conversion += 1
    

    # TODO: do something with failed_conversion print or log or something
    if failed_conversion > 0:
        log.error(f"Failed to convert {failed_conversion} outputs to bounding boxes")

    bbox = torch.stack(pred_boxes) if pred_boxes else torch.zeros((0, 4))
    bbox = unnormalize_bbox(
        bbox.to(device), width=384, height=384
    )  # TODO: get height and width from config

    # TODO: add index again for batches
    # Create return tensors with proper types
    return {
        "boxes": bbox,
        "labels": (
            torch.tensor(pred_labels, dtype=torch.long, device=device)
            if pred_labels
            else torch.zeros(0, dtype=torch.long).to(device)
        ),
        "scores": (
            torch.tensor(pred_scores, dtype=torch.float32, device=device)
            if pred_scores
            else torch.zeros(0).to(device)
        ),
    }


def unnormalize_bbox(bbox: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Unnormalize bounding boxes from [0,1] to pixel coordinates efficiently.
    Args:
        bbox: Tensor of shape (N, 4) or (4,) with normalized coordinates
        width: Original image width
        height: Original image height
    Returns:
        Tensor of same shape with pixel coordinates
    """
    if bbox.numel() == 0:
        return torch.zeros((0, 4), device=bbox.device, dtype=bbox.dtype)

    if bbox.dim() == 1:
        bbox = bbox.unsqueeze(0)

    # Create scaling factor tensor
    scale = torch.tensor(
        [width, height, width, height], device=bbox.device, dtype=bbox.dtype
    )

    # Vectorized multiplication
    bbox_unnorm = bbox * scale

    return bbox_unnorm if bbox.dim() == 2 else bbox_unnorm.squeeze(0)


class JSONStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.end_sequence = self.tokenizer.encode("]]}]")  # Get token ID for closing bracket

    def __call__(self, input_ids, scores, **kwargs):
        # Stop if we find the 4 closing bracket
        return input_ids[0][-4:] == self.end_sequence
        #return input_ids[0][-1] == self.end_sequence
