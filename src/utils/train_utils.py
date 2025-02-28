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
    max_tokens: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    batch_size: int = MISSING
    total_batch_size: int = MISSING
    
    epochs: int = MISSING
    # max_steps: Optional[int] = None
    # max_epochs: Optional[int] = None
    lr: float = MISSING
    # min_lr: float = MISSING
    # warmup_lr: Optional[float] = MISSING
    warmup_ratio: float = MISSING
    weight_decay: Optional[float] = MISSING
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

    temperature: float = MISSING
    use_amp: bool = MISSING # use automatic mixed precision


def build_train_dataloader(config: ExperimentConfig, model, subset_size=None):
    return build_dataloader(
        config=config,
        dataset_config=config.train_dataset,
        batch_size=config.batch_size,
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
        # TODO: improve raise exceptions
        for obj in objects:
            if not isinstance(obj, dict):
                # log.error(f"Failed ouput parsing: {obj} is not a dictionary")
                return None
            if "class" not in obj or "bbox" not in obj:
                # log.error(
                #     f"Failed ouput parsing: {obj} is missing 'class' or 'bbox' keys"
                # )
                return None
            if not isinstance(obj["bbox"], list) or not isinstance(obj["class"], str):
                # log.error(
                #     f"Failed ouput parsing: {obj} 'bbox' is not a list or 'class' is not a str"
                # )
                return None
            if (
                not all(isinstance(bbox_val, (int, float)) for bbox_val in obj["bbox"])
                or len(obj["bbox"]) != 4
            ):
                # log.error(
                #     f"Failed ouput parsing: {obj} 'bbox' is not a list of 4 numbers"
                # )
                return None

        return objects
    except json.JSONDecodeError as e:
        # log.error(
        #     f"Failed to parse model output text '{text}' (converted to json text '{json_text}') with error {e}"
        # )
        return None


def parse_model_output_to_boxes(batch_text, dataset, device):
    # TODO: improve code
    # parse generated ouput to extract boundingboxes
    batch_return = []
    failed_conversion = 0

    # Get original dataset from dataset if Subset is used
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset

    for text in batch_text:
        text_to_parse = text.strip()
        predictions = parse_model_output(text_to_parse)

        text_boxes = []
        text_labels = []
        text_scores = []

        if predictions is not None:

            for obj in predictions:
                class_name = obj.get("class")
                bbox = obj.get("bbox")

                # Convert bbox to tensor
                if len(bbox) == 4:
                    text_boxes.append(
                        torch.tensor(bbox, dtype=torch.float32).to(device)
                    )
                    # class_name to id
                    class_id = dataset.cat_name_to_id.get(class_name, -1)
                    text_labels.append(class_id)
                    text_scores.append(1.0)

        else:
            failed_conversion += 1

        # Create return tensors with proper types
        text_bbox = torch.stack(text_boxes) if text_boxes else torch.zeros((0, 4))
        # TODO: get height and width from config
        text_bbox = unnormalize_bbox(text_bbox, width=384, height=384)
        text_labels = (
            torch.tensor(text_labels, dtype=torch.long, device=device)
            if text_labels
            else torch.zeros(0, dtype=torch.long).to(device)
        )
        text_scores = (
            torch.tensor(text_scores, dtype=torch.float32, device=device)
            if text_scores
            else torch.zeros(0).to(device)
        )

        batch_return.append(
            {
                "boxes": text_bbox,
                "labels": text_labels,
                "scores": text_scores,
            }
        )

    # TODO: do something with failed_conversion print or log or something
    if failed_conversion > 0:
        log.error(f"Failed to convert {failed_conversion} outputs to bounding boxes")

    return batch_return


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
        self.end_sequence = self.tokenizer.encode(
            "<|im_end|>" # "]}]<|im_end|>"
        )  # Get token ID for closing bracket
        self.length = len(self.end_sequence)

    def __call__(self, input_ids, scores, **kwargs):
        # Stop if we find the end sequence
        # print(input_ids[0][-self.length :])
        # print(self.end_sequence)
        return input_ids[0][-self.length :] == self.end_sequence
        # return input_ids[0][-1] == self.end_sequence
