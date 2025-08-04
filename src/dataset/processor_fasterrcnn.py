import logging
from typing import Dict, List, Tuple, Union

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from transformers.processing_utils import ProcessorMixin

from utils.config import ExperimentConfig

logger = logging.getLogger(__name__)


class FastRCNNProcessor(ProcessorMixin):
    def __init__(
        self,
        config: ExperimentConfig,
    ):
        """
        Initialize the processor with a tokenizer.

        Args:
            config: Configuration object containing parameters for the processor.
        """
        self.config = config
        self.image_size = self.config.image_encoder.image_size
        self.tokenizer = None  # placeholder, otherwise it will throw an error

        # TODO: define transformes in config
        self.bbox_transform = A.Compose(
            [
                A.ToFloat(max_value=255.0),
                # norm and resizing is done in the model
                # A.LongestMaxSize(max_size=800),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0), #mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels", "class_ids"]
            ),
        )

    @staticmethod
    def from_config(config: ExperimentConfig):
        """Create a new processor from a configuration."""
        processor = FastRCNNProcessor(config=config)
        return processor

    ################
    # Preprocessing
    ################

    # TODO: currently duplicte as in dataset, cause it it still used outside
    def normalize_bbox(self, bbox, image_width, image_height):
        x1, y1, x2, y2 = bbox
        return (
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height,
        )

    def preprocess_img_text_batch(
        self, batch: List[Dict], train: bool = True
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Process a batch of dictionaries to create model inputs.

        Args:
            batch: List of dictionaries containing:
                - 'image': PIL.Image
                - 'instance_classes': List[str]
                - 'instance_bboxes': List[List[float]]
                - 'captions': str or List[str]
            train: Whether the processor is used for training data

        Returns:
            Dictionary with:
                - 'input_ids': padded token sequences
                - 'attention_mask': attention mask
                - 'pixel_values': stacked image tensors
        """
        # Pre-allocate lists
        batch_size = len(batch)
        transformed_images = [None] * batch_size
        transformed_bboxes = [None] * batch_size
        transformed_classes_id = [None] * batch_size

        for i, sample in enumerate(batch):
            # Check if there are any boxes and if they are valid, if not remove them
            boxes = np.array(sample["instance_bboxes"])
            if len(boxes) == 0:
                boxes = np.zeros((0, 4))

            valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

            if not np.all(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                boxes = boxes[valid_indices]
                classes_str = np.array(sample["instance_classes_str"])[valid_indices]
                classes_str = classes_str.tolist()
                classes_id = np.array(sample["instance_classes_id"])[valid_indices]
                classes_id = classes_id.tolist()
            else:
                classes_str = sample["instance_classes_str"]
                classes_id = sample["instance_classes_id"]

            transformed = self.bbox_transform(
                image=np.array(sample["image"]),
                bboxes=boxes,
                class_labels=classes_str,
                class_ids=classes_id,
            )
            transformed_images[i] = transformed["image"]
            transformed_classes_id[i] = torch.tensor(
                transformed["class_ids"], dtype=torch.int64
            )

            # Normalize values of bboxes
            transformed_img = transformed["image"]
            norm_bboxes = [
                self.normalize_bbox(
                    bbox, transformed_img.shape[1], transformed_img.shape[2]
                )
                for bbox in transformed["bboxes"]
            ]
            transformed_bboxes[i] = torch.tensor(norm_bboxes, dtype=torch.float32)

        # images = torch.stack(transformed_images)
        images = transformed_images
        return {
            "input_ids": torch.zeros((batch_size, 1), dtype=torch.int64),
            "attention_mask": torch.zeros((batch_size, 1), dtype=torch.int64),
            "images": images,
            "labels": torch.zeros((batch_size, 1), dtype=torch.int64),
            # for evaluation purposes
            # TODO: put here "targets": self.postprocess_target_batch(...)
            "instance_bboxes": transformed_bboxes,
            "instance_classes_id": transformed_classes_id,
            "bbox_str": "",
        }

    ################
    # Postprocessing
    ################

    def _unnormalize_bbox(
        self, bbox: torch.Tensor, size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Unnormalize bounding boxes from [0,1] to pixel coordinates efficiently.
        Args:
            bbox: Tensor of shape (N, 4) or (4,) with normalized coordinates
            size: Tuple of image dimensions (height, width)
        Returns:
            Tensor of same shape with pixel coordinates
        """
        if bbox.numel() == 0:
            return torch.zeros((0, 4), device=bbox.device, dtype=bbox.dtype)
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)
        width, height = size

        # Create scaling factor tensor
        scale = torch.tensor(
            [width, height, width, height], device=bbox.device, dtype=bbox.dtype
        )

        # Vectorized multiplication
        bbox_unnorm = bbox * scale
        return bbox_unnorm if bbox.dim() == 2 else bbox_unnorm.squeeze(0)

    def postprocess_target_batch(self, batch: Dict[str, torch.Tensor], device: str):
        assert batch is not None, "No batch provided"
        assert "instance_bboxes" in batch, "No instance bboxes provided"
        assert "instance_classes_id" in batch, "No instance classes provided"

        return [
            {
                "boxes": self._unnormalize_bbox(boxes.to(device), size=img.shape[-2:]),
                "labels": labels.to(device),
            }
            for boxes, labels, img in zip(
                batch["instance_bboxes"], batch["instance_classes_id"], batch["images"]
            )
        ]
