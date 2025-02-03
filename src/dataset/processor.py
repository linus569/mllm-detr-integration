import json
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Union
from transformers import AutoTokenizer


class Processor:
    def __init__(self, model):
        """
        Initialize the processor with a tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer for text processing
        """
        self.model = model
        self.tokenizer = model.tokenizer
        self.image_token = "<image>"

    def prepare_text_input(
        self,
        num_patches: int,
        instance_classes: List[str],
        instance_bboxes: List[List[float]],
        captions: Union[str, List[str]],
    ) -> str:
        """
        Prepare the input text string by combining image tokens and metadata.

        Args:
            num_patches: Number of image patches to include
            instance_classes: List of class names for each instance
            instance_bboxes: List of bounding boxes [x1, y1, x2, y2]
            captions: Image caption(s)
        """
        # Add image tokens based on number of patches
        image_tokens = f" {self.image_token} " * num_patches

        # Combine bboxes and classes into a list of instances
        # TODO: change bbox to special token instead of json
        instances = []
        for cls, bbox in zip(instance_classes, instance_bboxes):
            instances.append({"class": cls.tolist(), "bbox": bbox.tolist()})
        bbox_str = json.dumps(instances)

        # Handle both single string and list of captions
        if isinstance(captions, list):
            caption_text = " ".join(captions)
        else:
            caption_text = captions

        # Combine all elements with proper spacing
        combined_text = (
            f"{image_tokens} " f"Objects: {bbox_str} " f"Caption: {caption_text}"
        ).strip()

        return combined_text

    def process_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of dictionaries to create model inputs.

        Args:
            batch: List of dictionaries containing:
                - 'image': PIL.Image
                - 'instance_classes': List[str]
                - 'instance_bboxes': List[List[float]]
                - 'captions': str or List[str]

        Returns:
            Dictionary with:
                - 'input_ids': padded token sequences
                - 'attention_mask': attention mask
                - 'pixel_values': stacked image tensors
        """
        # Get number of tokens of image encoder
        num_tokens = self.model.vision_tower.vision_model.embeddings.position_embedding.weight.shape[
            0
        ]

        # Prepare text inputs
        text_inputs = [
            self.prepare_text_input(
                num_tokens,
                sample["instance_classes"],
                sample["instance_bboxes"],
                sample["captions"],
            )
            for sample in batch
        ]

        # Tokenize texts
        tokenized = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            # max_length=self.max_length,
            return_tensors="pt",
        )

        # Stack images
        images = torch.stack(
            [torch.from_numpy(np.array(sample["image"])) for sample in batch]
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "images": images,
        }

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function to be used with PyTorch DataLoader.

        Args:
            samples: List of batch dictionaries

        Returns:
            Processed batch dictionary
        """
        return self.process_batch(batch)
