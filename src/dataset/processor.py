from functools import cached_property
import json
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Union
from transformers import AutoTokenizer


class Processor:
    def __init__(self, model, train=True):
        """
        Initialize the processor with a tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer for text processing
        """
        self.model = model
        self.tokenizer = model.tokenizer
        self.image_token = "<image>"
        self.train = train

        self.answer_start_token = "<|im_start|>assistant\n"

    @cached_property
    def tokenized_start_prompt(self):
        # Tokenize the special sequence once to avoid repeated calls
        return self.tokenizer(self.answer_start_token, add_special_tokens=False)[
            "input_ids"
        ]

    def prepare_text_input(
        self,
        num_patches: int,
        instance_classes: List[str],
        instance_bboxes: List[List[float]],
        captions: Union[str, List[str]],
        num_images: int = 1,
        prompt: str = None,
    ) -> str:
        """
        Prepare the input text string by combining image tokens and metadata.

        Args:
            num_patches: Number of image patches to include
            instance_classes: List of class names for each instance
            instance_bboxes: List of bounding boxes [x1, y1, x2, y2]
            captions: Image caption(s)
            num_images: Number of images in the batch
            prompt: Prompt for the model
        """
        # Add image tokens based on number of patches
        image_tokens = (
            f"{self.image_token}"
            * num_patches
            * num_images  ## TODO removed for testing * num_patches
        ).strip()

        # Combine bboxes and classes into a list of instances
        # TODO: change bbox to special token instead of json
        instances = []
        for cls, bbox in zip(instance_classes, instance_bboxes):
            instances.append({"class": cls.tolist(), "bbox": bbox.tolist()})
        # randomize order of instances
        instances = np.random.permutation(instances).tolist()
        bbox_str = json.dumps(instances)

        # Combine all elements with proper spacing
        combined_text = (f"{image_tokens} " f"Objects: {bbox_str}").strip()

        if prompt is None:
            prompt = "What is shown in this image? Describe it in detail."
            prompt = "Detect all objects in this image! Only output list of json objects that are predicted. Example: [{'class': 'dog', 'bbox': [0.1, 0.2, 0.3, 0.4]}, {'class': 'cat', 'bbox': [0.5, 0.6, 0.7, 0.8]}]"

        system_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        # system_text = "<|im_start|>system\nYou are a helpful assistant trained to detect objects in images and output their locations in a standardized JSON format.<|im_end|>\n"
        user_text = f"<|im_start|>user\n{image_tokens}\n{prompt}<|im_end|>\n"
        assistent_text = "<|im_start|>assistant\n"

        combined_text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{image_tokens}\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        if self.train:
            combined_text += f"{bbox_str}<|im_end|>"

        # combined_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_tokens}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        # combined_text = f"{conv_structure_system}{conv_structure_user}{conv_structure_assistent}"
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
        num_tokens = self.model.image_encoder.vision_tower.vision_model.embeddings.position_embedding.weight.shape[
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

        ## Create loss_mask
        # Convert input_ids to NumPy array for faster search
        input_ids = tokenized["input_ids"]
        input_ids_np = input_ids.cpu().numpy()
        tokenized_prompt_len = len(self.tokenized_start_prompt)

        # Initialize loss masks (same shape as input_ids)
        loss_masks = np.zeros_like(input_ids_np)

        # Use NumPy's sliding window to find answer start in each sequence
        for i in range(len(input_ids_np)):  # Process each batch item independently
            seq = input_ids_np[i]

            if len(seq) >= tokenized_prompt_len:
                windows = np.lib.stride_tricks.sliding_window_view(
                    seq, tokenized_prompt_len
                )
                match_idx = np.where(
                    (windows == self.tokenized_start_prompt).all(axis=1)
                )[0]

                if match_idx.size > 0:
                    answer_start = match_idx[0] + tokenized_prompt_len
                    loss_masks[i, answer_start:] = 1  # Apply mask only to answer tokens

        # Convert loss masks back to PyTorch tensors
        loss_masks = torch.tensor(loss_masks, dtype=torch.float32)

        instance_bboxes = [sample["instance_bboxes"] for sample in batch]
        instance_classes_id = [sample["instance_classes_id"] for sample in batch]

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "images": images,
            "loss_masks": loss_masks,
            "instance_bboxes": instance_bboxes,
            "instance_classes_id": instance_classes_id,
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
