import json
from functools import cached_property
from typing import Dict, List, Tuple, Union

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from transformers import PreTrainedTokenizerFast as Tokenizer


class Processor:
    def __init__(
        self,
        config,
        tokenizer: Tokenizer,
        img_size: Tuple[int, int],
        num_img_tokens: int,
        train: bool = True,
    ):
        """
        Initialize the processor with a tokenizer.

        Args:
            tokenizer: PreTrainedTokenizer
            img_size: Tuple of image dimensions (height, width)
            num_img_tokens: Number of image tokens to include
            train: Whether the processor is used in training

        """
        self.config = config
        self.tokenizer = tokenizer
        self.num_img_tokens = num_img_tokens
        self.train = train
        self.img_size = img_size
        self.max_length = self.config.max_tokens  # None, 512

        self.image_token = "<image>"  # TODO: config
        self.answer_start_token = "<|im_start|>assistant\n"

        # TODO: define transformes in config
        self.bbox_transform = A.Compose(
            [
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                # A.HorizontalFlip(p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels", "class_ids"]
            ),
        )

    @cached_property
    def tokenized_start_prompt(self):
        # Tokenize the special sequence once to avoid repeated calls
        return self.tokenizer(self.answer_start_token, add_special_tokens=False)[
            "input_ids"
        ]

    def prepare_text_input(
        self,
        num_img_tokens: int,
        instance_classes_str: List[str],
        instance_bboxes: List[List[float]],
        captions: Union[str, List[str]],
        num_image_patches: int = 0,
        prompt: str = None,
    ) -> str:
        """
        Prepare the input text string by combining image tokens and metadata.

        Args:
            num_img_tokens: Number of image tokens to include
            instance_classes_str: List of class names for each instance
            instance_bboxes: List of bounding boxes [x1, y1, x2, y2]
            captions: Image caption(s)
            num_image_patches: Number of image patches in the batch
            prompt: Prompt for the model
        """
        # Add image tokens based on number of patches
        image_tokens = (
            f"{self.image_token}" * (num_img_tokens * (num_image_patches + 1))
        ).strip()

        # Combine bboxes and classes into a list of instances
        # TODO: change bbox to special token instead of json
        instances = []
        for classes, bbox in zip(instance_classes_str, instance_bboxes):
            instances.append({"class": classes, "bbox": bbox.tolist()})
        # randomize order of instances
        instances = np.random.permutation(instances).tolist()
        bbox_str = json.dumps(instances)

        if prompt is None:
            # prompt = "Detect all objects in this image! Only output list of json objects that are predicted. Example: [{'class': 'dog', 'bbox': [0.1, 0.2, 0.3, 0.4]}, {'class': 'cat', 'bbox': [0.5, 0.6, 0.7, 0.8]}]" #TODO: example and how string is generated is different
            # prompt = "Detect all objects in this image! Only output list of json objects that are predicted. BBox in YOLO format. Example: [{'class': ['class_1', 'class_2'], 'bbox': [[bbox_class_1], [[bbox_class_2]]}]" #TODO: example and how string is generated is different
            prompt = "Given the image, identify the objects present and provide their class indices and bounding boxes in the following format: [{'class': [<class_name_1>, <class_name_2>, ...], 'bbox': [[<x_min_1>, <y_min_1>, <x_max_1>, <y_max_1>], [<x_min_2>, <y_min_2>, <x_max_2>, <y_max_2>], ...]}]"

        # system_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        # system_text = "<|im_start|>system\nYou are a helpful assistant trained to detect objects in images and output their locations in a standardized JSON format.<|im_end|>\n"
        # user_text = f"<|im_start|>user\n{image_tokens}\n{prompt}<|im_end|>\n"
        # assistent_text = "<|im_start|>assistant\n"

        # Combine all elements with chat structure
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

    # TODO: currently duplicte as in dataset, cause it it still used outside
    def normalize_bbox(self, bbox, image_width, image_height):
        x1, y1, x2, y2 = bbox
        return (
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height,
        )

    def denormalize_bbox(self, bbox, image_width, image_height):
        x1, y1, x2, y2 = bbox
        return (
            x1 * image_width,
            y1 * image_height,
            x2 * image_width,
            y2 * image_height,
        )

    def process_batch(
        self, batch: List[Dict[str, Union[Image.Image, List[str], List[List[float]]]]]
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
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
        transformed_images = []
        transformed_bboxes = []
        transformed_classes = []
        transformed_classes_id = []

        for sample in batch:
            # # Check if x_max > x_min and y_max > y_min
            # bboxes = sample["instance_bboxes"]
            # valid_order = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])

            # # remove all invalid bboxes using valid order
            # s

            # if not np.all(valid_order):
            #     invalid_idx = np.where(~valid_order)[0][0]
            #     invalid_bbox = bboxes[invalid_idx]

            # remove all invalid bboxes and their corresponding classes
            valid_indices = [
                i
                for i, bbox in enumerate(sample["instance_bboxes"])
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]
            ]
            sample["instance_bboxes"] = [
                sample["instance_bboxes"][i] for i in valid_indices
            ]
            sample["instance_classes_str"] = [
                sample["instance_classes_str"][i] for i in valid_indices
            ]
            sample["instance_classes_id"] = [
                sample["instance_classes_id"][i] for i in valid_indices
            ]

            transformed = self.bbox_transform(
                image=np.array(sample["image"]),
                bboxes=sample["instance_bboxes"],
                class_labels=sample["instance_classes_str"],
                class_ids=sample["instance_classes_id"],
            )
            transformed_images.append(transformed["image"])
            transformed_classes.append(transformed["class_labels"])
            transformed_classes_id.append(
                torch.tensor(transformed["class_ids"], dtype=torch.int64)
            )

            # Normalize values of bboxes
            transformed_img = transformed["image"]
            norm_bboxes = [
                self.normalize_bbox(
                    bbox, transformed_img.shape[1], transformed_img.shape[2]
                )
                for bbox in transformed["bboxes"]
            ]
            transformed_bboxes.append(torch.tensor(norm_bboxes, dtype=torch.float16))

        images = torch.stack(transformed_images)

        # Prepare text inputs
        text_inputs = [
            self.prepare_text_input(
                self.num_img_tokens,
                transformed_classes,
                transformed_bboxes,
                sample["captions"],
                num_image_patches=0,
            )
            for sample in batch
        ]

        # Tokenize texts
        tokenized = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
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

        ## Create Labels
        # Prepare lables
        labels = tokenized["input_ids"].clone()
        # TODO: make image_token_id as attribute
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        labels[labels == image_token_id] = -100  # Mask image tokens
        labels[loss_masks == 0] = -100  # Mask everything except the answer tokens

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "images": images,
            "labels": labels,
            "instance_bboxes": transformed_bboxes,
            "instance_classes_id": transformed_classes_id,
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
