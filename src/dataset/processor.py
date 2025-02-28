import json
from functools import cached_property
from typing import Dict, List, Tuple, Union

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image


class Processor:
    def __init__(
        self,
        config,
        img_size: Tuple[int, int],
        num_img_tokens: int,
        train: bool = True,
    ):
        """
        Initialize the processor with a tokenizer.

        Args:
            img_size: Tuple of image dimensions (height, width)
            num_img_tokens: Number of image tokens to include
            train: Whether the processor is used in training

        """
        self.config = config
        self.num_img_tokens = num_img_tokens
        self.train = train
        self.img_size = img_size
        self.max_length = self.config.max_tokens
        self.pad_to_multiple_of = self.config.pad_to_multiple_of

        self._tokenizer = None

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

    @property
    def tokenizer(self):
        """Lazy loading of tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return self._tokenizer

    @cached_property
    def tokenized_start_prompt(self):
        # Tokenize the special sequence once to avoid repeated calls
        return self.tokenizer(self.answer_start_token, add_special_tokens=False)[
            "input_ids"
        ]

    @cached_property
    def image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def bbox_string(self, list_classes_str, list_bboxes):
        """Returns list in this format: [{'class': 'person', 'bbox': [0.2, 0.3, 0.5, 0.8]}, {'class': 'car', 'bbox': [0.6, 0.7, 0.9, 0.95]}]"""
        return [
            {"class": class_str, "bbox": bbox}
            for class_str, bbox in zip(list_classes_str, list_bboxes)
        ]

    def find_assistant_token_position(self, input_ids_np: torch.Tensor) -> int:
        """Optimized search for assistant token in input_ids."""
        batch_size, seq_len = input_ids_np.shape
        token_len = len(self.tokenized_start_prompt)

        # Create result array
        positions = np.zeros(batch_size, dtype=np.int32)

        # Fast path for common case
        start_token = self.tokenized_start_prompt[0]
        for i in range(batch_size):
            potential_start = np.where(input_ids_np[i] == start_token)[0]

            for pos in potential_start:
                if pos + token_len <= seq_len:
                    if np.array_equal(
                        input_ids_np[i, pos : pos + token_len],
                        self.tokenized_start_prompt,
                    ):
                        positions[i] = pos + token_len
                        break
        return positions

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
        # instances = []
        # for classes, bbox in zip(instance_classes_str, instance_bboxes):
        #     instances.append({"class": classes, "bbox": bbox.tolist()})
        instances = self.bbox_string(instance_classes_str, instance_bboxes)
        # randomize order of instances
        instances = np.random.permutation(instances).tolist()
        bbox_str = json.dumps(instances)

        if prompt is None:
            # prompt = "Detect all objects in this image! Only output list of json objects that are predicted. Example: [{'class': 'dog', 'bbox': [0.1, 0.2, 0.3, 0.4]}, {'class': 'cat', 'bbox': [0.5, 0.6, 0.7, 0.8]}]" #TODO: example and how string is generated is different
            # prompt = "Detect all objects in this image! Only output list of json objects that are predicted. BBox in YOLO format. Example: [{'class': ['class_1', 'class_2'], 'bbox': [[bbox_class_1], [[bbox_class_2]]}]" #TODO: example and how string is generated is different
            # prompt = "Given the image, identify the objects present and provide their class indices and bounding boxes in the following format: [{'class': [<class_name_1>, <class_name_2>, ...], 'bbox': [[<x_min_1>, <y_min_1>, <x_max_1>, <y_max_1>], [<x_min_2>, <y_min_2>, <x_max_2>, <y_max_2>], ...]}]"
            prompt = "Given the image, identify the objects present and provide their class indices and bounding boxes in the following format: [{'class': '<class_name_1>', 'bbox': [<x_min_1>, <y_min_1>, <x_max_1>, <y_max_1>]}, {'class': '<class_name_2>', 'bbox': [<x_min_2>, <y_min_2>, <x_max_2>, <y_max_2>]}, ...]"
            prompt = "Detect all objects in the image and output ONLY a valid JSON array of objects. Each object must have a 'class' (string name) and 'bbox' (normalized coordinates [x_min, y_min, x_max, y_max] between 0 and 1). Format: [{'class': 'person', 'bbox': [0.2, 0.3, 0.5, 0.8]}, {'class': 'car', 'bbox': [0.6, 0.7, 0.9, 0.95]}]. Include all visible objects, even if partially visible. Output nothing but the JSON array."
            # prompt = "Output ONLY a JSON array of detected objects: [{'class': 'person', 'bbox': [x_min, y_min, x_max, y_max]}] with normalized coordinates (0-1)."
        
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
        return combined_text, bbox_str

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
        self, batch: List[Dict]
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
        # Pre-allocate lists
        batch_size = len(batch)
        transformed_images = [None] * batch_size
        transformed_bboxes = [None] * batch_size
        transformed_classes = [None] * batch_size
        transformed_classes_id = [None] * batch_size
        text_inputs = [None] * batch_size
        bbox_str = [None] * batch_size

        for i, sample in enumerate(batch):
            # # Check if x_max > x_min and y_max > y_min
            # bboxes = sample["instance_bboxes"]
            # valid_order = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])

            # # remove all invalid bboxes using valid order
            # s

            # if not np.all(valid_order):
            #     invalid_idx = np.where(~valid_order)[0][0]
            #     invalid_bbox = bboxes[invalid_idx]

            # remove all invalid bboxes and their corresponding classes
            boxes = np.array(sample["instance_bboxes"])
            # Check if there are any boxes
            if len(boxes) == 0:
                boxes = np.zeros((0, 4))  # Empty array with correct shape

            valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

            if not np.all(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                boxes = boxes[valid_indices]
                classes_str = np.array(sample["instance_classes_str"])[
                    valid_indices
                ].tolist()
                classes_id = np.array(sample["instance_classes_id"])[
                    valid_indices
                ].tolist()
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
            transformed_classes[i] = transformed["class_labels"]
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

            text_inputs[i], bbox_str[i] = self.prepare_text_input(
                self.num_img_tokens,
                transformed["class_labels"],
                norm_bboxes,
                sample["captions"],
                num_image_patches=0,
            )

        images = torch.stack(transformed_images)

        if self.train:
            self.tokenizer.padding_side = "right"
        else:
            self.tokenizer.padding_side = "left"

        # Tokenize texts
        tokenized = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        ## Create loss_mask
        # Convert input_ids to NumPy array for faster search
        input_ids_np = tokenized["input_ids"].cpu().numpy()

        # with np.printoptions(threshold=np.inf):
        #     print(input_ids_np)
        #     encoded = self.tokenizer.decode(input_ids_np[0])
        #     print(encoded)

        # Initialize loss masks (same shape as input_ids)
        loss_masks = np.zeros_like(input_ids_np)

        positions = self.find_assistant_token_position(input_ids_np)
        for i, pos in enumerate(positions):
            loss_masks[i, pos:] = 1

        # Convert loss masks back to PyTorch tensors
        loss_masks = torch.tensor(loss_masks, dtype=torch.bool)

        ## Create Labels
        # Prepare lables
        labels = tokenized["input_ids"].clone()
        # TODO: make image_token_id as attribute

        labels[labels == self.image_token_id] = -100  # Mask image tokens
        labels[loss_masks == 0] = -100  # Mask everything except the answer tokens

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "images": images,
            "labels": labels,
            "instance_bboxes": transformed_bboxes,
            "instance_classes_id": transformed_classes_id,
            "bbox_str": bbox_str,
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
