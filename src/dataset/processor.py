import json
import logging
import math
import re
from functools import cached_property
from typing import Dict, List, Tuple, Union

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from lxml import etree
from PIL import Image
from tokenizers import AddedToken
from torch.utils.data import Subset
from transformers import AutoTokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dataset.dataset import COCODataset
from utils.config import ExperimentConfig
from utils.token_utils import generate_coordinate_tokens, get_token_initializers

log = logging.getLogger(__name__)


class Processor(ProcessorMixin):
    def __init__(
        self,
        config: ExperimentConfig,
        tokenizer: PreTrainedTokenizerBase = None,
    ):
        """
        Initialize the processor with a tokenizer.

        Args:
            config: Configuration object containing parameters for the processor.
            tokenizer: Pre-trained tokenizer for text processing.
        """
        self.config = config

        self.image_size = self.config.image_encoder.image_size
        self.mean = self.config.image_encoder.mean
        self.std = self.config.image_encoder.std
        self.interpolation = self.config.image_encoder.interpolation

        self.image_token = self.config.image_token
        self.num_image_tokens = (
            self.config.image_encoder.num_image_tokens
            if not self.config.image_encoder.use_pooler_output
            else 1
        )
        self.max_length = self.config.max_tokens
        self.pad_to_multiple_of = self.config.pad_to_multiple_of

        self.tokenizer = tokenizer
        self.loaded_tokenizer_len = None

        self.answer_start_token = "<|im_start|>assistant\n"  # TODO: config
        self.use_special_coord_tokens = False

        # TODO: define transformes in config
        self.bbox_transform = A.Compose(
            [
                A.Resize(
                    self.image_size[0],
                    self.image_size[1],
                    interpolation=self.interpolation,
                ),
                A.Normalize(mean=self.mean, std=self.std),
                # A.HorizontalFlip(p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["class_labels", "class_ids"]
            ),
        )

        if "resnet50" in config.image_encoder.name:
            self.bbox_transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=1333, interpolation=self.interpolation),
                    A.SmallestMaxSize(max_size=800, interpolation=self.interpolation),
                    # A.PadIfNeeded(min_height=800, min_width=800),
                    A.Normalize(mean=self.mean, std=self.std),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc", label_fields=["class_labels", "class_ids"]
                ),
            )

    @staticmethod
    def from_config(config: ExperimentConfig, add_special_tokens: bool = True):
        """Create a new processor from a configuration."""
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        processor = Processor(config=config, tokenizer=tokenizer)
        # store length of loadedtokenizer (vocab size + special tokens)
        processor.loaded_tokenizer_len = len(tokenizer)
        if add_special_tokens:
            processor.add_special_tokens()
            processor.use_special_coord_tokens = True

        return processor

    @property
    def image_token_index(self) -> int:
        """Index of the image token in the tokenizer."""
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def add_special_tokens(self):
        """Add special tokens to the tokenizer."""
        special_tokens = [AddedToken(content=tok) for tok in self.special_tokens]
        num_added_tokens = self.tokenizer.add_tokens(special_tokens)
        assert num_added_tokens == len(
            special_tokens
        ), f"Some tokens were not added: {num_added_tokens}/{len(special_tokens)} tokens added"

    @cached_property
    def special_tokens(self) -> List[str]:
        """Special tokens for the processor."""
        special_tokens = []
        num_bins = self.config.num_coordinate_bins
        shared_coords = True  # TODO: config
        coordinate_tokens = generate_coordinate_tokens(num_bins, shared_coords)
        special_tokens.extend(coordinate_tokens)
        return special_tokens

    @property
    def special_tokens_initializer(self) -> List[List[int]]:
        """Initializers for special tokens."""
        tok = AutoTokenizer.from_pretrained(self.config.model_name)
        initializers = get_token_initializers(tok, self.special_tokens)
        return [initializers.get(token, []) for token in self.special_tokens]

    @cached_property
    def tokenized_start_prompt(self):
        # Tokenize the special sequence once to avoid repeated calls
        return self.tokenizer(self.answer_start_token, add_special_tokens=False)[
            "input_ids"
        ]

    def bin_to_coord(self, bin_idx: List[int]) -> List[float]:
        """Convert bin indix to normalized coordinate."""
        num_bins = self.config.num_coordinate_bins
        bin_idx = np.array(bin_idx)
        return bin_idx / (num_bins - 1) if num_bins > 1 else [0.0] * len(bin_idx)

    def coord_to_bin(self, coord: List[List[float]]) -> List[List[int]]:
        """Convert normalized coordinate bbox batch to bin indices."""
        # TODO: could optimize to go to closest bin, but the higher the
        # number of bins, the less important this is
        num_bins = self.config.num_coordinate_bins
        assert num_bins > 0, "Number of coordinate bins must be greater than 0"

        coord_array = np.array(coord)
        coord_bins = np.minimum(
            (coord_array * (num_bins - 1)).astype(int), num_bins - 1
        )
        return coord_bins

    ################
    # Preprocessing
    ################

    def format_bbox_to_xml(self, list_classes_str, list_bboxes):
        """Returns list in xml format:"""
        xml_str = "<annotation>"
        list_bboxes_binned = self.coord_to_bin(list_bboxes)

        for classes_str, bbox in zip(list_classes_str, list_bboxes_binned):
            x0, y0, x1, y1 = bbox
            length = len(str(self.config.num_coordinate_bins - 1))

            bbox_xml = f"<bbox><x{x0:0{length}d}/><y{y0:0{length}d}/><x{x1:0{length}d}/><y{y1:0{length}d}/></bbox>"
            xml_str += f"<object><class>{classes_str}</class>{bbox_xml}</object>"
        # print(xml_str)
        return xml_str + "</annotation>"

    def find_assistant_token_position(self, input_ids_np: np.ndarray) -> int:
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
        train: bool = True,
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
            train: Whether the processor is used for training data
        """
        # Add image tokens based on number of patches
        # if self.config.image_encoder.name == "resnet50":
        #     image_tokens = f"{self.image_token}"
        # else:
        image_tokens = (
            f"{self.image_token}" * (num_img_tokens * (num_image_patches + 1))
        ).strip()

        # Combine bboxes and classes into a list of instances
        # TODO: change bbox to special token instead of json
        # instances = []
        # for classes, bbox in zip(instance_classes_str, instance_bboxes):
        #     instances.append({"class": classes, "bbox": bbox.tolist()})
        bbox_str = self.format_bbox_to_xml(instance_classes_str, instance_bboxes)
        # randomize order of instances
        # instances = np.random.permutation(instances).tolist()
        # bbox_str = json.dumps(instances)

        if prompt is None:
            # prompt = "Detect all objects in this image! Only output list of json objects that are predicted. BBox in YOLO format. Example: [{'class': ['class_1', 'class_2'], 'bbox': [[bbox_class_1], [[bbox_class_2]]}]" #TODO: example and how string is generated is different
            # prompt = "Given the image, identify the objects present and provide their class indices and bounding boxes in the following format: [{'class': [<class_name_1>, <class_name_2>, ...], 'bbox': [[<x_min_1>, <y_min_1>, <x_max_1>, <y_max_1>], [<x_min_2>, <y_min_2>, <x_max_2>, <y_max_2>], ...]}]"
            # prompt = "Given the image, identify the objects present and provide their class indices and bounding boxes in the following format: [{'class': '<class_name_1>', 'bbox': [<x_min_1>, <y_min_1>, <x_max_1>, <y_max_1>]}, {'class': '<class_name_2>', 'bbox': [<x_min_2>, <y_min_2>, <x_max_2>, <y_max_2>]}, ...]"
            # prompt = "Detect all objects in the image and output ONLY a valid JSON array of objects. Each object must have a 'class' (string name) and 'bbox' (normalized coordinates [x_min, y_min, x_max, y_max] between 0 and 1). Format: [{'class': 'person', 'bbox': [0.2, 0.3, 0.5, 0.8]}, {'class': 'car', 'bbox': [0.6, 0.7, 0.9, 0.95]}]. Include all visible objects, even if partially visible. Output nothing but the JSON array."
            # prompt = "Output ONLY a JSON array of detected objects: [{'class': 'person', 'bbox': [x_min, y_min, x_max, y_max]}] with normalized coordinates (0-1)."
            # prompt = "Detect all objects in the image and output ONLY a valid JSON array of objects. Each object must have a 'class' (string name) and 'bbox' (list of 4 special coordinate tokens [x_min, y_min, x_max, y_max]). Format: [{'class': 'person', 'bbox': ['<coord_2>', '<coord_3>', '<coord_5>', '<coord_8>']}, {'class': 'car', 'bbox': ['<coord_6>', '<coord_7>', '<coord_9>', '<coord_9>']}]. Each <coord_X> token represents a quantized position. Include all visible objects, even if partially visible. Output nothing but the JSON array."
            # prompt = "Detect all objects in the image and output ONLY a valid JSON array of objects. Each object must have a 'class' (string name) and 'bbox' (list of 4 special coordinate tokens). Format: [{'class': 'person', 'bbox': ['<coord_2>', '<coord_3>', '<coord_5>', '<coord_8>']}, {'class': 'car', 'bbox': ['<coord_6>', '<coord_7>', '<coord_9>', '<coord_9>']}]. Each <coord_X> token represents a quantized position. Include all visible objects, even if partially visible. Output nothing but the JSON array."

            example_xml = "<annotation><object><class>car</class><bbox><x14/><y36/><x18/><y44/></bbox></object><object><class>surfboard</class><bbox><x0/><y41/><x86/><y67/></bbox></object></annotation>"
            prompt = f"Detect all objects in the image and output ONLY a valid XML of <annotation> with multiple <object>. Each <object> must have a <class> (string name) and <bbox> (containing 4 relative position tokens x min, y min, x max, y max). Format: {example_xml}. Include all visible objects, even if partially visible. Output nothing but the XML."

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

        if train:
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

    def _order_bboxes(self, boxes, classes_str, classes_id):
        if len(boxes) <= 1 or self.config.bbox_ordering == "none":
            return boxes, classes_str, classes_id

        indices = np.arange(len(boxes))

        if self.config.bbox_ordering == "random":
            indices = np.random.permutation(indices)
        elif self.config.bbox_ordering == "size_desc":
            areas = np.array([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes])
            indices = indices[np.argsort(areas)[::-1]]
        else:
            raise ValueError(f"Invalid bbox ordering: {self.config.bbox_ordering}")

        return (
            boxes[indices],
            [classes_str[i] for i in indices],
            [classes_id[i] for i in indices],
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
        transformed_classes = [None] * batch_size
        transformed_classes_id = [None] * batch_size
        text_inputs = [None] * batch_size
        bbox_str = [None] * batch_size

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

            # Apply bbox ordering
            boxes, classes_str, classes_id = self._order_bboxes(
                np.array(boxes), classes_str, classes_id
            )

            # Apply transformations
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
            norm_bboxes = [
                self.normalize_bbox(
                    bbox, transformed["image"].shape[1], transformed["image"].shape[2]
                )
                for bbox in transformed["bboxes"]
            ]
            transformed_bboxes[i] = torch.tensor(norm_bboxes, dtype=torch.float32)

            if "resnet50" not in self.config.image_encoder.name:
                # for resnet50 we need padded image size, so do it later
                text_inputs[i], bbox_str[i] = self.prepare_text_input(
                    self.num_image_tokens,
                    transformed["class_labels"],
                    norm_bboxes,
                    sample["captions"],
                    num_image_patches=0,
                    train=train,
                )

        if "resnet50" in self.config.image_encoder.name:
            # variable sized images, pad images
            # Find max dimensions in this batch
            max_h = max(img.shape[1] for img in transformed_images)
            max_w = max(img.shape[2] for img in transformed_images)
            num_img_tokens = math.ceil(max_w / 32) * math.ceil(max_h / 32)

            for i, sample in enumerate(batch):
                text_inputs[i], bbox_str[i] = self.prepare_text_input(
                    num_img_tokens=num_img_tokens,
                    instance_classes_str=transformed_classes[i],
                    instance_bboxes=transformed_bboxes[i],
                    captions=sample["captions"],
                    num_image_patches=0,
                    train=train,
                )

            # log.info(f"padded image_size: ({max_h}, {max_w})")

            # Create a zero tensor of the max size
            channels = transformed_images[0].shape[0]
            images = torch.zeros(
                batch_size, channels, max_h, max_w, dtype=transformed_images[0].dtype
            )

            # Copy each image into the padded tensor
            for i, img in enumerate(transformed_images):
                c, h, w = img.shape
                images[i, :, :h, :w] = img

            # Store the original image sizes for proper feature map calculation
            image_sizes = [(img.shape[1], img.shape[2]) for img in transformed_images]
        else:
            # fixed_size images, stack tensors
            images = torch.stack(transformed_images)
            image_sizes = None

        if train:  # fast_att only allow left padding
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
        labels[tokenized["attention_mask"] == 0] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask padding tokens
        labels[labels == self.image_token_index] = -100  # Mask image tokens
        labels[loss_masks == 0] = -100  # Mask everything except the answer tokens

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "images": images,
            "labels": labels,
            # for evaluation purposes
            # TODO: put here "targets": self.postprocess_target_batch(...)
            "instance_bboxes": transformed_bboxes,
            "instance_classes_id": transformed_classes_id,
            "bbox_str": bbox_str,
            "image_sizes": image_sizes,
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
            width: Original image width
            height: Original image height
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
                "boxes": self._unnormalize_bbox(boxes.to(device), size=self.image_size),
                "labels": labels.to(device),
            }
            for boxes, labels in zip(
                batch["instance_bboxes"], batch["instance_classes_id"]
            )
        ]

    def postprocess_xml_batch(
        self,
        sequence: torch.LongTensor,
        dataset: Union[COCODataset, Subset],
        device: str,
    ):
        assert sequence is not None, "No batch provided"

        generated_text = self.tokenizer.batch_decode(sequence, skip_special_tokens=True)

        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        category_dict = dataset.cat_name_to_id

        results = []
        failed = 0  # TODO: return failed

        for i, text in enumerate(generated_text):
            try:
                results.append(
                    self._postprocess_xml(
                        text=text, cat_name_to_id=category_dict, device=device
                    )
                )
            except Exception as e:
                log.warning(f"Error processing item {i} in batch: {e}")
                failed += 1
                results.append(
                    {
                        "boxes": torch.zeros((0, 4), device=device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                        "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                    }
                )

        return generated_text, results

    def _postprocess_xml(self, text: str, cat_name_to_id: Dict[str, int], device: str):
        assert text is not None, "No text provided"

        start_idx = text.find("<annotation>")
        end_idx = text.rfind("</annotation>") + len("</annotation>")
        if start_idx == -1 or end_idx == 0:
            raise ValueError("Invalid XML format")

        xml_text = text[start_idx:end_idx]
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(xml_text, parser=parser)

        objects = root.findall("object")
        if not objects:
            raise ValueError("No objects found in XML")

        num_objects = len(objects)

        text_bbox = torch.zeros((num_objects, 4), dtype=torch.float32, device=device)
        text_labels = torch.zeros(num_objects, dtype=torch.int64, device=device)
        text_scores = torch.ones(num_objects, dtype=torch.float32, device=device)

        valid_count = 0

        for i, obj in enumerate(objects):
            class_name = obj.find("class").text
            bbox = obj.find("bbox")

            # Convert bbox to tensor, values in attrib
            if bbox is not None:
                childs = bbox.getchildren()
                # <x0/><y20/><x30/><y32/> get bins from token name
                if len(childs) == 4:
                    x0 = int(childs[0].tag[1:])
                    y0 = int(childs[1].tag[1:])
                    x1 = int(childs[2].tag[1:])
                    y1 = int(childs[3].tag[1:])

                    text_bbox[i] = torch.tensor(
                        self.bin_to_coord([x0, y0, x1, y1]),
                        dtype=torch.float32,
                        device=device,
                    )
                    text_labels[i] = cat_name_to_id.get(class_name, -1)
                    valid_count += 1

        if valid_count < num_objects:
            text_bbox = text_bbox[:valid_count]
            text_labels = text_labels[:valid_count]
            text_scores = text_scores[:valid_count]

        text_bbox = self._unnormalize_bbox(text_bbox, size=self.image_size)

        return {
            "boxes": text_bbox,
            "labels": text_labels,
            "scores": text_scores,
        }

    ################
    # JSON Postprocessing - not used
    ################
    def postprocess_json_batch(
        self,
        sequences: torch.LongTensor,
        dataset: Union[COCODataset, Subset],
        device: str,
    ):
        # TODO: move postprocessing here
        assert sequences is not None, "No output sequences provided"

        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        category_dict = dataset.cat_name_to_id

        generated_texts = self.tokenizer.batch_decode(
            sequences, skip_special_tokens=True
        )

        return generated_texts, [
            self._postprocess_json(
                text=text, cat_name_to_id=category_dict, device=device
            )
            for text in generated_texts
        ]

    def _postprocess_json(self, text: str, cat_name_to_id: Dict[str, int], device: str):
        assert text is not None, "No text provided"

        predictions = self._parse_model_output(text)

        if not predictions or len(predictions) == 0:
            return {
                "boxes": torch.zeros((0, 4), device=device),
                "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                "scores": torch.zeros((0,), dtype=torch.float32, device=device),
            }

        num_predictions = len(predictions)
        text_bbox = torch.zeros(
            (num_predictions, 4), dtype=torch.float32, device=device
        )
        text_labels = torch.zeros(num_predictions, dtype=torch.int64, device=device)
        text_scores = torch.ones(num_predictions, dtype=torch.float32, device=device)

        for i, obj in enumerate(predictions):
            class_name = obj.get("class", "")
            bbox = obj.get("bbox", [])

            # Convert bbox to tensor
            if len(bbox) == 4:
                text_bbox[i] = torch.tensor(bbox, dtype=torch.float32, device=device)
                text_labels[i] = cat_name_to_id.get(class_name, -1)

        text_bbox = self._unnormalize_bbox(text_bbox, size=self.image_size)

        return {
            "boxes": text_bbox,
            "labels": text_labels,
            "scores": text_scores,
        }

    def _parse_model_output(self, text: str) -> List[Dict]:
        try:
            # Parse JSON text
            # replace single quotes with double quotes, remove leading/training text, fix missing quotes

            start_idx = text.find("[")
            end_idx = text.rfind("]") + 1
            if start_idx == -1 or end_idx == 0:
                return []

            json_text = text[start_idx:end_idx]
            json_text = json_text.replace("'", '"')
            # quotes around keys
            json_text = re.sub(r"([{,]\s*)(\w+)(:)", r'\1"\2"\3', json_text)

            # quotes around unquoted coorindate tokens
            pattern = r'(?<![\'"])(<coord_\d+>)(?![\'"])'
            json_text = re.sub(pattern, r'"\1"', json_text)

            objects = json.loads(json_text)

            # Precompile token pattern for faster processing
            token_pattern = re.compile(r"<coord_(\d+)>")

            valid_objects = []

            # Validate output format, throw error when invalid
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                if "class" not in obj or "bbox" not in obj:
                    continue
                if not isinstance(obj["bbox"], list) or not isinstance(
                    obj["class"], str
                ):
                    continue
                if len(obj["bbox"]) != 4:
                    continue

                bbox = obj["bbox"]
                is_valid = True

                for i, coord in enumerate(bbox):
                    if isinstance(coord, str) and coord.startswith("<coord_"):
                        match = token_pattern.match(coord)
                        if match:
                            bbox[i] = self.bin_to_coord(int(match.group(1)))
                        else:
                            is_valid = False
                            break
                    elif not isinstance(coord, (int, float)):
                        is_valid = False
                        break

                if not is_valid:
                    continue

                # Ensure bbox values are all floats
                if not all(isinstance(bbox_val, (int, float)) for bbox_val in bbox):
                    continue

                # Ensure bbox values are within [0, 1]
                for i in range(4):
                    bbox[i] = min(max(bbox[i], 0.0), 1.0)

                valid_objects.append(obj)

            return valid_objects

        except json.JSONDecodeError as e:
            log.debug(f"Failed to parse model output text '{text}' with error {e}")
            return []
