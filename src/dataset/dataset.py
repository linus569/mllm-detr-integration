import random
import albumentations as A
import numpy as np
import torch

from albumentations.pytorch import ToTensorV2
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, Subset

from dataset.processor import Processor

from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)

from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from functools import partial


class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_ids = sorted([cat["id"] for cat in self.categories])
        self.cat_name_to_id = {cat["name"]: cat["id"] for cat in self.categories}
        self.index_to_cat_name = {cat["id"]: cat["name"] for cat in self.categories}

    def normalize_bbox(self, bbox, image_width, image_height):
        x1, y1, x2, y2 = bbox
        return [
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height,
        ]
    
    def denormalize_bbox(self, bbox, image_width, image_height):
        x1, y1, x2, y2 = bbox
        return [
            x1 * image_width,
            y1 * image_height,
            x2 * image_width,
            y2 * image_height,
        ]


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.image_dir}/{image_info['file_name']}"
        image = Image.open(image_path).convert("RGB")

        instance_classes = [ann["category_id"] for ann in anns]
        instance_bboxes = [ann["bbox"] for ann in anns]
        captions = [ann["caption"] for ann in anns if "caption" in ann]

        # convert bboxes from COCO format (x,y,w,h) to (x_min, y_min, x_max, y_max)
        instance_bboxes = [[x, y, x + w, y + h] for x, y, w, h in instance_bboxes]

        # apply transformations
        if self.transform:
            # Normal transforms
            # image_mean = (0.5, 0.5, 0.5)
            # image_std = (0.5, 0.5, 0.5)
            # size = (384, 384)
            # resample = PILImageResampling.BICUBIC
            # rescale_factor = 1 / 255
            # data_format = ChannelDimension.LAST  # .FIRST

            # transforms = [
            #     convert_to_rgb,
            #     to_numpy_array,
            #     # partial(resize, size=size, resample=resample, data_format=data_format),
            #     partial(rescale, scale=rescale_factor, data_format=data_format),
            #     partial(
            #         normalize, mean=image_mean, std=image_std, data_format=data_format
            #     ),
            #     # partial(to_channel_dimension_format, channel_dim=data_format, input_channel_dim=data_format),
            # ]

            # Transform image
            # for transform in transforms:
            #     image = transform(image)

            # Albumentation Transform image with bboxes
            transformed = self.transform(
                image=np.array(image),
                bboxes=instance_bboxes,
                class_labels=instance_classes,
            )
            image = transformed["image"]
            instance_bboxes = transformed["bboxes"]
            instance_classes = transformed["class_labels"]
            instance_classes_id = [
                self.cat_name_to_id[self.index_to_cat_name[cat_id]]
                for cat_id in instance_classes
            ]
            # Normalize values of bboxes
            instance_bboxes = [self.normalize_bbox(bbox, image.shape[1], image.shape[2]) for bbox in instance_bboxes]

        return {
            "image": image,
            "instance_classes": torch.tensor(instance_classes, dtype=torch.long),
            "instance_bboxes": torch.tensor(instance_bboxes, dtype=torch.float16),
            "instance_classes_id": torch.tensor(instance_classes_id, dtype=torch.long),
            "captions": captions,
        }


TRANSFORMATIONS = A.Compose(
    [
        #A.ToRGB(), # only if normal transforms not used
        A.Resize(384, 384),  # 336,336
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # only if normal transforms not used
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
)


def sample_indices(dataset_size: int, num_samples: int, seed: int = 42):
    """Get random subset of indices."""
    random.seed(seed)
    return random.sample(range(dataset_size), min(num_samples, dataset_size))


def build_dataloader(
    image_dir, annotations_file, batch_size, model, train, num_samples=None, random_idx=True
):
    # TODO: load transforms from config
    transform = TRANSFORMATIONS

    dataset = COCODataset(
        image_dir=image_dir, annotation_file=annotations_file, transform=transform
    )

    if num_samples:
        if random_idx:
            indices = sample_indices(len(dataset), num_samples)
        else:
            indices = range(num_samples)
        dataset = Subset(dataset, indices)

    processor = Processor(model, train=train)

    return DataLoader(
        dataset=dataset, batch_size=batch_size, collate_fn=processor.collate_fn
    )
