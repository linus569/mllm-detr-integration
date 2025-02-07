import albumentations as A
import numpy as np
import torch

from albumentations.pytorch import ToTensorV2
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

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

            image_mean = (0.5, 0.5, 0.5)
            image_std = (0.5, 0.5, 0.5)
            size = (384, 384)
            resample = PILImageResampling.BICUBIC
            rescale_factor = 1 / 255
            data_format = ChannelDimension.LAST  # .FIRST

            transforms = [
                convert_to_rgb,
                to_numpy_array,
                # partial(resize, size=size, resample=resample, data_format=data_format),
                partial(rescale, scale=rescale_factor, data_format=data_format),
                partial(
                    normalize, mean=image_mean, std=image_std, data_format=data_format
                ),
                # partial(to_channel_dimension_format, channel_dim=data_format, input_channel_dim=data_format),
            ]

            for transform in transforms:
                image = transform(image)

            transformed = self.transform(
                image=np.array(image),
                bboxes=instance_bboxes,
                class_labels=instance_classes,
            )
            image = transformed["image"]
            instance_bboxes = transformed["bboxes"]
            instance_classes = transformed["class_labels"]

        return {
            "image": image,
            "instance_classes": torch.tensor(instance_classes, dtype=torch.long),
            "instance_bboxes": torch.tensor(instance_bboxes, dtype=torch.float16),
            "captions": captions,
        }


TRANSFORMATIONS = A.Compose(
    [
        A.Resize(384, 384),  # 336,336
        # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
)


def build_dataloader(image_dir, annotations_file, batch_size, model, train):
    # TODO: load transforms from config
    transform = TRANSFORMATIONS

    dataset = COCODataset(
        image_dir=image_dir, annotation_file=annotations_file, transform=transform
    )

    processor = Processor(model, train=train)

    return DataLoader(
        dataset=dataset, batch_size=batch_size, collate_fn=processor.collate_fn
    )
