import random
from dataclasses import MISSING, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import PreTrainedTokenizerFast as Tokenizer

from dataset.processor import Processor


@dataclass
class DatasetConfig:
    name: str = MISSING

    # --- Normalization parameters ---
    # pixel_mean: List[float] = MISSING
    # pixel_std: List[float] = MISSING

    # --- Dataset parameters ---
    data_dir: str = MISSING
    annotations_dir: str = MISSING


class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_ids = sorted([cat["id"] for cat in self.categories])
        self.cat_name_to_id = {cat["name"]: cat["id"] for cat in self.categories}
        self.index_to_cat_name = {cat["id"]: cat["name"] for cat in self.categories}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx) -> dict[str, Image.Image | list]:
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.image_dir}/{image_info['file_name']}"
        image = Image.open(image_path).convert("RGB")

        instance_classes_id = [ann["category_id"] for ann in anns]
        instance_bboxes = [ann["bbox"] for ann in anns]
        captions = [ann["caption"] for ann in anns if "caption" in ann]

        # convert bboxes from COCO format (x,y,w,h) to (x_min, y_min, x_max, y_max)
        instance_bboxes = [[x, y, x + w, y + h] for x, y, w, h in instance_bboxes]

        instance_classes_str = [
            self.index_to_cat_name[cat_id] for cat_id in instance_classes_id
        ]

        return {
            "image": image,
            "instance_classes_id": instance_classes_id,
            "instance_classes_str": instance_classes_str,
            "instance_bboxes": instance_bboxes,
            "captions": captions,
        }


def sample_indices(dataset_size: int, num_samples: int, seed: int = 42):
    """Get random subset of indices."""
    random.seed(seed)
    return random.sample(range(dataset_size), min(num_samples, dataset_size))


def build_dataloader(
    config,
    dataset_config: DatasetConfig,
    batch_size: int,
    is_train: bool,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (384, 384),
    num_image_tokens: int = 729,
    subset_size: Optional[int] = None,
    use_random_subset: int = True,
) -> DataLoader:
    dataset = COCODataset(
        image_dir=dataset_config.data_dir,
        annotation_file=dataset_config.annotations_dir,
    )

    if subset_size:
        if use_random_subset:
            indices = sample_indices(len(dataset), subset_size)
        else:
            indices = range(subset_size)
        dataset = Subset(dataset, indices)

    processor = Processor(
        config=config,
        img_size=image_size,
        num_img_tokens=num_image_tokens,
        train=is_train,
    )

    # TODO: switch between batch_size for train and val?
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=3 if num_workers > 0 else None,
        collate_fn=processor.collate_fn,
        persistent_workers=False,
    )
