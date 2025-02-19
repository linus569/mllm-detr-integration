import random

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, Subset

from dataset.processor import Processor


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

        instance_classes = [ann["category_id"] for ann in anns]
        instance_bboxes = [ann["bbox"] for ann in anns]
        captions = [ann["caption"] for ann in anns if "caption" in ann]

        # convert bboxes from COCO format (x,y,w,h) to (x_min, y_min, x_max, y_max)
        instance_bboxes = [[x, y, x + w, y + h] for x, y, w, h in instance_bboxes]

        instance_classes_id = [
            self.cat_name_to_id[self.index_to_cat_name[cat_id]]
            for cat_id in instance_classes
        ]

        return {
            "image": image,
            "instance_classes": instance_classes,
            "instance_bboxes": instance_bboxes,
            "instance_classes_id": instance_classes_id,
            "captions": captions,
        }


def sample_indices(dataset_size: int, num_samples: int, seed: int = 42):
    """Get random subset of indices."""
    random.seed(seed)
    return random.sample(range(dataset_size), min(num_samples, dataset_size))


def build_dataloader(
    image_dir: str,
    annotations_file: str,
    batch_size: int,
    model: torch.nn.Module,
    train: bool,
    num_samples: int = None,
    random_idx: int = True,
) -> DataLoader:
    dataset = COCODataset(image_dir=image_dir, annotation_file=annotations_file)

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
