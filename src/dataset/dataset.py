import logging
import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


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
        try:
            image_id = self.image_ids[idx]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            image_info = self.coco.loadImgs(image_id)[0]
            image_path = f"{self.image_dir}/{image_info['file_name']}"

            # Check if file exists before attempting to open
            if not os.path.isfile(image_path):
                log.error(f"Image file not found: {image_path}")
                # Return a different image if this one is not found
                return self.__getitem__((idx + 1) % len(self))

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                log.error(f"Error opening image {image_path}: {str(e)}")
                # Return a different image if this one fails to open
                return self.__getitem__((idx + 1) % len(self))

            instance_classes_id = [ann["category_id"] for ann in anns]
            instance_bboxes = [ann["bbox"] for ann in anns]
            captions = [ann["caption"] for ann in anns if "caption" in ann]

            # convert bboxes from COCO format (x,y,w,h) to (x_min, y_min, x_max, y_max)
            instance_bboxes = [[x, y, x + w, y + h] for x, y, w, h in instance_bboxes]

            instance_classes_str = [
                self.index_to_cat_name[cat_id] for cat_id in instance_classes_id
            ]

            return {
                "id": idx,
                "image": image,
                "instance_classes_id": instance_classes_id,
                "instance_classes_str": instance_classes_str,
                "instance_bboxes": instance_bboxes,
                "captions": captions,
            }
        except Exception as e:
            log.error(f"Error in __getitem__ for idx {idx}: {str(e)}")
            # If all else fails, return a different item
            if idx < len(self) - 1:
                return self.__getitem__((idx + 1) % len(self))
            else:
                # Create an empty placeholder if we're at the last index
                log.warning(f"Returning empty placeholder for idx {idx}")
                dummy_image = Image.new("RGB", (384, 384), color="black")
                return {
                    "id": idx,
                    "image": dummy_image,
                    "instance_classes_id": [],
                    "instance_classes_str": [],
                    "instance_bboxes": [],
                    "captions": [],
                }
