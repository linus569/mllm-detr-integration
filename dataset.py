import torch
import albumentations as A
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO


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


transform = A.Compose(
    [
        A.Resize(336, 336),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
)

train_data_dir = "data/coco/images/train2017"
train_annotation_file = "data/coco/annotations/instances_train2017.json"

train_dataset = COCODataset(
    image_dir=train_data_dir,
    annotation_file=train_annotation_file,
    transform=transform,
)
