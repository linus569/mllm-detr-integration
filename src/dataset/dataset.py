import albumentations as A
import numpy as np
import torch

from albumentations.pytorch import ToTensorV2
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from dataset.processor import Processor
from transformers import AutoTokenizer


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


TRANSFORMATIONS = A.Compose(
    [
        A.Resize(384, 384),  # 336,336
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
)


def build_dataloader(image_dir, annotations_file, batch_size, model):
    # TODO: load transforms from config
    transform = TRANSFORMATIONS

    dataset = COCODataset(
        image_dir=image_dir, annotation_file=annotations_file, transform=transform
    )

    processor = Processor(model)

    return DataLoader(
        dataset=dataset, batch_size=batch_size, collate_fn=processor.collate_fn
    )


# test implementation of dataset, processor, and integration with model
if __name__ == "__main__":
    from model.model import VisionLanguageModel

    train_data_dir = "data/coco/images/train2017"
    train_annotation_file = "data/coco/annotations/instances_train2017.json"
    model_name = "lmms-lab/llava-onevision-qwen2-0.5b-si"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = VisionLanguageModel(model_name)
    print(model)

    # print number of image-tokens when forward in vision tower
    print(
        "vision_tower output shape:",
        model.vision_tower.forward(torch.randn(1, 3, 384, 384)).last_hidden_state.shape,
    )
    print(
        "num_image_token of vision_tower:",
        model.vision_tower.vision_model.embeddings.position_embedding.weight.shape[0],
    )

    dataloader = build_dataloader(
        image_dir=train_data_dir,
        annotations_file=train_annotation_file,
        batch_size=1,
        model=model,
    )

    batch = next(iter(dataloader))
    images = batch["images"]
    input_ids = batch["input_ids"]
    att_mask = batch["attention_mask"]

    encoded_images = model.vision_tower.forward(images)
    print("encoded_images last_hidden_state:", encoded_images.last_hidden_state.shape)
    print("encoded_images pooler_output:", encoded_images.pooler_output.shape)

    model_output = model(input_ids=input_ids, attention_mask=att_mask, images=images)

    print(model_output)
