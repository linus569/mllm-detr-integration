from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


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
