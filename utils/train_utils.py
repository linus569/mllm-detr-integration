from dataset import build_dataloader

# TODO: replace with config
TRAIN_DATA_DIR = "data/coco/images/train2017"
TRAIN_ANNOTATIONS_DIR = "data/coco/annotations/instances_train2017.json"
TRAIN_BATCH_SIZE = 1


def build_train_dataloader(model):
    return build_dataloader(
        image_dir=TRAIN_DATA_DIR,
        annotations_file=TRAIN_ANNOTATIONS_DIR,
        batch_size=TRAIN_BATCH_SIZE,
        model=model,
    )
