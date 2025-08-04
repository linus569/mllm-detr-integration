import os
import random
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))


from dataset.processor import Processor
from model.model import VisionLanguageModel
from utils.config import DatasetConfig, ExperimentConfig
from utils.train_metrics import TrainMetrics
from utils.train_utils import JSONStoppingCriteria, build_val_dataloader

OmegaConf.register_new_resolver(
    "ifel", lambda flag, val_true, val_false: val_true if flag else val_false
)

# load hydra configs
cs = ConfigStore.instance()
cs.store(name="ExperimentConfig", node=ExperimentConfig)
cs.store(name="DatasetConfig", group="dataset", node=DatasetConfig)
# OmegaConf.register_new_resolver("models_dir", lambda: MODELS_DIR)


def plot(id_to_cat_name, predicted_boxes, val_batch):
    # Plot predicted boxes, target boxes and labels on images
    print(predicted_boxes)

    # predicted_boxes = [{"class": [1, 32], "bbox": [[0.4879453125, 0.6142578125, 0.6474609375, 0.814453125], [0.0, 0.0, 0.99951171875, 0.9990234375]]}]

    for i in range(len(val_batch["images"])):
        fig, ax = plt.subplots()

        img, bboxes, categories = (
            val_batch["images"][i],
            predicted_boxes[i]["boxes"],
            predicted_boxes[i]["labels"],
        )

        img = img.permute(1, 2, 0).numpy()
        img = img - img.min()
        img = img / img.max()
        ax.imshow(img)

        for cat, bbox in zip(categories, bboxes):
            # print(bbox)
            x1, y1, x2, y2 = bbox  # x_min, y_min, x_max, y_max -> YOLO format
            # x1, y1, x2, y2 = x1*img.shape[1], y1*img.shape[0], x2*img.shape[1], y2*img.shape[0]
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)

            # add label text to rect
            if cat.item() in id_to_cat_name:
                class_name = id_to_cat_name[cat.item()]  # no .item()
            else:
                class_name = "Unknown"
            ax.text(x1, y1 - 5, class_name, fontsize=12, color="red")

        corr_boxes, corr_labels = (
            val_batch["instance_bboxes"][i],
            val_batch["instance_classes_id"][i],
        )

        for cat, bbox in zip(corr_labels, corr_boxes):
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = (
                x1 * img.shape[1],
                y1 * img.shape[0],
                x2 * img.shape[1],
                y2 * img.shape[0],
            )
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="g", facecolor="none"
            )
            ax.add_patch(rect)

            # add label text to rect
            if cat.item() in id_to_cat_name:
                class_name = id_to_cat_name[cat.item()]
            ax.text(x1, y1 - 5, class_name, fontsize=12, color="green")

        plt.savefig(f"test_random_{random.randint(0, 1000)}.png")


def main():
    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name="train",
            overrides=["+experiment=train_local_test", "main_dir='.'"],
        )
        # print(OmegaConf.to_yaml(config))

    MODEL_NAME = "last_model_silver-field-126.pt"
    # "last_model_legendary-cloud-125.pt"
    config.num_coordinate_bins = 100
    config.add_special_tokens = False
    config.device = "cpu"
    config.num_workers = 0

    device = torch.device(config.device)

    processor = Processor.from_config(
        config, add_special_tokens=config.add_special_tokens
    )

    model = VisionLanguageModel(
        config=config,
        image_token_index=processor.image_token_index,
        num_new_tokens=len(processor.special_tokens),
        initializers=processor.special_tokens_initializer,
        do_init=config.add_special_tokens,
    ).to(device)
    state_dict = torch.load(
        os.path.join(config.main_dir, "..", "checkpoints-trained", MODEL_NAME),
        map_location=device,
    ).get("model_state_dict")
    model.load_state_dict(state_dict)

    val_dataloader = build_val_dataloader(config, processor, 5, use_random_subset=False)
    metrics = TrainMetrics(device)

    model.eval()

    for batch in tqdm(val_dataloader):
        outputs = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            image=batch["images"].to(device),
            stopping_criteria=[JSONStoppingCriteria(processor.tokenizer)],
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            # max_new_tokens=config.max_tokens,
        )

        # Decode predictions
        generated_text, predicted_boxes = processor.postprocess_xml_batch(
            outputs, val_dataloader.dataset, device
        )

        target_boxes = processor.postprocess_target_batch(batch=batch, device=device)
        target_text = batch["bbox_str"]

        plot(val_dataloader.dataset.dataset.index_to_cat_name, predicted_boxes, batch)

        # Update metrics
        metrics.update(predicted_boxes, target_boxes, generated_text, target_text)

    # Calculate metrics
    result = metrics.compute()
    print(result)
    print(f"mAP: {result['map']}")
    print(f"mAP@50: {result['map_50']}")
    print(f"mAP@75: {result['map_75']}")
    print(f"BLEU: {result['bleu_score']}")
    print(f"METEOR: {result['meteor_score']}")


if __name__ == "__main__":
    main()
