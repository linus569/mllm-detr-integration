import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))


from dataset.processor import Processor
from model.model import VisionLanguageModel
from utils.config import DatasetConfig, ExperimentConfig
from utils.train_metrics import TrainMetrics
from utils.train_utils import JSONStoppingCriteria, build_val_dataloader

# OmegaConf.register_new_resolver(
#     "ifel", lambda flag, val_true, val_false: val_true if flag else val_false
# )

# load hydra configs
cs = ConfigStore.instance()
cs.store(name="ExperimentConfig", node=ExperimentConfig)
cs.store(name="DatasetConfig", group="dataset", node=DatasetConfig)
# OmegaConf.register_new_resolver("models_dir", lambda: MODELS_DIR)


def main():
    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name="train",
            overrides=["+experiment=train_local_test", "main_dir='.'"],
        )
        # print(OmegaConf.to_yaml(config))

    MODEL_NAME = "last_model_legendary-cloud-125.pt"
    config.num_coordinate_bins = 100
    config.add_special_tokens = False
    config.device = "mps"
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
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            max_new_tokens=config.max_tokens,
        )

        # Decode predictions
        generated_text, predicted_boxes = processor.postprocess_xml_batch(
            outputs, val_dataloader.dataset, device
        )

        target_boxes = processor.postprocess_target_batch(batch=batch, device=device)
        target_text = batch["bbox_str"]

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
