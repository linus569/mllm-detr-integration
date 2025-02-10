import copy
from matplotlib import pyplot as plt
import requests
import torch
from model.model import VisionLanguageModel
from PIL import Image
import numpy as np
from tqdm import tqdm
from dataset.processor import Processor

from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from dataset.dataset import build_dataloader
from utils.train_utils import build_test_dataloader, build_val_dataloader


MODEL_NAME = "lmms-lab/llava-onevision-qwen2-0.5b-si"
TEST_DATA_DIR = "data/coco/images/test2017"
TEST_ANNOTATIONS_DIR = "data/coco/annotations/image_info_test2017.json"


def load_model(model_path=None, device="cuda"):
    """
    Load the vision language model from checkpoint or pretrained weights
    """
    model = VisionLanguageModel(model_name=MODEL_NAME).to(device)

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    return model

def evaluate_batch(model, image_paths, processor, device="cuda"):
    """
    Run inference on a batch of images
    """
    # results = []

    # for image_path in tqdm(image_paths, desc="Running inference"):
    #     try:
    #         generated_text = run_inference(model, image_path, processor, device)
    #         results.append({"image_path": image_path, "generated_text": generated_text})
    #     except Exception as e:
    #         print(f"Error processing {image_path}: {str(e)}")
    #         results.append({"image_path": image_path, "error": str(e)})

    # return results
    pass


def lmms_lab_inference_example_adapted(
    model, dataloader, device="cpu"
):
    for batch in dataloader:
        image_tensor = batch["images"]
        input_ids = batch["input_ids"]
        # input_ids[input_ids==151646] = -200 # use when using processing of llava model
        attention_mask = batch["attention_mask"]

        cont = model.generate(  # .text_encoder
            input_ids,
            image=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            attention_mask=attention_mask,
        )
        text_outputs = model.tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(f"Generated text: {text_outputs}")



def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(
        model_path=None, device=device
    )  # Update with your model checkpoint path
    processor = Processor(model)

    # Dataloader
    # test_dataset has no labels
    dataloader = build_val_dataloader(model, num_samples=1, random_index=False)

    print("\nRunning single image inference...")

    lmms_lab_inference_example_adapted(
        model,
        dataloader=dataloader,
        device=device,
    )
    # result = run_inference(model, test_image_path, processor, device)


    # Example batch inference
    # test_image_paths = ["path1.jpg", "path2.jpg"]  # Update with your test image paths
    # print("\nRunning batch inference...")
    # results = evaluate_batch(model, test_image_paths, processor, device)

    # # Save results
    # import json
    # with open('evaluation_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
