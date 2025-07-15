import torch

from dataset.processor import Processor
from model.model import VisionLanguageModel
from utils.config import DatasetConfig, ExperimentConfig, ImageEncoderConfig

# Example config (adapt as needed)
config = ExperimentConfig(
    model_name="lmms-lab/llava-onevision-qwen2-0.5b-si",
    batch_size=1,
    device="cpu",
    image_encoder=ImageEncoderConfig(
        name="siglip",
        image_size=[384, 384],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        num_image_tokens=729,
        interpolation=2,
    ),
    add_special_tokens=True,
    val_dataset=DatasetConfig(
        name="coco_val",
        data_dir="data/coco/images/val2017",
        annotations_dir="data/coco/annotations/instances_val2017.json",
    ),
    num_workers=0,
    add_detr_layers=True,
    feedback_detr_to_llm=True,
    detr_loss=True,
    detr_type="full_detr",
    num_query_tokens=60,
    num_coordinate_bins=200,
    temperature=0.3,
    # ...add other config fields as needed
)


# Initialize processor and model
processor = Processor.from_config(config, add_special_tokens=config.add_special_tokens)
model = VisionLanguageModel(
    config=config,
    image_token_index=processor.image_token_index,
    num_new_tokens=(len(processor.special_tokens) if config.add_special_tokens else 0),
    tokenizer_size=processor.loaded_tokenizer_len,
    initializers=(
        processor.special_tokens_initializer if config.add_special_tokens else None
    ),
    do_init=config.add_special_tokens,
    query_tokens_id=(
        processor.tokenizer.encode("<query00/>")
        if config.num_query_tokens > 0
        else None
    ),
)

model.eval()

# Load checkpoint if available
checkpoint_path = 'checkpoints/last_model_electric-brook-344.pt'
checkpoint = torch.load(checkpoint_path, map_location=config.device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Example input (replace with your own image and text)
# from PIL import Image

# image_path = (
#     "data/coco/images/test2017/000000000001.jpg"  # Replace with your image path
# )
#image = Image.open(image_path).convert("RGB")

# Preprocess input
# input_prompt, _ = processor.prepare_text_input(
#     num_img_tokens=config.image_encoder.num_image_tokens,
#     instance_classes_str="",
#     instance_bboxes="",
#     captions="",
#     train=False,
#     #text="A photo of a cat on a chair.",  # Example prompt
#     #images=image,
#     #return_tensors="pt"
# )

# inputs = processor.preprocess_img_text_batch(
#     batch=[{
#         'image': image,
#         #'text': input_prompt,
#     }],
#     train=False
# )
# print("Processed inputs:", inputs)

from utils.train_utils import build_val_dataloader

dataloader = build_val_dataloader(
    config=config,
    processor=processor,
    subset_size=1,  # For testing, use a single sample
    use_random_subset=True,  # Use the first sample for testing
)

inputs = next(iter(dataloader))

#print("Inputs:", inputs)

# Move tensors to device
inputs = {
    k: v.to(config.device) if torch.is_tensor(v) else v for k, v in inputs.items()
}

# Inference
with torch.no_grad():
    # outputs = model.generate(
    #     input_ids=inputs['input_ids'],
    #     attention_mask=inputs['attention_mask'],
    #     image=inputs['pixel_values'] if 'pixel_values' in inputs else inputs['images'],
    #     max_new_tokens=64,
    #     stopping_criteria=None,
    #     image_sizes=None,
    #     tokenizer=processor.tokenizer,
    # )
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        # not .to(device) as fasterrcnn returns list, done in model
        image=inputs["images"],
        # stopping_criteria=[JSONStoppingCriteria(self.processor.tokenizer)],
        do_sample=True,  # TODO: hardcoded to config
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        image_sizes=inputs["image_sizes"],
        pixel_values=inputs["images"],  # for image_processor in full_detr
        tokenizer=processor.tokenizer,
    )

#print("Model output:", outputs)
# Decode output tokens
generated_text, predicted_boxes = processor.postprocess_xml_batch(
    outputs,
    dataset=dataloader.dataset,
    device=config.device,
    image_sizes=inputs["image_sizes"],
)
target_boxes = processor.postprocess_target_batch(batch=inputs, device=config.device)

print("Generated text:", generated_text)
print("Predicted boxes:", predicted_boxes)

index_to_cat_name = dataloader.dataset.dataset.index_to_cat_name
# Show image with predicted bounding boxes, implement function here
def show_img_with_bbox(image, predicted_boxes, target_boxes=None, title=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(image, torch.Tensor):
        img = image.cpu().numpy().transpose(1, 2, 0)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
    else:
        img = np.array(image)

    plt.imshow(img)
    plt.axis("off")

    for prediction in predicted_boxes:
        boxes, labels, scores = prediction.values()
        print(boxes, labels, scores)

        for box, label, score in zip(boxes, labels, scores):
            print(box, label, score)
            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
            )
            plt.gca().add_patch(rect)
            plt.text(
                x1,
                y1,
                f"{index_to_cat_name[label.item()]} {score:.2f}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="red", alpha=0.5),
            )
    if target_boxes:
        for target in target_boxes:
            boxes, labels = target.values()
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="g", facecolor="none"
                )
                plt.gca().add_patch(rect)
                plt.text(
                    x1,
                    y1,
                    f"{index_to_cat_name[label.item()]}",
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor="green", alpha=0.5),
                )

    if title:
        plt.title(title)

    #plt.show()
    plt.savefig("inference/output.png", bbox_inches='tight', pad_inches=0.1)


show_img_with_bbox(inputs["images"][0] , predicted_boxes, target_boxes=target_boxes, title="Predicted Bounding Boxes")

from utils.train_metrics import TrainMetrics

metrics = TrainMetrics(device=config.device, download_nltk=True)
metrics.update(
    predicted_boxes=predicted_boxes,
    target_boxes=target_boxes,
    generated_text=generated_text,
    target_texts=inputs["bbox_str"],
)

print("Metrics:")
print(metrics.metric.compute())
