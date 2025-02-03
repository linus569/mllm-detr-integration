import torch
from model import VisionLanguageModel
from torch.optim import AdamW
from transformers import get_scheduler
from utils.train_utils import build_train_dataloader

MODEL_NAME = "lmms-lab/llava-onevision-qwen2-0.5b-si"
TRAIN_DATA_DIR = "data/coco/images/train2017"
TRAIN_ANNOTATIONS_DIR = "data/coco/annotations/instances_train2017.json"


def train_model(model, dataloader, optimizer, scheduler, device, num_epochs=5):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"].to(device)

            # Create labels by shifting input_ids
            labels = input_ids[:, 1:].clone()
            labels = torch.cat(
                (labels, torch.full((labels.shape[0], 1), -100, device=device)), dim=1
            )
            # TODO: replace image_id values with -100
            image_token_id = model.tokenizer.convert_tokens_to_ids(model.image_token)
            labels[labels == image_token_id] = -100  # Mask image tokens
            # labels[labels >= vocab_size] = -100 # Mask tokens outside vocab range
            # labels[labels == 1040] = -100 # Mask <class_1040> tokens

            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                labels=labels,  # labels
            )
            loss = outputs.loss
            # logits = outputs.hidden_states[-1]

            # # Reshape and compute loss
            # logits = logits.view(-1, logits.shape[-1])
            # labels = labels.view(-1)

            # # Debug prints
            # if epoch == 0:
            #     print(f"Logits shape: {logits.shape}")
            #     print(f"Labels shape: {labels.shape}")
            #     print(f"Vocab size: {vocab_size}")
            #     print("Model vocab size:", model.text_encoder.config.vocab_size)

            #     print(f"Max label value: {labels.max()}")
            #     print(f"Min label value: {labels.min()}")
            #     print(f"Unique labels: {torch.unique(labels).tolist()}")
            #     print(logits.view(-1, logits.shape[-1]).size(), "and", labels.view(-1).size())

            # # Compute loss
            # loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")


def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"].to(device)

            # Generate text outputs
            outputs = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                num_beams=3,
            )

            # Decode generated text
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Post-process generated text to extract bboxes and captions
            for text in generated_texts:
                # Parse the text (e.g., using regex or JSON parsing for structured output)
                # Example: Assume output format includes JSON-like bbox and captions
                try:
                    parsed_output = eval(text)
                    results.append(parsed_output)
                except Exception as e:
                    print(f"Failed to parse text: {text}, Error: {e}")
                    results.append({"bboxes": [], "captions": []})

    return results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"Using device: {device}")

model = VisionLanguageModel(model_name=MODEL_NAME).to(device)

# Create dataloader
dataloader = build_train_dataloader(model)

# Freeze all layers except projection layer and new token embeddings
for param in model.parameters():
    param.requires_grad = False

# Unfreeze projection layer
for param in model.projector.parameters():
    param.requires_grad = True

# Unfreeze embeddings of new tokens
new_token_ids = [
    model.tokenizer.convert_tokens_to_ids(token)
    for token in model.tokenizer.additional_special_tokens
]
for token_id in new_token_ids:
    model.text_encoder.get_input_embeddings().weight[token_id].requires_grad = True

trainable_params = [
    {"params": model.projector.parameters(), "lr": 1e-4},
    {
        "params": [
            model.text_encoder.get_input_embeddings().weight[token_id]
            for token_id in model.tokenizer.convert_tokens_to_ids(
                model.tokenizer.additional_special_tokens
            )
        ],
        "lr": 1e-4,
    },
]


optimizer = AdamW(trainable_params, lr=5e-5)

num_training_steps = len(dataloader) * 5
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

train_model(model, dataloader, optimizer, scheduler, device, num_epochs=5)


# Evaluate the model
results = evaluate_model(model, dataloader, model.tokenizer, device)

print(f"Generated Results: {results}")


# # Example: Compute metrics
# from sklearn.metrics import mean_squared_error


# def compute_metrics(results, ground_truth):
#     bbox_errors = []
#     for pred, gt in zip(results, ground_truth):
#         pred_bboxes = pred.get("bboxes", [])
#         gt_bboxes = gt.get("bboxes", [])
#         if pred_bboxes and gt_bboxes:
#             bbox_errors.append(mean_squared_error(pred_bboxes, gt_bboxes))
#     return {"bbox_error": sum(bbox_errors) / len(bbox_errors)}


# # Assume ground_truth is available
# metrics = compute_metrics(results, ground_truth)
# print(f"Evaluation Metrics: {metrics}")
