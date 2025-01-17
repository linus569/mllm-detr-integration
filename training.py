import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler


def train_model(model, dataloader, optimizer, scheduler, device, num_epochs=5):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"].to(device)

            # shift input_ids for target
            labels = input_ids[:, 1:].clone()
            labels = torch.cat(
                (labels, torch.full((labels.shape[0], 1), -100, device=device)), dim=1
            )

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, images=images
            )
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
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


from processor import processor, dataloader
from model import VisionLanguageModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"Using device: {device}")
model = VisionLanguageModel(model_name="lmms-lab/llava-onevision-qwen2-0.5b-si").to(
    device
)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(dataloader) * 5
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

train_model(model, dataloader, optimizer, scheduler, device, num_epochs=5)


# Evaluate the model
results = evaluate_model(model, dataloader, processor.tokenizer, device)

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
