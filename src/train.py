import os
import time
import numpy as np
import torch
import wandb
from model.model import VisionLanguageModel
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_scheduler
from utils.train_utils import (
    build_train_dataloader,
    build_val_dataloader,
    parse_model_output_to_boxes,
    unnormalize_bbox,
    JSONStoppingCriteria,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch import autocast
from torch.cuda.amp import GradScaler


MODEL_NAME = "lmms-lab/llava-onevision-qwen2-0.5b-si"

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="object-dection-vlm",
    # track hyperparameters and run metadata
    # config={
    # "learning_rate": 0.02,
    # "architecture": "CNN",
    # "dataset": "CIFAR-100",
    # "epochs": 10,
    # }
)


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        device,
        gradient_accumulation_steps=4,
        max_grad_norm=None,  # 1.0,
        checkpoint_dir="checkpoints",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Initialize metric once
        self.metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(
            device
        )

        # Mixed precision
        self.scaler = GradScaler()

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        total_loss = 0
        running_loss = 0

        # add tqdm
        progress_bar = tqdm(
            self.train_dataloader, desc=f"Train/Epoch {epoch+1}/{num_epochs}"
        )

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            images = batch["images"].to(self.device)
            loss_masks = batch["loss_masks"].to(self.device)

            # Prepare lables
            labels = input_ids.clone()
            # TODO: make image_token_id as attribute
            image_token_id = self.model.tokenizer.convert_tokens_to_ids(
                self.model.image_token
            )
            labels[labels == image_token_id] = -100  # Mask image tokens
            labels[loss_masks == 0] = -100  # Mask everything except the answer tokens

            # Forward pass
            self.optimizer.zero_grad()

            with autocast(device_type=self.device):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    labels=labels,
                )
                loss = outputs.loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                # unscale gradients
                if self.max_grad_norm is not None:  # grad_clip_norm
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            running_loss = total_loss / (step + 1)

            # progress_bar.set_postfix({"loss": running_loss})
            wandb.log(
                {
                    "train/step_loss": running_loss,
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                }
            )

        progress_bar.close()
        return total_loss / len(self.train_dataloader)

    @torch.inference_mode()
    def evaluate(self, epoch, num_epochs):
        self.model.eval()
        self.metric.reset()

        progress_bar = tqdm(
            self.val_dataloader, desc=f"Eval/Epoch {epoch+1}/{num_epochs}"
        )

        for batch in progress_bar:
            # TODO: maybe better to patch images for localization performance

            # Generate predictions
            outputs = self.model.generate(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                image=batch["images"].to(self.device),
                stopping_criteria=[JSONStoppingCriteria(self.model.tokenizer)],
                temperature=0.0,
                do_sample=False,
                max_new_tokens=800,
            )

            # Decode predictions
            generated_text = self.model.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # print(generated_text)

            # Parse model ouput to bbox and lables
            predicted_boxes = [
                parse_model_output_to_boxes(
                    generated_text, self.val_dataloader.dataset, i, device
                )
                for i in range(len(batch["instance_bboxes"]))
            ]

            target_boxes = [
                {
                    # TODO: get height and width from config
                    "boxes": unnormalize_bbox(boxes.to(device), 384, 384),
                    "labels": labels.to(device),
                }
                for boxes, labels in zip(
                    batch["instance_bboxes"], batch["instance_classes_id"]
                )
            ]

            # print(predicted_boxes)
            # print(target_boxes)

            # Update metrics
            self.metric.update(predicted_boxes, target_boxes)

        # Compute final metrics
        metrics = self.metric.compute()
        return {
            "map": metrics["map"].item(),
            "map_50": metrics["map_50"].item(),
            "map_75": metrics["map_75"].item(),
        }

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.projector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_{epoch}_{int(time.time())}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)

        # _cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self, keep_last_n=3):
        checkpoints = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_")]
        )
        for checkpoint in checkpoints[:-keep_last_n]:
            os.remove(os.path.join(self.checkpoint_dir, checkpoint))

    def train(self, num_epochs=5):
        best_map = 0

        for epoch in range(num_epochs):
            # Training loop
            train_loss = self.train_epoch(epoch, num_epochs)

            # Validation
            val_metrics = self.evaluate(epoch, num_epochs)

            print(val_metrics)
            wandb.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }
            )

            is_best = val_metrics["map"] > best_map
            if is_best:
                best_map = val_metrics["map"]

            # save model projection layer after each epoch with current timestamp and epoch
            self.save_checkpoint(epoch, val_metrics, is_best)

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
# device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"Using device: {device}")

model = VisionLanguageModel(model_name=MODEL_NAME).to(device)

# Create dataloader
train_dataloader = build_train_dataloader(model, batch_size=2, num_samples=100)
val_dataloader = build_val_dataloader(model, batch_size=2, num_samples=20)

# Freeze all layers except projection layer and new token embeddings
for param in model.parameters():
    param.requires_grad = False

# Unfreeze projection layer
for param in model.projector.parameters():
    param.requires_grad = True

trainable_params = [{"params": model.projector.parameters(), "lr": 1e-4}]


optimizer = AdamW(trainable_params, lr=5e-5)

num_training_steps = len(train_dataloader) * 5
scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=num_training_steps * 0.1,
    num_training_steps=num_training_steps,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    checkpoint_dir="/u/home/salzmann/Documents/dev/checkpoints",
    gradient_accumulation_steps=4,
    # max_grad_norm=1.0
)

# Train model
trainer.train(num_epochs=10)
