import logging
import os
import time

from dataset.processor import Processor
from utils.config import ExperimentConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import numpy as np
import torch
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch import autocast
from torch.amp import GradScaler
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_scheduler

from model.model import VisionLanguageModel
from utils.config import DatasetConfig
from utils.train_metrics import TrainMetrics
from utils.train_utils import (
    JSONStoppingCriteria,
    build_train_dataloader,
    build_val_dataloader,
)

hydra.verbose = True
log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: VisionLanguageModel,
        processor: Processor,
        config: ExperimentConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.processor = processor

        self.train_dataloader = None
        self.val_dataloader = None
        self.optimizer = None
        self.scheduler = None

        self.checkpoint_dir = config.checkpoint_dir
        self.gradient_accumulation_steps = config.total_batch_size // config.batch_size
        self.max_grad_norm = config.max_grad_norm

        # Initialize metric once
        self.metric = TrainMetrics(device)

        # Mixed precision
        self.scaler = GradScaler(
            "cuda" if torch.cuda.is_available() else "cpu", enabled=self.config.use_amp
        )

        self.dtype = None
        if self.config.torch_dtype is not None:
            self.dtype = getattr(torch, self.config.torch_dtype, default=None)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, num_epochs=None):
        if (
            self.train_dataloader == None
            or self.val_dataloader == None
            or self.optimizer == None
            or self.scheduler == None
        ):
            self.lazy_init_training_objects()

        if num_epochs == None:
            num_epochs = self.config.epochs

        # Initialize variables
        steps_per_epoch = len(self.train_dataloader)
        best_map = 0
        step = 0
        total_loss = 0
        running_loss = 0

        # Initialize progress bar to size of val_freq
        progress_bar = tqdm(
            total=(
                self.config.val_freq
                if self.config.val_freq is not None
                else steps_per_epoch * self.config.val_ep
            ),
            desc=f"Train/Epoch 0/{num_epochs}",
            # position=0,
            leave=False,
        )

        # Train loop
        for epoch in range(num_epochs):
            progress_bar.set_description(f"Train/Epoch {epoch+1}/{num_epochs}")

            for batch in self.train_dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.train_step(
                    step, input_ids, attention_mask, images, labels
                )

                total_loss += outputs.loss.item()
                running_loss = total_loss / (step + 1)

                progress_bar.update(1)
                step += 1

                # Log metrics
                if step % self.config.print_freq == 0:
                    progress_bar.set_postfix({"loss": running_loss})
                    wandb.log(
                        {
                            "train/loss": running_loss,
                            "train/lr": self.scheduler.get_last_lr()[0],
                        },
                        step=step,
                    )

                # Validate
                if step % self.config.val_freq == 0:
                    val_metrics = self.evaluate(step)
                    log.info(val_metrics)

                    is_best = val_metrics["map"] > best_map
                    if is_best:
                        best_map = val_metrics["map"]

                    # save model projection layer after each epoch with current timestamp and epoch
                    self.save_checkpoint(epoch, val_metrics, is_best)

                    # progress_bar.refresh()
                    progress_bar.reset()

            train_loss = total_loss / len(self.train_dataloader)
            log.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}")
            # epoch end

        progress_bar.close()
        self.save_checkpoint(epoch, val_metrics, is_last=True)

        return best_map

    def train_step(self, step, input_ids, attention_mask, images, labels):
        with autocast(
            device_type=self.device.type,
            dtype=self.dtype,  # self.model.torch_dtype,  # torch.bfloat16, more stable than float16
            enabled=self.config.use_amp,
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                labels=labels,
            )
        loss = outputs.loss / self.gradient_accumulation_steps

        if torch.isnan(loss):
            log.warning("NaN loss detected, skipping backward pass")
            return outputs

        self.scaler.scale(loss).backward()

        if (step + 1) % self.gradient_accumulation_steps == 0:
            # unscale gradients
            if self.max_grad_norm is not None:  # grad_clip_norm
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return outputs

    @torch.inference_mode()
    def evaluate(self, step, full_eval=False):
        if (
            self.train_dataloader == None
            or self.val_dataloader == None
            or self.optimizer == None
            or self.scheduler == None
        ):
            self.lazy_init_training_objects()

        if full_eval:
            val_dataloader = build_val_dataloader(
                config=self.config,
                processor=self.processor,
                subset_size=None,
            )
        else:
            # TODO: currently always same val_dataloader subset, init here again for random each eval
            val_dataloader = self.val_dataloader

        self.model.eval()
        self.metric.reset()

        progress_bar = tqdm(
            val_dataloader, desc=f"Eval/Step {step}", leave=False, position=1
        )  # , leave=False)#, position=1, leave=True)

        for batch in progress_bar:
            # Generate predictions
            with autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.config.use_amp,
            ):
                outputs = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    image=batch["images"].to(self.device),
                    stopping_criteria=[JSONStoppingCriteria(self.processor.tokenizer)],
                    do_sample=True,  # TODO: hardcoded to config
                    temperature=self.config.temperature,
                    top_p=0.9,
                    top_k=50,
                )

            # Parse model ouput to bbox and lables
            generated_text, predicted_boxes = self.processor.postprocess_xml_batch(
                outputs, val_dataloader.dataset, self.device
            )
            target_boxes = self.processor.postprocess_target_batch(
                batch=batch, device=self.device
            )

            # Update metrics
            self.metric.update(
                predicted_boxes,
                target_boxes,
                generated_text=generated_text,
                target_texts=batch["bbox_str"],
            )

        # progress_bar.clear()
        progress_bar.close()
        self.model.train()

        # Compute final metrics
        val_metrics = self.metric.compute()
        wandb.log({**{f"val/{k}": v for k, v in val_metrics.items()}}, step=step)
        return val_metrics

    def save_checkpoint(self, epoch, metrics, is_best=False, is_last=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }

        # TODO: save model as artifact to wandb
        # TODO: update best values in wandb

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{epoch}_{wandb.run.name}_{int(time.time())}.pt",
        )
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir, f"best_model_{wandb.run.name}.pt"
            )
            torch.save(checkpoint, best_path)

        if is_last:
            last_path = os.path.join(
                self.checkpoint_dir, f"last_model_{wandb.run.name}.pt"
            )
            torch.save(checkpoint, last_path)

        # _cleanup_old_checkpoints() # TODO: enable

    def _cleanup_old_checkpoints(self, keep_last_n=3):
        checkpoints = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_")]
        )
        for checkpoint in checkpoints[:-keep_last_n]:
            os.remove(os.path.join(self.checkpoint_dir, checkpoint))

    def lazy_init_training_objects(self):
        # Create dataloader
        self.train_dataloader = build_train_dataloader(
            config=self.config,
            processor=self.processor,
            subset_size=self.config.num_samples,
        )
        self.val_dataloader = build_val_dataloader(
            config=self.config,
            processor=self.processor,
            subset_size=self.config.val_num_samples,
        )
        epochs = self.config.epochs

        # Create optimizer, different lr for projection layer and frozen embedding layers
        # Group parameters by learning rate
        projection_params = []
        frozen_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            elif "projector" in name:
                projection_params.append(param)
            elif "embedding" in name:
                frozen_params.append(param)
            else:
                other_params.append(param)

        params = [p for p in self.model.parameters() if p.requires_grad]
        log.info(f"Number of trainable parameters: {sum(p.numel() for p in params)}")
        log.info(
            f"Projection params: {sum(p.numel() for p in projection_params)}, Embedding params: {sum(p.numel() for p in frozen_params)}, Other params: {sum(p.numel() for p in other_params)}"
        )

        # Create parameter groups with different learning rates
        param_groups = [
            # Higher LR for projection
            {"params": projection_params, "lr": self.config.lr * 1.0},
            # Lower LR for embeddings
            {"params": frozen_params, "lr": self.config.lr * 0.1},
            # Default LR for rest
            {"params": other_params, "lr": self.config.lr},
        ]

        # TODO: add weight_decay to conf
        self.optimizer = AdamW(param_groups, lr=self.config.lr)

        # Create lr scheduler
        num_training_steps = len(self.train_dataloader) * epochs
        if self.gradient_accumulation_steps > 1:
            num_training_steps = num_training_steps // self.gradient_accumulation_steps
        warmup_steps = num_training_steps * self.config.warmup_ratio
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        log.info(
            f"LR scheduler with {num_training_steps} training steps and warmup {warmup_steps}"
        )
        log.info(f"Training objects initialized")


@hydra.main(config_path="../conf", config_name="train", version_base=None)
def run_training(config: ExperimentConfig):
    log.info(OmegaConf.to_yaml(config))

    # Init wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="object-dection-vlm",
        # track hyperparameters and run metadata
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=False),
        # settings=wandb.Settings(start_method='thread', init_timeout=300, _service_wait=300),
        mode="disabled" if config.debug else "online",
        # TODO: log experiment_notes, name, tags, pre_trained, model_name
    )

    # TODO: log files as artifacts

    device = torch.device("cuda" if torch.cuda.is_available() else config.device)
    log.info(f"Using device: {device}")

    processor = Processor.from_config(
        config, add_special_tokens=config.add_special_tokens
    )
    model = VisionLanguageModel(
        config=config,
        image_token_index=processor.image_token_index,
        num_new_tokens=(
            len(processor.special_tokens) if config.add_special_tokens else 0
        ),
        initializers=(
            processor.special_tokens_initializer if config.add_special_tokens else None
        ),
        do_init=config.add_special_tokens,
    ).to(device)

    if config.load_checkpoint:
        path = os.path.join(config.checkpoint_dir, config.load_checkpoint)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if not config.debug:
        model = torch.compile(model)  # 2.3 it/s without -> 4.5 it/s with

    # Initialize trainer
    trainer = Trainer(
        model=model,
        processor=processor,
        config=config,
        device=device,
    )

    # Train model
    if config.train:
        best_result = trainer.train()
        log.info(f"Best result: {best_result}")

    if config.evaluate:
        # Evaluate model
        eval_results = trainer.evaluate(0, full_eval=True)
        wandb.log(eval_results)
        wandb.summary.update(eval_results)

    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cs = ConfigStore.instance()
    cs.store(name="ExperimentConfig", node=ExperimentConfig)
    cs.store(name="DatasetConfig", group="dataset", node=DatasetConfig)
    # OmegaConf.register_new_resolver("models_dir", lambda: MODELS_DIR)
    OmegaConf.register_new_resolver(
        "ifel", lambda flag, val_true, val_false: val_true if flag else val_false
    )
    # OmegaConf.register_new_resolver("project_dir", lambda: PROJECT_DIR)

    # import model
    # from model import img_encoder, txt_encoder, txt_decoder, detector
    # ModelRegistry.init_registries([model, img_encoder, txt_encoder, txt_decoder, detector])

    run_training()
