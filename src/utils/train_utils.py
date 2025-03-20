from functools import cached_property
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from transformers import StoppingCriteria

from dataset.dataset import COCODataset
from dataset.processor import Processor
from utils.config import DatasetConfig, ExperimentConfig

log = logging.getLogger(__name__)


def collate_fn(train, processor):
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function to be used with PyTorch DataLoader.

        Args:
            samples: List of batch dictionaries

        Returns:
            Processed batch dictionary
        """
        return processor.preprocess_img_text_batch(batch, train=train)

    return _collate_fn


def sample_indices(dataset_size: int, num_samples: int, seed: int = 42):
    """Get random subset of indices."""
    random.seed(seed)
    return random.sample(range(dataset_size), min(num_samples, dataset_size))


def build_dataloader(
    processor: Processor,
    dataset_config: DatasetConfig,
    is_train: bool,
    batch_size: int,
    num_workers: int = 0,
    load_precomputed_embeddings: bool = False,
    subset_size: Optional[int] = None,
    use_random_subset: int = True,
) -> DataLoader:
    """Builds a PyTorch DataLoader for a given dataset."""

    if load_precomputed_embeddings:
        raise NotImplementedError("Precomputed embeddings are not supported yet")

    dataset = COCODataset(
        image_dir=dataset_config.data_dir,
        annotation_file=dataset_config.annotations_dir,
    )

    # Allow for subset of dataset
    if subset_size:
        if use_random_subset:
            indices = sample_indices(len(dataset), subset_size)
        else:
            indices = range(subset_size)
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=num_workers,
        prefetch_factor=3 if num_workers > 0 else None,
        collate_fn=collate_fn(processor=processor, train=is_train),
        pin_memory=True,
        persistent_workers=False,
    )


def build_train_dataloader(config: ExperimentConfig, processor: Processor, subset_size=None):
    return build_dataloader(
        processor=processor,
        dataset_config=config.train_dataset,
        is_train=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers, 
        load_precomputed_embeddings=False,
        subset_size=subset_size,
        use_random_subset=True,
    )


def build_val_dataloader(
    config: ExperimentConfig, processor: Processor, subset_size=None, use_random_subset=True
):
    return build_dataloader(
        processor=processor,
        dataset_config=config.val_dataset,
        is_train=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers, 
        load_precomputed_embeddings=False,
        subset_size=subset_size,
        use_random_subset=use_random_subset,
    )
    


def build_test_dataloader(
    config: ExperimentConfig, processor: Processor, subset_size=None, use_random_subset=True
):
    return build_dataloader(
        processor=processor,
        dataset_config=config.train_dataset,
        is_train=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers, 
        load_precomputed_embeddings=False,
        subset_size=subset_size,
        use_random_subset=use_random_subset,
    )


def seed_everything(seed):
    pass


def save_training_checkpoint(todo):
    pass


class JSONStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @cached_property
    def end_sequence(self):
        return self.tokenizer.encode(
            "<|im_end|>"  # "]}]<|im_end|>"
        )  # Get token ID for closing bracket
    
    @cached_property
    def length(self):
        return len(self.end_sequence)

    def __call__(self, input_ids, scores, **kwargs):
        # Stop if we find the end sequence
        # print(input_ids[0][-self.length :])
        # print(self.end_sequence)
        return input_ids[0][-self.length :] == self.end_sequence
        # return input_ids[0][-1] == self.end_sequence
