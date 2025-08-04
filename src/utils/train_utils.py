import logging
import random
from functools import cached_property
from typing import Any, Dict, List, Optional

import numpy as np
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
            batch: List of batch dictionaries

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
    use_random_subset: bool = True,
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


def build_train_dataloader(
    config: ExperimentConfig, processor: Processor, subset_size=None
):
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
    config: ExperimentConfig,
    processor: Processor,
    subset_size=None,
    use_random_subset=True,
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
    config: ExperimentConfig,
    processor: Processor,
    subset_size=None,
    use_random_subset=True,
):
    return build_dataloader(
        processor=processor,
        dataset_config=config.test_dataset,
        is_train=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        load_precomputed_embeddings=False,
        subset_size=subset_size,
        use_random_subset=use_random_subset,
    )


def seed_everything(seed):
    """
    Set the seed for all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # Set deterministic behavior for CUDA if available
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class JSONStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @cached_property
    def end_sequence(self):
        # Get token ID for closing bracket
        return self.tokenizer.encode("<|im_end|>")  # "]}]<|im_end|>"

    @cached_property
    def length(self):
        return len(self.end_sequence)

    def __call__(self, input_ids, scores, **kwargs):
        # return input_ids[0][-self.length :] == self.end_sequence
        should_stop = torch.ones(input_ids.shape[0], dtype=torch.bool)

        for i in range(input_ids.shape[0]):
            # Stop if we find the end sequence
            if input_ids[i][-self.length :].tolist() != self.end_sequence:
                should_stop[i] = False

        return should_stop.all()
