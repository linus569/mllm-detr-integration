import logging
import time
from typing import List

import numpy as np
import pytest


class MockProcessor:
    def __init__(self, tokenized_start_prompt: List[int]):
        self.tokenized_start_prompt = np.array(tokenized_start_prompt)

    def find_assistant_token_position(self, input_ids_np: np.ndarray) -> int:
        """Optimized search for assistant token in input_ids."""
        batch_size, seq_len = input_ids_np.shape
        token_len = len(self.tokenized_start_prompt)

        # Create result array
        positions = np.zeros(batch_size, dtype=np.int32)

        # Fast path for common case
        start_token = self.tokenized_start_prompt[0]
        for i in range(batch_size):
            potential_start = np.where(input_ids_np[i] == start_token)[0]

            for pos in potential_start:
                if pos + token_len <= seq_len:
                    if np.array_equal(
                        input_ids_np[i, pos : pos + token_len],
                        self.tokenized_start_prompt,
                    ):
                        positions[i] = pos + token_len
                        break
        return positions


def method1_sliding_window(processor, input_ids_np: np.ndarray) -> np.ndarray:
    loss_masks = np.zeros_like(input_ids_np)
    tokenized_prompt_len = len(processor.tokenized_start_prompt)

    for i in range(len(input_ids_np)):
        seq = input_ids_np[i]
        if len(seq) >= tokenized_prompt_len:
            windows = np.lib.stride_tricks.sliding_window_view(
                seq, tokenized_prompt_len
            )
            match_idx = np.where(
                (windows == processor.tokenized_start_prompt).all(axis=1)
            )[0]
            if match_idx.size > 0:
                answer_start = match_idx[0] + tokenized_prompt_len
                loss_masks[i, answer_start:] = 1
    return loss_masks


def method2_find_position(processor, input_ids_np: np.ndarray) -> np.ndarray:
    loss_masks = np.zeros_like(input_ids_np)
    positions = processor.find_assistant_token_position(input_ids_np)
    for i, pos in enumerate(positions):
        loss_masks[i, pos:] = 1
    return loss_masks


@pytest.mark.parametrize(
    "batch_size,seq_length",
    [
        # (1, 100),
        (16, 512),
        (32, 1024),
        (64, 2048),
        (512, 4096),
    ],
)
def test_performance_comparison(batch_size: int, seq_length: int):
    # Setup
    prompt = [1, 2, 3, 4, 5]  # Example prompt
    processor = MockProcessor(prompt)
    input_ids = np.random.randint(0, 1000, size=(batch_size, seq_length))

    # Insert the prompt randomly in each sequence
    for i in range(batch_size):
        pos = np.random.randint(0, seq_length - len(prompt))
        input_ids[i, pos : pos + len(prompt)] = prompt

    # Test method 1
    start_time = time.time()
    result1 = method1_sliding_window(processor, input_ids)
    time1 = time.time() - start_time

    # Test method 2
    start_time = time.time()
    result2 = method2_find_position(processor, input_ids)
    time2 = time.time() - start_time

    # Use pytest's logging functionality instead of print
    # pytestmark = pytest.mark.filterwarnings('ignore::DeprecationWarning')

    # logger = logging.getLogger(__name__)
    print(f"\nBatch size: {batch_size}, Sequence length: {seq_length}")
    print(f"Method 1 (sliding window) time: {time1:.4f} seconds")
    print(f"Method 2 (find position) time: {time2:.4f} seconds")
    print(
        f"Speed difference: Method 2 is {time1/time2:.2f}x {'faster' if time1 > time2 else 'slower'}"
    )

    # Verify both methods produce the same results
    assert np.array_equal(result1, result2), "Methods produced different results!"
