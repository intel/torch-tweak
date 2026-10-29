# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file has been modified by Intel Corporation.
from pathlib import Path

import datasets
import pytest
from transformers import AutoTokenizer

from torch_tweak.torch.dataloader import DataLoaderFactory

PROMPTS_PATH = Path(__file__).parent.parent.parent / "fixtures/synthetic_prompts_100.json"


@pytest.mark.functional
def test_huggingface_dataset():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return {
            "input_ids": tokenizer(
                examples["prompt"],  # filter just text column
                padding="max_length",  # Pad to max_length
                truncation=True,  # Truncate if needed
                max_length=128,  # Set max sequence length
                return_tensors="pt",  # Return PyTorch tensors
            )["input_ids"].squeeze(0)  # Note: NOT COOL!
        }

    # Load from local fixture instead of HuggingFace API
    dataset = datasets.load_dataset("json", data_files=str(PROMPTS_PATH), split="train").map(  # nosec B615
        tokenize_function, remove_columns=["prompt", "act"]
    )

    dataloader = DataLoaderFactory(dataset).create_dataloader(batch_size=4)
    samples = list(dataloader)

    assert len(samples) == 25

    kwargs = samples[0]

    assert len(kwargs) == 1

    assert len(kwargs["input_ids"]) == 4

    assert kwargs["input_ids"].shape == (4, 128)  # Batch size 4, sequence length 128


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
