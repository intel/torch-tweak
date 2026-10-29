# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file has been modified by Intel Corporation.
from pathlib import Path

import datasets
import pytest
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding

from torch_tweak.torch.dataloader import DataLoaderFactory

PROMPTS_PATH = Path(__file__).parent.parent.parent / "fixtures/synthetic_prompts_100.json"
MAX_SEQUENCE_LENGTH = 64


def get_dataloader():
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased", revision="12040accade4e8a0f71eabdb258fecc2e7e948be"
    )

    def tokenize_function(examples):
        return tokenizer(examples["prompt"], truncation=True, max_length=MAX_SEQUENCE_LENGTH)

    dataset = datasets.load_dataset(  # nosec B615
        "json", data_files=str(PROMPTS_PATH), split="train[:99]"
    ).map(tokenize_function, remove_columns=["prompt", "act"])

    return DataLoaderFactory(
        dataset,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=MAX_SEQUENCE_LENGTH),
    ).create_dataloader(batch_size=4)


@pytest.mark.functional
def test_transformers_distilbert():
    dataloader = get_dataloader()
    data = list(dataloader)

    assert len(data) == 24
    assert data[0]["input_ids"].shape == (4, MAX_SEQUENCE_LENGTH)
    assert data[0]["attention_mask"].shape == (4, MAX_SEQUENCE_LENGTH)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
