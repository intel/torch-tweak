# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file has been modified by Intel Corporation.
import datasets
import pytest
from transformers import T5Tokenizer

from torch_tweak.torch.dataloader import DataLoaderFactory


@pytest.mark.functional
def test_transformers_t5_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained("t5-small", revision="df1b051c49625cf57a3d0d8d3863ed4d13564fe4")

    def tokenize(examples):
        data = tokenizer(
            examples["prompt"],
            padding="max_length",  # Pad to max_length
            truncation=True,  # Truncate if needed
            max_length=128,  # Set max sequence length
            return_tensors="pt",
        )
        return {
            "input_ids": data["input_ids"].squeeze(0),
            "attention_mask": data["attention_mask"].squeeze(0),
        }

    dataset = [{"prompt": "Studies have been shown that owning a dog is good for you"} for _ in range(10)]
    dataset = datasets.Dataset.from_list(dataset).map(tokenize, remove_columns=["prompt"])

    dataloader = DataLoaderFactory(dataset).create_dataloader(batch_size=4)

    samples = list(dataloader)
    assert len(samples) == 2
    assert samples[0]["input_ids"].shape == (4, 128), f"Invalid input shape: {samples[0]['input_ids'].shape}"
    assert samples[0]["attention_mask"].shape == (
        4,
        128,
    ), f"Invalid attention mask shape: {samples[0]['attention_mask'].shape}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
