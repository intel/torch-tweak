# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Tune ESM2 model."""

import logging

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

import torch_tweak.torch as tt
from torch_tweak.torch.backend import (
    TorchEagerBackend,
    TorchInductorBackend,
    XPUGraphBackend,
)

DEVICE = torch.device("xpu")
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
LOG_LEVEL = "INFO"
SAMPLE_SEQUENCE = "MQIFVKTLTGKTITLEVEPS<mask>TIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

logger = logging.getLogger(__name__)


def get_model(model_name: str = MODEL_NAME, device: torch.device = DEVICE):
    logger.info("Loading model '%s' on %s...", model_name, device)
    model = EsmForMaskedLM.from_pretrained(model_name, revision="08e4846e537177426273712802403f7ba8261b6c")
    model.to(device)
    model.eval()
    return model


def prepare_sample(device=DEVICE):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision="08e4846e537177426273712802403f7ba8261b6c")

    input_data = tokenizer([SAMPLE_SEQUENCE], return_tensors="pt")
    return {
        "input_ids": input_data["input_ids"].squeeze(0).to(device),
        "attention_mask": input_data["attention_mask"].squeeze(0).to(device),
    }


def tune(
    model_path: str = "esm2_tuned",
    model_name: str = MODEL_NAME,
    device: torch.device = DEVICE,
    batch_sizes: list[int] | None = None,
):
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
    logger.info("Tuning ESM2 model...")
    with torch.no_grad():
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]

        model = get_model(model_name=model_name, device=device)
        input_data = prepare_sample(device=device)

        logger.info("Inspecting model...")
        modules_info = tt.inspect(model, [input_data], number_of_iterations=1, warmup_iterations=1)
        modules_info.describe()

        strategy = tt.FirstWinsStrategy(
            backends=[
                TorchInductorBackend(),
                XPUGraphBackend(),
                TorchEagerBackend(),
            ]
        )
        strategy.enable_find_max_batch_size(enable=False)

        logger.info("Wrapping modules...")
        model = tt.wrap(model, modules_info.get_modules(), strategy=strategy)

        logger.info("Tuning model...")
        tt.tune(model, [input_data], batch_sizes=batch_sizes)

        logger.info("Saving model...")
        tt.save(model, model_path)


if __name__ == "__main__":
    tune()
