# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Inference for ASR model."""

import os
from logging import basicConfig, getLogger
from pathlib import Path

import torch
from nemo.collections.asr.parts.mixins.transcription import InternalTranscribeConfig, TranscribeConfig

from torch_tweak.torch import load

from .model import get_model
from .tune import parse_args

logger = getLogger(__name__)


def do_inference(
    model_name: str,
    audio_path: Path,
    tuned_model_path: Path,
):
    """Do inference on a tuned ParakeetCTC model."""
    model = get_model(model_name=model_name)
    pipeline = load(model, tuned_model_path)

    def infer(*args, **kwargs):
        return pipeline.transcribe(
            *args,
            **kwargs,
            override_config=TranscribeConfig(
                batch_size=kwargs["batch_size"],
                verbose=False,
                # TranscribeConfig.num_workers defaults to None, so NeMo starts DataLoader workers.
                # Python 3.14's forkserver cannot pickle the tokenizer into those workers.
                num_workers=0,
                _internal=InternalTranscribeConfig(device=torch.device("xpu")),
            ),
            verbose=False,
        )

    batch_size = 16
    results = infer(audio=[str(audio_path)] * batch_size, batch_size=batch_size)
    for result in results:
        logger.info(result.text)
        assert (
            result.text
            == "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait"
        )


def main():
    """Main function."""
    log_level = os.environ.get("TORCH_TWEAK_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()

    do_inference(
        model_name=args.model_name,
        audio_path=args.audio_path,
        tuned_model_path=args.tuned_model_path,
    )


if __name__ == "__main__":
    main()
