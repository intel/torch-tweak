# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Tune ResNet model."""

import logging
from logging import basicConfig

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from llm.cmd_args import get_tune_parser
from torch_tweak.torch import OneBackendStrategy, TunableModule
from torch_tweak.torch import tune as torch_tweak
from torch_tweak.torch.backend import TorchEagerBackend, TorchInductorBackend, TorchInductorBackendConfig


def get_model_and_tokenizer(model_id="microsoft/Phi-3-mini-4k-instruct"):
    """Get the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        trust_remote_code=False,
        revision="f39ac1d28e925b323eae81227eaba4464caced4e",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding_side="left", revision="f39ac1d28e925b323eae81227eaba4464caced4e"
    )
    tokenizer.pad_token = tokenizer.eos_token  # some models does not have it set
    return model.to("xpu"), tokenizer


def print_conversation(output, title="CONVERSATION OUTPUT"):
    """Print the conversation in a readable format.

    Args:
        output: The pipeline output containing generated text with messages.
    """
    print("=" * 60)
    print(title)
    print("=" * 60)
    for message in output[0]["generated_text"]:
        role = message["role"].upper()
        content = message["content"]
        print(f"{role}: {content}\n")


def tune_model(model, tokenizer, cache="static"):
    """Tune the model.

    There are different strategies to tune the model depending on the cache implementation:
    - Dynamic cache: is not compile friendly and is ignored, hence we can't distinguish between prefill and decode phases.
    - Static cache: For prefill phase we have to use torch eager backend because torch.compile would trigger every time
    as the prompt sequence length changes. However for decode phase we can use torch inductor backend as there is
    always static sequence length = 1 and static cache is torch.compile friendly.

    Args:
        model: The model to tune.
        tokenizer: The tokenizer to use.
        cache: The cache implementation to use.

    Returns:
        The tuned model.

    """
    if cache == "dynamic":
        model = TunableModule(
            model,
            model.__class__.__name__,
            strategy=OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(
                False
            ),  # just one strategy, there is no prefill/decode distinction
        )
    elif cache == "static":
        model = TunableModule(
            model,
            model.__class__.__name__,
            strategies=[
                OneBackendStrategy(TorchEagerBackend()).enable_find_max_batch_size(False),  # for prefill phase
                OneBackendStrategy(
                    TorchInductorBackend(TorchInductorBackendConfig(mode="reduce-overhead"))
                ).enable_find_max_batch_size(False),  # for decode phase
            ],
        )

    generate_args = {
        "do_sample": False,
        "disable_compile": True,
        "cache_implementation": cache,
        "use_cache": True,
    }

    def pipe(messages):
        inputs = tokenizer(messages, return_tensors="pt", padding=True)
        inputs = {k: v.to("xpu") for k, v in inputs.items()}
        with torch.no_grad():
            return model.generate(**inputs, **generate_args)

    torch_tweak(pipe, ["2+2?", "How big is the universe?"], batch_sizes=[1, 2], device="xpu", dry_run=False)
    return model


def main():
    """Entry point for the script."""
    basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = get_tune_parser().parse_args()

    generate_args = {
        "do_sample": False,  # make deterministic
        "disable_compile": True,
        "cache_implementation": args.cache,
        "use_cache": True,
        "max_new_tokens": 512,
    }
    messages = [
        {
            "role": "user",
            "content": "How big is the universe?",
        }
    ]
    model, tokenizer = get_model_and_tokenizer(args.model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    original_output = pipe(messages, **generate_args)

    model, tokenizer = get_model_and_tokenizer(args.model_id)
    tuned_model = tune_model(model, tokenizer, args.cache)
    pipe = pipeline("text-generation", model=tuned_model, tokenizer=tokenizer)
    tuned_output = pipe(messages, **generate_args)

    print_conversation(original_output, title="Original model output")
    print_conversation(tuned_output, title="Tuned model output")


if __name__ == "__main__":
    main()
