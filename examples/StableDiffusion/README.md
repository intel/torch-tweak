<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
NOTE: This file has been modified by Intel Corporation.
-->

# Stable Diffusion tuning with Torch Tweak

This example shows how to tune a Hugging Face **Stable Diffusion** pipeline with Torch Tweak.
Selected `nn.Module` blocks are wrapped, and a strategy picks backends per module and per input graph.

**Backends:** Torch Inductor (`torch.compile`), OpenVINO, then Torch Eager. See `stable_diffusion/tuning_strategy.py`.

**Default model:** `stable-diffusion-v1-5/stable-diffusion-v1-5` (fits ~12 GB VRAM; larger checkpoints such as SD 3 need much more memory).

## Environment Setup

From this example directory (`examples/StableDiffusion`). PyTorch comes from the **`xpu-stable`** dependency group (Intel XPU wheels). Load oneAPI before `uv sync` / `uv run`.

```bash
for _d in "${ONEAPI_ROOT:+$ONEAPI_ROOT}" "$HOME/intel/oneapi" /opt/intel/oneapi /usr/local/intel/oneapi; do
  [ -n "$_d" ] && [ -f "$_d/setvars.sh" ] && . "$_d/setvars.sh" && break
done
cd examples/StableDiffusion
uv sync
uv run benchmark-hybrid --help
```

Nightly PyTorch:
```bash
export UV_PROJECT_ENVIRONMENT=.venv-nightly
uv sync --no-default-groups --group xpu-nightly
```

In case you are a developer making change in Torch Tweak sources, then reinstall it:
```bash
uv sync --no-default-groups --group xpu-nightly --reinstall-package torch_tweak
```


## Usage

### Tuning the model

```bash
cd examples/StableDiffusion
uv run tune --prompt "A futuristic cityscape with neon lights"
```

Common CLI options (shared by `tune` and `inference`):

- `--model-name`: Hugging Face model id (default: SD 1.5)
- `--prompt`: Text prompt for image generation
- `--sizes`: Space-separated `width,height` pairs (default: `512,512`)
- `--steps`: Inference steps (default: 50)
- `--tuned-model-path`: Path to save or load the tuned checkpoint (default: `stable_diffusion.tt`)

### Hybrid vs single-backend benchmark

Compare **HighestThroughput** (Inductor, OpenVINO, and Eager per module) against single-backend baselines:

```bash
uv run benchmark-hybrid \
  --model-name stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --sizes "512,512" \
  --steps 8 \
  --scenarios hybrid,inductor,eager,openvino \
  --output-json benchmark_sd_hybrid_results.json
```

Use `--scenarios hybrid` for a quicker run. Results include per-module backend choices and seconds per pipeline run.

### Generating images with the tuned model

After tuning, generate images with:

```bash
uv run inference --prompt "A beautiful landscape with mountains and a lake" --output-dir output
```

The generated image will be saved in the specified output directory.

#### Dynamic batching

The service uses dynamic batching - requests are grouped and processed together for efficiency. Currently, there is one frontend and one worker. To support multiple workers, move batching to a separate service that handles request grouping.

## Model Details

The Stable Diffusion model is a text-to-image diffusion model that generates high-quality images from text descriptions. The model is trained on a large dataset of images and text, and can generate realistic images across various domains.

For more information, visit the [Stable Diffusion model page on HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers).
