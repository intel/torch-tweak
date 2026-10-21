# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Benchmark hybrid Torch Tweak tuning vs single-backend pipelines (Intel XPU)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from stable_diffusion.cmd_args import parse_sizes
from stable_diffusion.model import get_pipeline
from stable_diffusion.tuning_strategy import hybrid_strategy, single_backend_strategy
from torch_tweak.torch import inspect, tune, wrap
from torch_tweak.torch.config import torch_tweak_cache_dir
from torch_tweak.torch.module.tensor_spec import InfoLevel
from torch_tweak.torch.module.wrapper_module import ModuleState
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.utils.xpu import synchronize as device_synchronize

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_SIZES = [(512, 512)]
DEFAULT_STEPS = 8


@dataclass
class BackendChoice:
    module_name: str
    graph_summary: str
    backend: str


@dataclass
class BenchResult:
    scenario: str
    seconds_per_run: float
    backend_choices: list[BackendChoice] = field(default_factory=list)


def _pick_device(device: str | None) -> str:
    if device is not None:
        return device
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _run_generation(pipe, prompt: str, sizes: list[tuple[int, int]], steps: int) -> None:
    for width, height in sizes:
        pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps)


def _collect_backend_choices() -> list[BackendChoice]:
    return [
        BackendChoice(
            module_name=name,
            graph_summary=metadata.describe(InfoLevel.MEDIUM).strip().replace("\n", " "),
            backend=backend.describe(),
        )
        for name, module in MODULE_REGISTRY.modules.items()
        if module.state == ModuleState.TUNED
        for metadata, backend in module.module.backends.items()
    ]


def _prepare_pipeline(
    model_name: str,
    device: str,
    prompt: str,
    sizes: list[tuple[int, int]],
    steps: int,
    strategy,
    min_execution_pct: float,
    clear_cache: bool,
):
    MODULE_REGISTRY.clear()
    cache_root = torch_tweak_cache_dir()
    if clear_cache and cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)

    pipe = get_pipeline(model_name=model_name, device=device)
    input_data = [{"prompt": prompt}]

    def call_for_inspect(*args, **kwargs):
        p = kwargs.get("prompt", prompt)
        if args and isinstance(args[0], str):
            p = args[0]
        _run_generation(pipe, p, sizes, steps)

    modules_info = inspect(
        pipe,
        input_data,
        inference_function=call_for_inspect,
        number_of_iterations=2,
        warmup_iterations=1,
    )
    modules = modules_info.get_modules(min_execution_percentage=min_execution_pct)
    logger.info("Wrapping %d modules (min_execution_percentage=%.2f)", len(modules), min_execution_pct)
    for m in modules:
        logger.info("  - %s", m.name)

    return wrap(pipe, modules, strategy=strategy)


def _tune_pipeline(pipe, prompt: str, sizes: list[tuple[int, int]], steps: int, batch_sizes: list[int], device: str):
    tune(
        lambda *_a, **_k: _run_generation(pipe, prompt, sizes, steps),
        [{"prompt": prompt}],
        batch_sizes=batch_sizes,
        device=device,
        clear_cache=False,
    )


def benchmark_seconds(
    pipe,
    prompt: str,
    sizes: list[tuple[int, int]],
    steps: int,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        _run_generation(pipe, prompt, sizes, steps)
    device_synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _run_generation(pipe, prompt, sizes, steps)
    device_synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iterations


def run_scenario(
    scenario: str,
    model_name: str,
    device: str,
    prompt: str,
    sizes: list[tuple[int, int]],
    steps: int,
    batch_sizes: list[int],
    strategy,
    min_execution_pct: float,
    warmup: int,
    iterations: int,
    clear_cache: bool,
) -> BenchResult:
    logger.info("=== Scenario: %s ===", scenario)
    pipe = _prepare_pipeline(
        model_name, device, prompt, sizes, steps, strategy, min_execution_pct, clear_cache=clear_cache
    )
    _tune_pipeline(pipe, prompt, sizes, steps, batch_sizes, device)
    choices = _collect_backend_choices()
    sec = benchmark_seconds(pipe, prompt, sizes, steps, warmup, iterations)
    MODULE_REGISTRY.clear()
    return BenchResult(scenario=scenario, seconds_per_run=sec, backend_choices=choices)


def _scenario_for(name: str) -> tuple[object, str]:
    if name == "hybrid":
        return hybrid_strategy(), "hybrid (HighestThroughput: OpenVINO, Inductor, Eager)"
    try:
        return single_backend_strategy(name), f"single-backend ({name})"
    except ValueError as exc:
        raise ValueError(f"Unknown scenario: {name}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid vs single-backend Stable Diffusion benchmark (Intel XPU).")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt", default="A red cube on a white table, product photo")
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default="512,512",
        help="Space-separated width,height pairs (default: 512,512)",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--min-execution-pct", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_sd_hybrid_results.json"))
    parser.add_argument(
        "--scenarios",
        default="hybrid,openvino,inductor,eager",
        help="Comma-separated: hybrid,openvino,inductor,eager",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = _pick_device(args.device)
    sizes = args.sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    results: list[BenchResult] = []
    for name in scenario_names:
        strategy, label = _scenario_for(name)
        try:
            res = run_scenario(
                scenario=label,
                model_name=args.model_name,
                device=device,
                prompt=args.prompt,
                sizes=sizes,
                steps=args.steps,
                batch_sizes=batch_sizes,
                strategy=strategy,
                min_execution_pct=args.min_execution_pct,
                warmup=args.warmup,
                iterations=args.iterations,
                clear_cache=True,
            )
            results.append(res)
            logger.info("Scenario %s: %.3f s/run", label, res.seconds_per_run)
        except Exception:
            logger.exception("Scenario %s failed", label)
            results.append(BenchResult(scenario=label, seconds_per_run=float("nan")))

    payload = {
        "device": device,
        "model_name": args.model_name,
        "sizes": sizes,
        "steps": args.steps,
        "batch_sizes": batch_sizes,
        "results": [asdict(r) for r in results],
    }
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
