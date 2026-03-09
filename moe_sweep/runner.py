"""Sweep orchestration: generate N samples per (routing_k, temperature, sample)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from datasets import load_dataset

from src.eval.parsing import extract_tag_content, get_groundtruth

from .config import SweepConfig
from .generator import generate_batch
from .variants import create_variants
from .vllm_manager import start_vllm, stop_vllm


def load_implicature_samples(config: SweepConfig) -> list[tuple[int, dict]]:
    """Load test samples that have implicature annotations."""
    dataset = load_dataset(config.dataset_name)["train"]
    split = dataset.train_test_split(test_size=config.test_size, seed=config.seed)
    test = split["test"]

    samples = []
    for idx in range(len(test)):
        sample = test[idx]
        gt = get_groundtruth(sample)
        if gt["gt_implicature"] and gt["gt_implicature"].strip():
            samples.append((idx, sample))

    print(f"Loaded {len(samples)} implicature-bearing samples from {len(test)} test samples")
    return samples


def load_prompt(path: str) -> str:
    """Load a prompt template from file."""
    return Path(path).read_text()


def fill_speech(template: str, speech: str) -> str:
    return template.replace("{speech}", speech)


def output_path_for(config: SweepConfig, routing_k: int, temperature: float, mode: str = "separate") -> Path:
    """Build output JSONL path for a (k, temp, mode) config."""
    p = Path(config.output_dir) / f"k{routing_k}_t{temperature}_{mode}_raw.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


async def run_sweep_for_k(
    config: SweepConfig,
    routing_k: int,
    variant_path: str,
    port: int,
    model_name: str,
    gpu_ids: list[int] | None = None,
    mode: str = "separate",
    vllm_extra_args: list[str] | None = None,
) -> None:
    """Run the full temperature sweep for a single routing-k value.

    Starts vLLM, generates for all temperatures, then stops vLLM.
    """
    proc = start_vllm(variant_path, port, gpu_ids, extra_args=vllm_extra_args)
    try:
        samples = load_implicature_samples(config)
        quantity_prompt_tpl = load_prompt(config.prompt_quantity_path)
        implicature_prompt_tpl = load_prompt(config.prompt_implicature_path)

        for temp in config.temperatures:
            out_path = output_path_for(config, routing_k, temp, mode)

            # Skip if already complete
            if out_path.exists():
                existing = sum(1 for _ in open(out_path))
                if existing >= len(samples):
                    print(f"Skipping k={routing_k} t={temp}: {existing} records already exist")
                    continue

            if mode == "separate":
                await _run_separate(
                    config, samples, quantity_prompt_tpl, implicature_prompt_tpl,
                    temp, port, model_name, routing_k, out_path,
                )
            elif mode == "implicature_first":
                impl_first_prompt_tpl = load_prompt("prompts/prompt_moe_implicature_first.md")
                await _run_sequential(
                    config, samples, impl_first_prompt_tpl,
                    temp, port, model_name, routing_k, out_path,
                )
            elif mode == "quantity_first":
                qty_first_prompt_tpl = load_prompt("prompts/prompt_moe_quantity_first.md")
                await _run_sequential(
                    config, samples, qty_first_prompt_tpl,
                    temp, port, model_name, routing_k, out_path,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            print(f"Done: k={routing_k} t={temp} mode={mode} -> {out_path}")
    finally:
        stop_vllm(proc)


async def _run_separate(
    config: SweepConfig,
    samples: list[tuple[int, dict]],
    quantity_prompt_tpl: str,
    implicature_prompt_tpl: str,
    temperature: float,
    port: int,
    model_name: str,
    routing_k: int,
    out_path: Path,
) -> None:
    """Separate mode: generate quantity and implicature predictions independently."""
    # Build prompts
    qty_prompts = [(idx, fill_speech(quantity_prompt_tpl, s["user1"])) for idx, s in samples]
    impl_prompts = [(idx, fill_speech(implicature_prompt_tpl, s["user1"])) for idx, s in samples]

    # Generate in parallel
    print(f"Generating quantity predictions (k={routing_k}, t={temperature})...")
    qty_results = await generate_batch(
        qty_prompts, config.n_samples, temperature, port, model_name, config.max_concurrent_requests,
    )

    print(f"Generating implicature predictions (k={routing_k}, t={temperature})...")
    impl_results = await generate_batch(
        impl_prompts, config.n_samples, temperature, port, model_name, config.max_concurrent_requests,
    )

    # Write JSONL
    with open(out_path, "w") as f:
        for idx, sample in samples:
            gt = get_groundtruth(sample)
            # Parse quantity from each generation
            qty_preds = [extract_tag_content(g, "quantity") for g in qty_results.get(idx, [])]
            impl_preds = [extract_tag_content(g, "implicature") for g in impl_results.get(idx, [])]

            record = {
                "idx": idx,
                "speech": sample["user1"],
                "gt_quantity": gt["gt_maxim_quantity"],
                "gt_implicature": gt["gt_implicature"],
                "quantity_preds": qty_preds,
                "quantity_raw": qty_results.get(idx, []),
                "implicature_preds": impl_preds,
                "implicature_raw": impl_results.get(idx, []),
                "routing_k": routing_k,
                "temperature": temperature,
                "mode": "separate",
            }
            f.write(json.dumps(record) + "\n")


async def _run_sequential(
    config: SweepConfig,
    samples: list[tuple[int, dict]],
    prompt_tpl: str,
    temperature: float,
    port: int,
    model_name: str,
    routing_k: int,
    out_path: Path,
) -> None:
    """Sequential mode: single prompt produces both quantity and implicature."""
    prompts = [(idx, fill_speech(prompt_tpl, s["user1"])) for idx, s in samples]

    print(f"Generating sequential predictions (k={routing_k}, t={temperature})...")
    results = await generate_batch(
        prompts, config.n_samples, temperature, port, model_name, config.max_concurrent_requests,
    )

    with open(out_path, "w") as f:
        for idx, sample in samples:
            gt = get_groundtruth(sample)
            generations = results.get(idx, [])
            qty_preds = [extract_tag_content(g, "quantity") for g in generations]
            impl_preds = [extract_tag_content(g, "implicature") for g in generations]

            record = {
                "idx": idx,
                "speech": sample["user1"],
                "gt_quantity": gt["gt_maxim_quantity"],
                "gt_implicature": gt["gt_implicature"],
                "quantity_preds": qty_preds,
                "implicature_preds": impl_preds,
                "routing_k": routing_k,
                "temperature": temperature,
                "mode": prompt_tpl,  # will be overwritten below
            }
            # Detect mode from out_path
            if "implicature_first" in str(out_path):
                record["mode"] = "implicature_first"
            elif "quantity_first" in str(out_path):
                record["mode"] = "quantity_first"
            f.write(json.dumps(record) + "\n")
