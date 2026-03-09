"""MoE Routing-k Sweep CLI.

Usage:
    python -m moe_sweep create-variants
    python -m moe_sweep generate --ks 1,2,4 --gpus 0,1,2
    python -m moe_sweep judge
    python -m moe_sweep metrics
    python -m moe_sweep analyze
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from dotenv import load_dotenv

from .config import load_sweep_config


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="MoE Routing-k Sweep")
    parser.add_argument("command", choices=["create-variants", "generate", "judge", "metrics", "analyze"])
    parser.add_argument("--config", default="moe_sweep/sweep_config.yaml", help="Sweep config YAML")
    parser.add_argument("--ks", type=str, default=None, help="Comma-separated routing-k values to run (default: all from config)")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs (one vLLM per k)")
    parser.add_argument("--port", type=int, default=None, help="Override vLLM base port")
    parser.add_argument("--mode", choices=["separate", "implicature_first", "quantity_first"], default="separate")
    parser.add_argument("--model-name", type=str, default=None, help="Model name for vLLM API (auto-detected from config if not set)")
    parser.add_argument("--vllm-args", type=str, default=None, help="Extra vLLM args (space-separated)")
    parser.add_argument("-n", type=int, default=None, help="Override n_samples")

    args = parser.parse_args()
    config = load_sweep_config(args.config)

    if args.n is not None:
        config.n_samples = args.n
    if args.port is not None:
        config.vllm_base_port = args.port

    ks = config.routing_ks
    if args.ks:
        ks = [int(k) for k in args.ks.split(",")]

    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]

    vllm_extra = args.vllm_args.split() if args.vllm_args else None

    if args.command == "create-variants":
        from .variants import create_variants, verify_variants
        create_variants(config.model_base_path, ks)
        verify_variants(config.model_base_path, ks)

    elif args.command == "generate":
        from pathlib import Path
        from .runner import run_sweep_for_k

        model_name = args.model_name or Path(config.model_base_path).name

        # Run each k sequentially (each needs its own vLLM instance)
        # If multiple GPUs provided, assign round-robin
        for i, k in enumerate(ks):
            variant_path = str(Path(config.model_base_path) / "variants" / f"ept{k}")
            port = config.vllm_base_port + i

            # Assign GPUs round-robin if provided
            k_gpus = None
            if gpu_ids:
                k_gpus = [gpu_ids[i % len(gpu_ids)]]

            print(f"\n{'='*50}")
            print(f"Running k={k} on port {port} (GPUs: {k_gpus})")
            print(f"{'='*50}")

            asyncio.run(run_sweep_for_k(
                config, k, variant_path, port, model_name,
                gpu_ids=k_gpus, mode=args.mode, vllm_extra_args=vllm_extra,
            ))

    elif args.command == "judge":
        from .judge_batch import judge_all_outputs
        asyncio.run(judge_all_outputs(config))

    elif args.command == "metrics":
        from .metrics import compute_sweep_metrics, print_sweep_results
        df = compute_sweep_metrics(config.output_dir)
        print_sweep_results(df)
        if not df.empty:
            from pathlib import Path
            out = Path(config.output_dir) / "sweep_results.csv"
            df.to_csv(out, index=False)
            print(f"\nSaved to {out}")

    elif args.command == "analyze":
        from .analyze import analyze
        analyze(config.output_dir)


if __name__ == "__main__":
    main()
