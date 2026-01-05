"""Command-line interface for MLIP finetuning strategies."""

import argparse
import yaml
from pathlib import Path
import sys

from .runner import run_finetuning_experiment


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MLIP Finetuning Strategies - Systematic evaluation of continual learning"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--run-name",
        type=str,
        help="Name for this experiment run"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Directory to save experiment outputs"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Weights & Biases project name"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actual training"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    if args.run_name:
        config["run_name"] = args.run_name
    if args.wandb_project:
        config["wandb_project"] = args.wandb_project

    config["output_dir"] = args.output_dir
    config["device"] = args.device
    config["seed"] = args.seed
    config["dry_run"] = args.dry_run
    config["verbose"] = args.verbose

    # Run experiment
    try:
        run_finetuning_experiment(config)
    except KeyboardInterrupt:
        print("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running experiment: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()