"""CLI for memory retention analysis."""

import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict

from .analysis import analyze_memory_retention


def parse_args():
    """Parse command line arguments for memory retention analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze memory retention during MLIP finetuning"
    )

    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to experiment directory containing checkpoints"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment config.yaml file (if not in experiment-dir)"
    )

    parser.add_argument(
        "--reference-datasets",
        type=str,
        nargs="+",
        help="Paths to reference datasets (format: name:path name:path ...)"
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project name for logging results"
    )

    parser.add_argument(
        "--wandb-run-name",
        type=str,
        help="W&B run name"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Maximum number of epochs to analyze"
    )

    parser.add_argument(
        "--epoch-step",
        type=int,
        default=1,
        help="Evaluate every N epochs (default: 1)"
    )

    parser.add_argument(
        "--subset-size",
        type=int,
        help="Use subset of dataset for faster evaluation"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for evaluation"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def load_config_from_experiment(experiment_dir: Path) -> Dict:
    """Load configuration from experiment directory."""
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def parse_reference_datasets(datasets_arg: list) -> Dict[str, str]:
    """Parse reference datasets from command line argument.

    Args:
        datasets_arg: List of strings in format "name:path"

    Returns:
        Dictionary mapping dataset names to paths
    """
    datasets = {}
    for item in datasets_arg:
        if ":" not in item:
            raise ValueError(f"Invalid dataset format: {item}. Expected 'name:path'")
        name, path = item.split(":", 1)
        datasets[name] = path
    return datasets


def extract_reference_datasets_from_config(config: Dict) -> Dict[str, str]:
    """Extract reference datasets from configuration."""
    datasets = {}

    # Check memory_retention section
    if 'memory_retention' in config:
        mem_config = config['memory_retention']
        if 'reference_datasets' in mem_config:
            datasets.update(mem_config['reference_datasets'])
        else:
            # Legacy format
            if 'reference_data_ti_contained' in mem_config:
                datasets['ti_contained'] = mem_config['reference_data_ti_contained']
            if 'reference_data_ti_excluded' in mem_config:
                datasets['ti_excluded'] = mem_config['reference_data_ti_excluded']

    # Check root level (legacy)
    if not datasets:
        if 'reference_data_ti_contained' in config:
            datasets['ti_contained'] = config['reference_data_ti_contained']
        if 'reference_data_ti_excluded' in config:
            datasets['ti_excluded'] = config['reference_data_ti_excluded']

    # Add validation split from finetuning data
    if 'data' in config and 'finetune_data' in config['data']:
        datasets['finetune_validation'] = config['data']['finetune_data']

    return datasets


def create_model_config(config: Dict, experiment_dir: Path) -> Dict:
    """Create model configuration from experiment config."""
    model_config = {
        "architecture": "nequip",  # Default
    }

    # Extract model info
    if 'model' in config:
        model_section = config['model']
        if 'architecture' in model_section:
            model_config['architecture'] = model_section['architecture']
        if 'pretrained_path' in model_section:
            model_config['pretrained_path'] = model_section['pretrained_path']
        if 'package_path' in model_section:
            model_config['pretrained_path'] = model_section['package_path']

    # Try to find pretrained model path
    if 'pretrained_path' not in model_config:
        # Look for common patterns
        possible_paths = [
            config.get('pretrained_model_path'),
            config.get('model', {}).get('model_path'),
            config.get('model', {}).get('pretrained_model_path')
        ]
        for path in possible_paths:
            if path:
                model_config['pretrained_path'] = path
                break

    return model_config


def main():
    """Main CLI entry point for memory retention analysis."""
    args = parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    # Load configuration
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        try:
            config = load_config_from_experiment(experiment_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please provide --config path or ensure config.yaml exists in experiment directory")
            sys.exit(1)

    if args.verbose:
        print(f"Loaded configuration from {config_path if args.config else experiment_dir / 'config.yaml'}")

    # Get reference datasets
    if args.reference_datasets:
        try:
            reference_datasets = parse_reference_datasets(args.reference_datasets)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        reference_datasets = extract_reference_datasets_from_config(config)

    if not reference_datasets:
        print("Error: No reference datasets found. Specify --reference-datasets or add to config.")
        sys.exit(1)

    if args.verbose:
        print(f"Reference datasets: {list(reference_datasets.keys())}")

    # Create model configuration
    model_config = create_model_config(config, experiment_dir)

    if args.verbose:
        print(f"Model config: {model_config}")

    # Run analysis
    try:
        results = analyze_memory_retention(
            experiment_dir=experiment_dir,
            reference_datasets=reference_datasets,
            model_config=model_config,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            device=args.device,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            epoch_step=args.epoch_step,
            subset_size=args.subset_size
        )

        print("\nMemory retention analysis completed successfully!")
        print(f"Results saved in: {experiment_dir / 'results'}")

        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)

        for dataset_name, data in results.items():
            if not data["epochs"]:
                continue

            epochs = data["epochs"]
            metrics = data["metrics"]

            # Core metrics
            initial_energy = metrics[0].get("energy_mae", 0) if metrics else 0
            final_energy = metrics[-1].get("energy_mae", 0) if metrics else 0

            initial_energy_per_atom = metrics[0].get("energy_mae_per_atom", 0) if metrics else 0
            final_energy_per_atom = metrics[-1].get("energy_mae_per_atom", 0) if metrics else 0

            initial_force = metrics[0].get("force_mae", 0) if metrics else 0
            final_force = metrics[-1].get("force_mae", 0) if metrics else 0

            initial_cosine = metrics[0].get("force_cosine", 0) if metrics else 0
            final_cosine = metrics[-1].get("force_cosine", 0) if metrics else 0

            # Forgetting calculations
            energy_forgetting = ((final_energy - initial_energy) / initial_energy * 100
                               if initial_energy > 0 else 0)
            energy_per_atom_forgetting = ((final_energy_per_atom - initial_energy_per_atom) / initial_energy_per_atom * 100
                                         if initial_energy_per_atom > 0 else 0)
            force_forgetting = ((final_force - initial_force) / initial_force * 100
                              if initial_force > 0 else 0)
            cosine_degradation = initial_cosine - final_cosine

            print(f"\nDataset: {dataset_name}")
            print(f"  Epochs analyzed: {len(epochs)} (from {min(epochs)} to {max(epochs)})")
            print(f"  Core Metrics:")
            print(f"    Energy MAE: {initial_energy:.4f} → {final_energy:.4f} eV ({energy_forgetting:+.1f}%)")
            print(f"    Energy MAE/atom: {initial_energy_per_atom:.4f} → {final_energy_per_atom:.4f} eV/atom ({energy_per_atom_forgetting:+.1f}%)")
            print(f"    Force MAE: {initial_force:.4f} → {final_force:.4f} eV/Å ({force_forgetting:+.1f}%)")
            print(f"    Force Cosine: {initial_cosine:.4f} → {final_cosine:.4f} (Δ{cosine_degradation:+.4f})")

    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()