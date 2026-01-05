"""Memory retention analysis for MLIP finetuning experiments.

This module provides tools to analyze how well a model retains its knowledge
from pretraining after being finetuned on new data.
"""

import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from ..models import MLIPModel
from ..data import MLIPDataset, create_dataloader
from ..utils import compute_metrics, setup_wandb_logging, log_metrics


class MemoryRetentionAnalyzer:
    """Analyzer for memory retention during finetuning."""

    def __init__(
        self,
        experiment_dir: Union[str, Path],
        model_config: Dict,
        device: str = "auto"
    ):
        """Initialize memory retention analyzer.

        Args:
            experiment_dir: Path to experiment directory containing checkpoints
            model_config: Model configuration dictionary
            device: Device to use for evaluation ("auto", "cpu", "cuda", "mps")
        """
        self.experiment_dir = Path(experiment_dir)
        self.model_config = model_config
        self.device = self._get_device(device)

        # Paths
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir = self.experiment_dir / "models"
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir = self.experiment_dir

        self.results_dir = self.experiment_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

    def _get_device(self, device_arg: str) -> torch.device:
        """Get appropriate device for evaluation."""
        if device_arg == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_arg)

    def get_checkpoint_paths(self, max_epochs: Optional[int] = None) -> List[Tuple[int, str]]:
        """Get all checkpoint paths sorted by epoch.

        Args:
            max_epochs: Maximum number of epochs to include

        Returns:
            List of (epoch, checkpoint_path) tuples
        """
        checkpoint_paths = []

        # Look for checkpoint files with various patterns
        patterns = [
            self.checkpoint_dir / "checkpoint_epoch_*.pt",
            self.checkpoint_dir / "checkpoint_epoch_*.pth",
            self.checkpoint_dir / "epoch_*.pt",
            self.checkpoint_dir / "epoch_*.pth"
        ]

        for pattern in patterns:
            checkpoints = glob.glob(str(pattern))
            for ckpt in checkpoints:
                # Extract epoch number from filename
                basename = os.path.basename(ckpt)
                for regex_pattern in [r'checkpoint_epoch_(\d+)\.pt', r'checkpoint_epoch_(\d+)\.pth',
                                     r'epoch_(\d+)\.pt', r'epoch_(\d+)\.pth']:
                    match = re.search(regex_pattern, basename)
                    if match:
                        epoch = int(match.group(1))
                        if max_epochs is None or epoch <= max_epochs:
                            checkpoint_paths.append((epoch, ckpt))
                        break

        # Remove duplicates and sort by epoch
        checkpoint_paths = list(set(checkpoint_paths))
        checkpoint_paths.sort(key=lambda x: x[0])

        return checkpoint_paths

    def load_model_from_checkpoint(self, checkpoint_path: str) -> MLIPModel:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded MLIP model
        """
        # Load pretrained model first
        model = MLIPModel(
            architecture=self.model_config["architecture"],
            model_path=self.model_config.get("pretrained_path")
        )
        model = model.to(self.device)

        # Load checkpoint state
        if checkpoint_path.endswith(('.pt', '.pth')):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Extract state dict from checkpoint
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Load state dict
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model

    def evaluate_model_on_dataset(
        self,
        model: MLIPModel,
        dataset_path: str,
        batch_size: int = 32,
        subset_size: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.

        Args:
            model: MLIP model to evaluate
            dataset_path: Path to evaluation dataset
            batch_size: Batch size for evaluation
            subset_size: Optional subset size for faster evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        # Create dataset and dataloader
        dataset = MLIPDataset(dataset_path)

        # Use subset if specified
        if subset_size and subset_size < len(dataset):
            indices = torch.randperm(len(dataset))[:subset_size]
            dataset = torch.utils.data.Subset(dataset, indices)

        dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions = {}
        all_targets = {}

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                predictions = model(batch)

                # Collect predictions and targets
                for key in predictions:
                    if key not in all_predictions:
                        all_predictions[key] = []
                        all_targets[key] = []
                    all_predictions[key].append(predictions[key].cpu())
                    if key in batch:
                        all_targets[key].append(batch[key].cpu())

        # Concatenate all predictions and targets
        for key in all_predictions:
            all_predictions[key] = torch.cat(all_predictions[key])
            if key in all_targets and all_targets[key]:
                all_targets[key] = torch.cat(all_targets[key])

        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets)
        return metrics

    def analyze_retention(
        self,
        reference_datasets: Dict[str, str],
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        batch_size: int = 32,
        max_epochs: Optional[int] = None,
        epoch_step: int = 1,
        subset_size: Optional[int] = None
    ) -> Dict[str, Dict]:
        """Analyze memory retention across training epochs.

        Args:
            reference_datasets: Dict mapping dataset names to paths
            wandb_project: W&B project name for logging
            wandb_run_name: W&B run name
            batch_size: Batch size for evaluation
            max_epochs: Maximum epochs to analyze
            epoch_step: Evaluate every N epochs
            subset_size: Optional subset size for faster evaluation

        Returns:
            Dictionary containing retention analysis results
        """
        # Setup W&B logging if requested
        wandb_run = None
        if wandb_project:
            try:
                wandb_run = setup_wandb_logging(
                    project_name=wandb_project,
                    run_name=wandb_run_name or f"memory_retention_{self.experiment_dir.name}",
                    config={
                        "experiment_dir": str(self.experiment_dir),
                        "reference_datasets": list(reference_datasets.keys()),
                        "batch_size": batch_size,
                        "max_epochs": max_epochs,
                        "epoch_step": epoch_step
                    },
                    tags=["memory_retention", "analysis"]
                )
                if wandb_run:
                    wandb.define_metric("epoch")
                    wandb.define_metric("*", step_metric="epoch")
            except Exception as e:
                print(f"Warning: Failed to initialize W&B: {e}")
                wandb_run = None

        # Get checkpoint paths
        checkpoint_paths = self.get_checkpoint_paths(max_epochs)
        if epoch_step > 1:
            checkpoint_paths = checkpoint_paths[::epoch_step]

        # Results storage
        all_results = {}

        # Evaluate pretrained model (epoch -1)
        pretrained_path = self.model_config.get("pretrained_path")
        if pretrained_path and os.path.exists(pretrained_path):
            print("Evaluating pretrained model (epoch -1)...")
            pretrained_model = MLIPModel(
                architecture=self.model_config["architecture"],
                model_path=pretrained_path
            )
            pretrained_model = pretrained_model.to(self.device)

            for dataset_name, dataset_path in reference_datasets.items():
                print(f"  Evaluating on {dataset_name}...")
                metrics = self.evaluate_model_on_dataset(
                    pretrained_model, dataset_path, batch_size, subset_size
                )

                if dataset_name not in all_results:
                    all_results[dataset_name] = {"epochs": [], "metrics": []}

                all_results[dataset_name]["epochs"].append(-1)
                all_results[dataset_name]["metrics"].append(metrics)

                # Log to W&B
                if wandb_run:
                    for key, value in metrics.items():
                        log_metrics({f"{dataset_name}/{key}": value, "epoch": -1})

        # Evaluate checkpoints
        for epoch, checkpoint_path in tqdm(checkpoint_paths, desc="Analyzing epochs"):
            print(f"\nEvaluating epoch {epoch}...")

            try:
                model = self.load_model_from_checkpoint(checkpoint_path)

                for dataset_name, dataset_path in reference_datasets.items():
                    metrics = self.evaluate_model_on_dataset(
                        model, dataset_path, batch_size, subset_size
                    )

                    if dataset_name not in all_results:
                        all_results[dataset_name] = {"epochs": [], "metrics": []}

                    all_results[dataset_name]["epochs"].append(epoch)
                    all_results[dataset_name]["metrics"].append(metrics)

                    # Log to W&B
                    if wandb_run:
                        for key, value in metrics.items():
                            log_metrics({f"{dataset_name}/{key}": value, "epoch": epoch})

            except Exception as e:
                print(f"Error evaluating epoch {epoch}: {e}")
                continue

        # Generate plots
        self._create_retention_plots(all_results, wandb_run)

        # Save results
        results_path = self.results_dir / "memory_retention_analysis.json"
        self._save_results(all_results, results_path)

        # Cleanup W&B
        if wandb_run:
            wandb.finish()

        return all_results

    def _create_retention_plots(self, all_results: Dict, wandb_run=None):
        """Create memory retention plots."""
        if not all_results:
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Memory Retention Analysis - {self.experiment_dir.name}", fontsize=16)

        # Color map for different datasets
        colors = plt.cm.Set1(np.linspace(0, 1, len(all_results)))
        dataset_colors = dict(zip(all_results.keys(), colors))

        # Plot 1: Energy MAE evolution
        ax = axes[0, 0]
        for dataset_name, results in all_results.items():
            epochs = results["epochs"]
            energy_mae = [m.get("energy_mae", 0) for m in results["metrics"]]
            if energy_mae and any(e > 0 for e in energy_mae):
                ax.plot(epochs, energy_mae, 'o-', label=dataset_name,
                       color=dataset_colors[dataset_name], alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Energy MAE (eV)")
        ax.set_title("Energy Error Evolution")
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Force MAE evolution
        ax = axes[0, 1]
        for dataset_name, results in all_results.items():
            epochs = results["epochs"]
            force_mae = [m.get("force_mae", 0) for m in results["metrics"]]
            if force_mae and any(f > 0 for f in force_mae):
                ax.plot(epochs, force_mae, 'o-', label=dataset_name,
                       color=dataset_colors[dataset_name], alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Force MAE (eV/Å)")
        ax.set_title("Force Error Evolution")
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Energy RMSE evolution
        ax = axes[0, 2]
        for dataset_name, results in all_results.items():
            epochs = results["epochs"]
            energy_rmse = [m.get("energy_rmse", 0) for m in results["metrics"]]
            if energy_rmse and any(e > 0 for e in energy_rmse):
                ax.plot(epochs, energy_rmse, 'o-', label=dataset_name,
                       color=dataset_colors[dataset_name], alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Energy RMSE (eV)")
        ax.set_title("Energy RMSE Evolution")
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Normalized energy evolution (forgetting)
        ax = axes[1, 0]
        for dataset_name, results in all_results.items():
            epochs = results["epochs"]
            energy_mae = [m.get("energy_mae", 0) for m in results["metrics"]]
            if len(energy_mae) > 0 and energy_mae[0] > 0:
                baseline = energy_mae[0]
                normalized = [e / baseline for e in energy_mae]
                ax.plot(epochs, normalized, 'o-', label=dataset_name,
                       color=dataset_colors[dataset_name], alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized Energy MAE")
        ax.set_title("Energy Forgetting (1 = pretrained)")
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: Normalized force evolution (forgetting)
        ax = axes[1, 1]
        for dataset_name, results in all_results.items():
            epochs = results["epochs"]
            force_mae = [m.get("force_mae", 0) for m in results["metrics"]]
            if len(force_mae) > 0 and force_mae[0] > 0:
                baseline = force_mae[0]
                normalized = [f / baseline for f in force_mae]
                ax.plot(epochs, normalized, 'o-', label=dataset_name,
                       color=dataset_colors[dataset_name], alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized Force MAE")
        ax.set_title("Force Forgetting (1 = pretrained)")
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Forgetting percentage
        ax = axes[1, 2]
        for dataset_name, results in all_results.items():
            epochs = results["epochs"]
            energy_mae = [m.get("energy_mae", 0) for m in results["metrics"]]
            if len(energy_mae) > 0 and energy_mae[0] > 0:
                baseline = energy_mae[0]
                forgetting_pct = [(e - baseline) / baseline * 100 for e in energy_mae]
                ax.plot(epochs, forgetting_pct, 'o-', label=dataset_name,
                       color=dataset_colors[dataset_name], alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Forgetting (%)")
        ax.set_title("Energy Forgetting Rate")
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.results_dir / f"memory_retention_{self.experiment_dir.name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved retention plot to {plot_path}")

        # Log to W&B
        if wandb_run:
            wandb.log({"memory_retention_analysis": wandb.Image(fig)})

        plt.close()

    def _save_results(self, results: Dict, output_path: Path):
        """Save results to JSON file."""
        import json

        # Convert results to JSON-serializable format
        json_results = {}
        for dataset_name, data in results.items():
            json_results[dataset_name] = {
                "epochs": data["epochs"],
                "metrics": []
            }
            for metrics in data["metrics"]:
                # Convert tensors to float
                json_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        json_metrics[key] = float(value.item())
                    else:
                        json_metrics[key] = float(value)
                json_results[dataset_name]["metrics"].append(json_metrics)

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Saved results to {output_path}")


def analyze_memory_retention(
    experiment_dir: Union[str, Path],
    reference_datasets: Dict[str, str],
    model_config: Dict,
    wandb_project: Optional[str] = None,
    device: str = "auto",
    **kwargs
) -> Dict[str, Dict]:
    """Convenience function for memory retention analysis.

    Args:
        experiment_dir: Path to experiment directory
        reference_datasets: Dict mapping dataset names to paths
        model_config: Model configuration
        wandb_project: Optional W&B project name
        device: Device to use for evaluation
        **kwargs: Additional arguments for analyzer

    Returns:
        Analysis results dictionary
    """
    analyzer = MemoryRetentionAnalyzer(experiment_dir, model_config, device)
    return analyzer.analyze_retention(
        reference_datasets=reference_datasets,
        wandb_project=wandb_project,
        **kwargs
    )