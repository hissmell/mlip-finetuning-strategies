"""Main experiment runner for MLIP finetuning strategies."""

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
import wandb

from .models import MLIPModel
from .data import MLIPDataset, create_dataloader
from .strategies import BaseStrategy, NaiveStrategy, EWCStrategy, ReplayStrategy, PnCStrategy
from .utils import setup_wandb_logging, compute_metrics, save_metrics_to_file


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str = "auto") -> torch.device:
    """Get the appropriate device for training."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_arg)


def create_strategy(
    strategy_name: str,
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> BaseStrategy:
    """Create a strategy instance based on configuration."""
    strategy_config = config.get("strategy", {})

    if strategy_name.lower() == "naive":
        return NaiveStrategy(model, strategy_config)
    elif strategy_name.lower() == "ewc":
        return EWCStrategy(model, strategy_config)
    elif strategy_name.lower() == "replay":
        return ReplayStrategy(model, strategy_config)
    elif strategy_name.lower() == "pnc":
        return PnCStrategy(model, strategy_config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def setup_experiment_directory(config: Dict[str, Any]) -> Path:
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name = config.get("strategy", {}).get("name", "unknown")
    run_name = config.get("run_name", "experiment")

    exp_dir = Path(config["output_dir"]) / f"{timestamp}_{strategy_name}_{run_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    import yaml
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    return exp_dir


def run_finetuning_experiment(config: Dict[str, Any]) -> None:
    """Run a complete finetuning experiment."""
    # Set up environment
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))

    if config.get("verbose", False):
        print(f"Using device: {device}")
        print(f"Configuration: {config}")

    # Create experiment directory
    exp_dir = setup_experiment_directory(config)
    if config.get("verbose", False):
        print(f"Experiment directory: {exp_dir}")

    # Setup logging
    if config.get("wandb_project"):
        setup_wandb_logging(
            project_name=config["wandb_project"],
            run_name=config.get("run_name"),
            config=config
        )

    try:
        # Load model
        model_config = config.get("model", {})
        model = MLIPModel(
            architecture=model_config["architecture"],
            model_path=model_config.get("pretrained_path")
        )
        model = model.to(device)

        if config.get("verbose", False):
            print(f"Loaded model with {model.get_num_parameters()} parameters")

        # Load datasets
        data_config = config.get("data", {})

        # Pretrain dataset (for memory retention evaluation)
        if "pretrain_data" in data_config:
            pretrain_dataset = MLIPDataset(data_config["pretrain_data"])
            pretrain_dataloader = create_dataloader(
                pretrain_dataset,
                batch_size=data_config.get("batch_size", 32),
                shuffle=False
            )
        else:
            pretrain_dataloader = None

        # Finetune dataset
        finetune_dataset = MLIPDataset(data_config["finetune_data"])
        finetune_dataloader = create_dataloader(
            finetune_dataset,
            batch_size=data_config.get("batch_size", 32),
            shuffle=True
        )

        # Create strategy
        strategy_config = config.get("strategy", {})
        strategy = create_strategy(
            strategy_config["name"],
            model,
            strategy_config
        )

        if config.get("verbose", False):
            print(f"Using strategy: {strategy_config['name']}")

        # Evaluate initial performance
        if pretrain_dataloader:
            initial_metrics = evaluate_model(model, pretrain_dataloader, device)
            save_metrics_to_file(
                {"initial_memory_retention": initial_metrics},
                exp_dir / "results" / "initial_metrics.json"
            )
            if config.get("verbose", False):
                print(f"Initial metrics: {initial_metrics}")

        if config.get("dry_run", False):
            print("Dry run completed successfully")
            return

        # Run training
        train_strategy(
            strategy=strategy,
            train_dataloader=finetune_dataloader,
            eval_dataloader=pretrain_dataloader,
            device=device,
            config=config,
            exp_dir=exp_dir
        )

        # Final evaluation
        if pretrain_dataloader:
            final_metrics = evaluate_model(model, pretrain_dataloader, device)
            save_metrics_to_file(
                {"final_memory_retention": final_metrics},
                exp_dir / "results" / "final_metrics.json"
            )
            if config.get("verbose", False):
                print(f"Final metrics: {final_metrics}")

    except Exception as e:
        print(f"Error in experiment: {e}")
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()


def train_strategy(
    strategy: BaseStrategy,
    train_dataloader,
    eval_dataloader,
    device: torch.device,
    config: Dict[str, Any],
    exp_dir: Path
) -> None:
    """Train using the specified strategy."""
    training_config = config.get("training", {})
    epochs = training_config.get("epochs", 100)
    learning_rate = training_config.get("lr", 1e-4)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        strategy.get_trainable_parameters(),
        lr=learning_rate
    )

    # Training loop placeholder
    print(f"Starting training for {epochs} epochs...")

    # This is a simplified training loop
    # In practice, you would implement the full training logic
    for epoch in range(epochs):
        if config.get("verbose", False) and epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}")

    print("Training completed")


def evaluate_model(model, dataloader, device: torch.device) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    all_predictions = {}
    all_targets = {}

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
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
        if key in all_targets:
            all_targets[key] = torch.cat(all_targets[key])

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    model.train()
    return metrics