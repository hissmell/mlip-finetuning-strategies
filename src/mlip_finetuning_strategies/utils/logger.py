"""Logging utilities for experiment tracking."""

from typing import Dict, Any, Optional
import wandb
import json
from pathlib import Path


def setup_wandb_logging(
    project_name: str,
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None
) -> None:
    """Initialize Weights & Biases logging.

    Args:
        project_name: Name of the W&B project
        run_name: Name for this run
        config: Configuration dictionary to log
        tags: Tags for the run
    """
    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=tags
    )


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to W&B.

    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
    """
    wandb.log(metrics, step=step)


def save_metrics_to_file(
    metrics: Dict[str, Any],
    filepath: Path,
    append: bool = False
) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        filepath: Path to save file
        append: Whether to append to existing file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if append and filepath.exists():
        # Load existing metrics and append
        with open(filepath, 'r') as f:
            existing_metrics = json.load(f)
        if isinstance(existing_metrics, list):
            existing_metrics.append(metrics)
        else:
            existing_metrics = [existing_metrics, metrics]
        metrics_to_save = existing_metrics
    else:
        metrics_to_save = metrics

    with open(filepath, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)


def load_metrics_from_file(filepath: Path) -> Dict[str, Any]:
    """Load metrics from JSON file.

    Args:
        filepath: Path to metrics file

    Returns:
        Loaded metrics dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)