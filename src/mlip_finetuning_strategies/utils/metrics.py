"""Evaluation metrics for MLIP models."""

from typing import Dict, Any
import torch
import numpy as np


def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute evaluation metrics for MLIP predictions.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Dictionary of computed metrics
    """
    metrics = {}

    # Energy metrics
    if "energy" in predictions and "energy" in targets:
        energy_mae = compute_energy_mae(predictions["energy"], targets["energy"])
        metrics["energy_mae"] = energy_mae

        energy_rmse = compute_energy_rmse(predictions["energy"], targets["energy"])
        metrics["energy_rmse"] = energy_rmse

    # Force metrics
    if "forces" in predictions and "forces" in targets:
        force_mae = compute_force_mae(predictions["forces"], targets["forces"])
        metrics["force_mae"] = force_mae

        force_rmse = compute_force_rmse(predictions["forces"], targets["forces"])
        metrics["force_rmse"] = force_rmse

    # Stress metrics
    if "stress" in predictions and "stress" in targets:
        stress_mae = compute_stress_mae(predictions["stress"], targets["stress"])
        metrics["stress_mae"] = stress_mae

    return metrics


def compute_energy_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error for energy predictions.

    Args:
        pred: Predicted energies
        target: Target energies

    Returns:
        Energy MAE in eV
    """
    return torch.mean(torch.abs(pred - target)).item()


def compute_energy_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Root Mean Square Error for energy predictions.

    Args:
        pred: Predicted energies
        target: Target energies

    Returns:
        Energy RMSE in eV
    """
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def compute_force_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error for force predictions.

    Args:
        pred: Predicted forces [N_atoms, 3]
        target: Target forces [N_atoms, 3]

    Returns:
        Force MAE in eV/Å
    """
    return torch.mean(torch.abs(pred - target)).item()


def compute_force_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Root Mean Square Error for force predictions.

    Args:
        pred: Predicted forces [N_atoms, 3]
        target: Target forces [N_atoms, 3]

    Returns:
        Force RMSE in eV/Å
    """
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def compute_stress_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error for stress predictions.

    Args:
        pred: Predicted stress tensor
        target: Target stress tensor

    Returns:
        Stress MAE in eV/Å³
    """
    return torch.mean(torch.abs(pred - target)).item()


def compute_forgetting_rate(
    initial_metrics: Dict[str, float],
    final_metrics: Dict[str, float]
) -> Dict[str, float]:
    """Compute forgetting rate for each metric.

    Args:
        initial_metrics: Metrics before finetuning
        final_metrics: Metrics after finetuning

    Returns:
        Dictionary of forgetting rates
    """
    forgetting_rates = {}

    for metric_name in initial_metrics:
        if metric_name in final_metrics:
            initial_value = initial_metrics[metric_name]
            final_value = final_metrics[metric_name]

            if initial_value != 0:
                forgetting_rate = (final_value - initial_value) / initial_value
                forgetting_rates[f"{metric_name}_forgetting_rate"] = forgetting_rate

    return forgetting_rates