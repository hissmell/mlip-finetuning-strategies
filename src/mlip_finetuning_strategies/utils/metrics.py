"""Evaluation metrics for MLIP models."""

from typing import Dict, Any, Optional
import torch
import numpy as np


def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    batch_info: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, float]:
    """Compute evaluation metrics for MLIP predictions.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        batch_info: Additional batch information (e.g., number of atoms)

    Returns:
        Dictionary of computed metrics including required core metrics:
        - energy_mae: Total energy MAE
        - energy_mae_per_atom: Energy MAE per atom
        - force_mae: Force component MAE
        - force_cosine: Force cosine similarity
    """
    metrics = {}

    # Core Metric 1: Energy MAE (total energy)
    if "energy" in predictions and "energy" in targets:
        energy_mae = compute_energy_mae(predictions["energy"], targets["energy"])
        metrics["energy_mae"] = energy_mae

        energy_rmse = compute_energy_rmse(predictions["energy"], targets["energy"])
        metrics["energy_rmse"] = energy_rmse

        # Core Metric 2: Energy MAE per atom
        energy_mae_per_atom = compute_energy_mae_per_atom(
            predictions["energy"], targets["energy"], batch_info
        )
        if energy_mae_per_atom is not None:
            metrics["energy_mae_per_atom"] = energy_mae_per_atom

    # Core Metric 3: Force MAE (component-wise)
    if "forces" in predictions and "forces" in targets:
        force_mae = compute_force_mae(predictions["forces"], targets["forces"])
        metrics["force_mae"] = force_mae

        force_rmse = compute_force_rmse(predictions["forces"], targets["forces"])
        metrics["force_rmse"] = force_rmse

        # Core Metric 4: Force cosine similarity
        force_cosine = compute_force_cosine_similarity(
            predictions["forces"], targets["forces"]
        )
        if force_cosine is not None:
            metrics["force_cosine"] = force_cosine

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


def compute_energy_mae_per_atom(
    pred_energy: torch.Tensor,
    target_energy: torch.Tensor,
    batch_info: Optional[Dict[str, torch.Tensor]] = None
) -> Optional[float]:
    """Compute Mean Absolute Error for energy predictions per atom.

    Args:
        pred_energy: Predicted energies [batch_size]
        target_energy: Target energies [batch_size]
        batch_info: Batch information containing atom counts

    Returns:
        Energy MAE per atom in eV/atom, or None if atom count unavailable
    """
    if batch_info is None:
        return None

    # Try to get number of atoms per structure
    n_atoms = None
    if "num_atoms" in batch_info:
        n_atoms = batch_info["num_atoms"]
    elif "num_nodes" in batch_info:
        n_atoms = batch_info["num_nodes"]
    elif "batch" in batch_info:
        # Count atoms per batch from batch indices
        batch_idx = batch_info["batch"]
        n_atoms = torch.bincount(batch_idx.long())

    if n_atoms is None:
        return None

    # Compute per-atom energy errors
    energy_errors = torch.abs(pred_energy - target_energy)
    per_atom_errors = energy_errors / n_atoms

    return torch.mean(per_atom_errors).item()


def compute_force_cosine_similarity(
    pred_forces: torch.Tensor,
    target_forces: torch.Tensor
) -> Optional[float]:
    """Compute mean cosine similarity for force predictions.

    Args:
        pred_forces: Predicted forces [N_atoms, 3]
        target_forces: Target forces [N_atoms, 3]

    Returns:
        Mean cosine similarity (-1 to 1), or None if computation fails
    """
    if pred_forces.shape != target_forces.shape:
        return None

    # Flatten to [N_atoms, 3]
    pred_flat = pred_forces.reshape(-1, 3)
    target_flat = target_forces.reshape(-1, 3)

    # Compute cosine similarity for each atom
    cosine_similarities = []

    for pred_f, target_f in zip(pred_flat, target_flat):
        pred_norm = torch.norm(pred_f)
        target_norm = torch.norm(target_f)

        # Skip zero vectors
        if pred_norm < 1e-10 or target_norm < 1e-10:
            if pred_norm < 1e-10 and target_norm < 1e-10:
                cosine_similarities.append(1.0)  # Both zero vectors
            else:
                cosine_similarities.append(0.0)  # One is zero
        else:
            # Compute cosine similarity
            cosine_sim = torch.dot(pred_f, target_f) / (pred_norm * target_norm)
            cosine_similarities.append(cosine_sim.item())

    if not cosine_similarities:
        return None

    return float(np.mean(cosine_similarities))


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