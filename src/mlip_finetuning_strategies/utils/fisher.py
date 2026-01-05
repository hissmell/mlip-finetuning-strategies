"""Fisher Information Matrix computation for EWC."""

from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_fisher_information(
    model: nn.Module,
    dataloader: DataLoader,
    num_samples: int = 500,
    diagonal_only: bool = True
) -> Dict[str, torch.Tensor]:
    """Compute Fisher Information Matrix for model parameters.

    Args:
        model: The neural network model
        dataloader: DataLoader for computing Fisher information
        num_samples: Number of samples to use for computation
        diagonal_only: Whether to compute only diagonal elements

    Returns:
        Dictionary mapping parameter names to Fisher information tensors
    """
    model.eval()
    fisher_information = {}

    # Initialize Fisher information storage
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_information[name] = torch.zeros_like(param.data)

    num_batches = min(num_samples // dataloader.batch_size + 1, len(dataloader))
    samples_processed = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        # Zero gradients
        model.zero_grad()

        # Forward pass
        try:
            output = model(batch)

            # Compute log-likelihood loss (approximate)
            if isinstance(output, dict) and 'energy' in output:
                # For energy predictions, use negative log-likelihood approximation
                loss = torch.sum(output['energy'] ** 2) / 2.0
            else:
                # Fallback: use sum of squared outputs
                if isinstance(output, torch.Tensor):
                    loss = torch.sum(output ** 2) / 2.0
                else:
                    loss = torch.sum(torch.cat([v.flatten() for v in output.values()]) ** 2) / 2.0

            # Backward pass to compute gradients
            loss.backward()

            # Accumulate squared gradients (Fisher information approximation)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if diagonal_only:
                        # Only diagonal elements
                        fisher_information[name] += param.grad.data ** 2
                    else:
                        # Full matrix (not implemented for memory efficiency)
                        fisher_information[name] += param.grad.data ** 2

            samples_processed += batch['energy'].size(0) if 'energy' in batch else 1

        except Exception as e:
            print(f"Warning: Error processing batch {batch_idx}: {e}")
            continue

    # Normalize by number of samples
    if samples_processed > 0:
        for name in fisher_information:
            fisher_information[name] /= samples_processed

    model.train()
    return fisher_information


def regularize_fisher_information(
    fisher_info: Dict[str, torch.Tensor],
    epsilon: float = 1e-8
) -> Dict[str, torch.Tensor]:
    """Add regularization to Fisher information to avoid numerical issues.

    Args:
        fisher_info: Fisher information dictionary
        epsilon: Regularization constant

    Returns:
        Regularized Fisher information
    """
    regularized = {}
    for name, fisher in fisher_info.items():
        regularized[name] = fisher + epsilon
    return regularized


def compute_ewc_penalty(
    model: nn.Module,
    optimal_params: Dict[str, torch.Tensor],
    fisher_info: Dict[str, torch.Tensor],
    lambda_ewc: float = 1000.0
) -> torch.Tensor:
    """Compute EWC penalty term.

    Args:
        model: Current model
        optimal_params: Optimal parameters from previous task
        fisher_info: Fisher information matrix
        lambda_ewc: EWC regularization strength

    Returns:
        EWC penalty value
    """
    penalty = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad and name in optimal_params and name in fisher_info:
            optimal = optimal_params[name]
            fisher = fisher_info[name]
            penalty += (fisher * (param - optimal) ** 2).sum()

    return lambda_ewc * penalty / 2.0