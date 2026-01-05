"""Naive finetuning strategy (baseline)."""

from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseStrategy


class NaiveStrategy(BaseStrategy):
    """Standard finetuning without any forgetting prevention.

    This serves as a baseline for comparison with other continual learning strategies.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """Initialize naive strategy.

        Args:
            model: The MLIP model to train
            config: Strategy configuration
        """
        super().__init__(model, config)

    def before_task(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """No special preparation needed for naive strategy."""
        pass

    def compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        task_id: int
    ) -> torch.Tensor:
        """Compute standard MSE loss for energy and forces.

        Args:
            model_output: Model predictions containing 'energy' and 'forces'
            target: Ground truth values containing 'energy' and 'forces'
            task_id: Task ID (unused in naive strategy)

        Returns:
            Combined loss value
        """
        total_loss = 0.0

        # Energy loss
        if 'energy' in model_output and 'energy' in target:
            energy_loss = F.mse_loss(model_output['energy'], target['energy'])
            total_loss += energy_loss * self.config.get('energy_weight', 1.0)

        # Force loss
        if 'forces' in model_output and 'forces' in target:
            force_loss = F.mse_loss(model_output['forces'], target['forces'])
            total_loss += force_loss * self.config.get('force_weight', 1.0)

        # Stress loss (optional)
        if 'stress' in model_output and 'stress' in target:
            stress_loss = F.mse_loss(model_output['stress'], target['stress'])
            total_loss += stress_loss * self.config.get('stress_weight', 0.1)

        return total_loss

    def after_task(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """No special cleanup needed for naive strategy."""
        pass

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return all model parameters.

        Returns:
            List of all model parameters
        """
        return list(self.model.parameters())