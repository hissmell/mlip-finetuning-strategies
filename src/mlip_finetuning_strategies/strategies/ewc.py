"""Elastic Weight Consolidation (EWC) strategy."""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseStrategy


class EWCStrategy(BaseStrategy):
    """Elastic Weight Consolidation strategy.

    Penalizes changes to important parameters using Fisher Information Matrix.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """Initialize EWC strategy.

        Args:
            model: The MLIP model to train
            config: Strategy configuration with EWC-specific parameters
        """
        super().__init__(model, config)
        self.lambda_ewc = config.get('lambda', 1000.0)
        self.fisher_samples = config.get('fisher_samples', 500)
        self.diagonal_only = config.get('diagonal_only', True)

        # Store optimal parameters and Fisher information
        self.optimal_params: Optional[Dict[str, torch.Tensor]] = None
        self.fisher_information: Optional[Dict[str, torch.Tensor]] = None

    def before_task(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Compute Fisher Information Matrix on previous task data."""
        if task_id > 0 and self.optimal_params is None:
            # Store current parameters as optimal for the previous task
            self.optimal_params = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.optimal_params[name] = param.data.clone()

            # Compute Fisher Information Matrix
            self._compute_fisher_information(dataloader)

    def compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        task_id: int
    ) -> torch.Tensor:
        """Compute MSE loss + EWC penalty.

        Args:
            model_output: Model predictions
            target: Ground truth values
            task_id: Task ID

        Returns:
            Total loss including EWC penalty
        """
        # Standard loss
        total_loss = 0.0

        # Energy loss
        if 'energy' in model_output and 'energy' in target:
            energy_loss = F.mse_loss(model_output['energy'], target['energy'])
            total_loss += energy_loss * self.config.get('energy_weight', 1.0)

        # Force loss
        if 'forces' in model_output and 'forces' in target:
            force_loss = F.mse_loss(model_output['forces'], target['forces'])
            total_loss += force_loss * self.config.get('force_weight', 1.0)

        # EWC penalty
        if task_id > 0 and self.optimal_params is not None and self.fisher_information is not None:
            ewc_penalty = self._compute_ewc_penalty()
            total_loss += self.lambda_ewc * ewc_penalty

        return total_loss

    def after_task(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Store optimal parameters after task completion."""
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return all model parameters."""
        return list(self.model.parameters())

    def _compute_fisher_information(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Compute Fisher Information Matrix."""
        # TODO: Implement Fisher Information computation
        # This is a placeholder - full implementation would compute gradients
        # on a subset of the previous task data
        self.fisher_information = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.ones_like(param.data)

    def _compute_ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty term."""
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                penalty += (fisher * (param - optimal) ** 2).sum()
        return penalty / 2.0