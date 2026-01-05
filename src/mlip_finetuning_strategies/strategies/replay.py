"""Experience Replay strategy."""

from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseStrategy


class ReplayStrategy(BaseStrategy):
    """Experience Replay strategy.

    Retains subset of previous data and mixes with new data during training.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """Initialize Replay strategy."""
        super().__init__(model, config)
        self.buffer_size = config.get('buffer_size', 1000)
        self.replay_buffer = []

    def before_task(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Update replay buffer with samples from previous tasks."""
        # TODO: Implement replay buffer management
        pass

    def compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        task_id: int
    ) -> torch.Tensor:
        """Compute standard loss (replay handled in data loading)."""
        # TODO: Implement replay loss computation
        total_loss = 0.0

        if 'energy' in model_output and 'energy' in target:
            energy_loss = F.mse_loss(model_output['energy'], target['energy'])
            total_loss += energy_loss * self.config.get('energy_weight', 1.0)

        if 'forces' in model_output and 'forces' in target:
            force_loss = F.mse_loss(model_output['forces'], target['forces'])
            total_loss += force_loss * self.config.get('force_weight', 1.0)

        return total_loss

    def after_task(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Select samples for replay buffer."""
        # TODO: Implement sample selection for replay buffer
        pass

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return all model parameters."""
        return list(self.model.parameters())