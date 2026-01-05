"""Progress & Compress (P&C) strategy."""

from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseStrategy


class PnCStrategy(BaseStrategy):
    """Progress & Compress strategy.

    Separates learning (active column) from knowledge preservation (knowledge base).
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """Initialize P&C strategy."""
        super().__init__(model, config)
        # TODO: Implement P&C-specific model modifications

    def before_task(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Freeze knowledge base, reset active column."""
        # TODO: Implement P&C task preparation
        pass

    def compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        task_id: int
    ) -> torch.Tensor:
        """Compute P&C-specific loss."""
        # TODO: Implement P&C loss computation
        total_loss = 0.0

        if 'energy' in model_output and 'energy' in target:
            energy_loss = F.mse_loss(model_output['energy'], target['energy'])
            total_loss += energy_loss * self.config.get('energy_weight', 1.0)

        if 'forces' in model_output and 'forces' in target:
            force_loss = F.mse_loss(model_output['forces'], target['forces'])
            total_loss += force_loss * self.config.get('force_weight', 1.0)

        return total_loss

    def after_task(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Distill active column into knowledge base."""
        # TODO: Implement knowledge distillation
        pass

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return active column parameters during progress phase."""
        # TODO: Return only active parameters during progress phase
        return list(self.model.parameters())