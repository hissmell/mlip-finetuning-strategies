"""Base class for all continual learning strategies."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn


class BaseStrategy(ABC):
    """Base class for all continual learning strategies.

    All strategies inherit from this class to ensure consistent API.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """Initialize the strategy.

        Args:
            model: The MLIP model to train
            config: Strategy-specific configuration
        """
        self.model = model
        self.config = config

    @abstractmethod
    def before_task(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Called before training on a new task.

        Args:
            task_id: ID of the current task
            dataloader: DataLoader for the current task

        Examples:
            - EWC: Compute Fisher Information Matrix on previous task data
            - P&C: Freeze knowledge base, reset active column
            - Replay: Update replay buffer
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        task_id: int
    ) -> torch.Tensor:
        """Compute strategy-specific loss.

        Args:
            model_output: Model predictions (energy, forces, stress)
            target: Ground truth values
            task_id: ID of the current task

        Returns:
            Total loss value

        Examples:
            - Naive: Standard MSE loss
            - EWC: MSE + Fisher-weighted parameter penalty
            - P&C: MSE (progress) or KL divergence + EWC (compress)
        """
        pass

    @abstractmethod
    def after_task(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> None:
        """Called after completing training on a task.

        Args:
            task_id: ID of the completed task
            dataloader: DataLoader for the completed task

        Examples:
            - EWC: Store optimal parameters and Fisher diagonal
            - P&C: Distill active column into knowledge base
            - Replay: Select samples for replay buffer
        """
        pass

    @abstractmethod
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return parameters to be updated.

        Returns:
            List of parameters to optimize

        Examples:
            - P&C: Only active column parameters during progress phase
            - Others: All model parameters
        """
        pass

    def save_checkpoint(self, path: str, epoch: int, additional_data: Optional[Dict] = None) -> None:
        """Save strategy checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            additional_data: Strategy-specific data to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }
        if additional_data:
            checkpoint.update(additional_data)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load strategy checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Loaded checkpoint data
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint