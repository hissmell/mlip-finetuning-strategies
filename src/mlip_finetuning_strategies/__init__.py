"""MLIP Finetuning Strategies

A systematic evaluation framework for continual learning strategies
in Machine Learning Interatomic Potentials (MLIP) finetuning.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .strategies import *
from .models import *
from .data import *
from .utils import *
from .analysis import *

__all__ = [
    "BaseStrategy",
    "NaiveStrategy",
    "EWCStrategy",
    "ReplayStrategy",
    "PnCStrategy",
    "MLIPModel",
    "compute_metrics",
    "setup_wandb_logging",
    "MemoryRetentionAnalyzer",
    "analyze_memory_retention",
]